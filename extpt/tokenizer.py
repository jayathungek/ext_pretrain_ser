import io
import math
from pathlib import Path
from typing import Optional, Callable, Union, List

import ffmpeg
import numpy as np
import torch
from torch import nn
from torchvision.utils import save_image
import torchaudio
from torchvision import transforms
from PIL import Image
# from timm.layers.format import Format, nchw_to
# from timm.layers.helpers import to_2tuple

try:
    from extpt.face import CropFace, EdgeDetector
    from extpt.constants import *
    from extpt.datasets import enterface
    from extpt.open_clip.utils import to_2tuple
except ModuleNotFoundError:
    from face import CropFace, EdgeDetector
    from constants import *
    from datasets import enterface
    from open_clip.utils import to_2tuple



class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(
            self,
            img_size: Union[int, tuple] = 224,
            patch_size: int = 16,
            num_frames: int = 10,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            output_fmt: Optional[str] = None,
            bias: bool = True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        # if output_fmt is not None:
        #     self.flatten = False
        #     self.output_fmt = Format(output_fmt)
        # else:
        #     # flatten spatial dim and transpose to channels last, kept for bwd compat
        #     self.flatten = flatten
        #     self.output_fmt = Format.NCHW

        self.proj = nn.Conv2d(in_chans * num_frames, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
        assert W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        # elif self.output_fmt != Format.NCHW:
        #     x = nchw_to(x, self.output_fmt)
        x = self.norm(x)
        return x


def normalize(arr: np.ndarray):
    return arr / arr.max()

    
def get_rgb_frames(video_path: str, video_length: float, ensure_frames_len: int=None, video_fps: int=None) -> List:
    assert (ensure_frames_len is None) ^ (video_fps is None), "Either ensure_frames_len or video_fps must be set (XOR)"
    min_resize = 256
    new_width = "(iw/min(iw,ih))*{}".format(min_resize)
    fps = math.ceil(FRAMES/video_length) if ensure_frames_len else video_fps
    cmd = (
        ffmpeg
        .input(video_path)
        .trim(start=0, end=10)
        .filter("fps", fps=fps)
        .filter("scale", new_width, -1)
        .output("pipe:", format="image2pipe")
    )
    jpeg_bytes, _ = cmd.run(capture_stdout=True, quiet=True)
    jpeg_bytes = jpeg_bytes.split(JPEG_HEADER)[1:]
    jpeg_bytes = map(lambda x: JPEG_HEADER + x, jpeg_bytes)
    all_frames = list(jpeg_bytes)
    if ensure_frames_len is not None:
        if len(all_frames) > ensure_frames_len:
            all_frames = all_frames[:ensure_frames_len]
        elif len(all_frames) < ensure_frames_len:
            # repeat last frame by the difference between target length and actual length
            diff = ensure_frames_len - len(all_frames)
            all_frames = all_frames + [all_frames[-1] for _ in range(diff)]
            if DEBUG:
                print(f"get_rgb_frames: appended {diff} repeated frames to {video_path}")
    return all_frames


def make_audio_input(file):
    samples, _ = torchaudio.load(file, normalize=True)
    samples = samples[0] # use only 1 channel, for simplicity
    return samples.squeeze(0)


def save_video_frames_bytes(frames: List[bytearray], save_dir: str):
    newdir = Path(save_dir)
    newdir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        with open(newdir / f"{i}.jpeg", "wb") as fh:
            frame = frame / 255.0
            fh.write(frame)

def save_video_frames_tensors(frames: torch.tensor, save_dir: str, rows: int=2):
    save_image(frames, fp=f"{save_dir}.jpg", nrow=rows)


def  frame_offset_to_timestamp(frame_offset: int, fps: int) -> float:
    return frame_offset * (1 / fps)


_rgb_transform = transforms.Compose([
    # CropFace(size=WIDTH, margin=enterface.FACE_MARGIN),
    transforms.ToTensor(),
    transforms.Resize((WIDTH, HEIGHT)),
    EdgeDetector(kernel_sz=GAUSS_BLUR_KERNEL_SZ) if EDGE_DETECTION else nn.Identity()
])

def make_rgb_input(file: str, video_len: float, constant_fps: bool) -> np.ndarray:
    frames = get_rgb_frames(file,
                            video_len,
                            ensure_frames_len=None if constant_fps else FRAMES,
                            video_fps=enterface.VIDEO_FPS if constant_fps else None)
    tensor_frames = []
    for vid_frame in frames:
        tfmed = _rgb_transform(Image.open(io.BytesIO(vid_frame))) 
        tensor_frames.append(tfmed)
        
    stacked_frames = np.stack(tensor_frames, axis=0)
    if DEBUG:
        pfile = Path(file)
        save_video_frames_tensors(stacked_frames, f"{pfile.stem}")
    return stacked_frames


def make_input(file: str, video_len: float, modalites: List[str], constant_fps: bool=False):
    all_audio_samples = None
    all_video_frames = None
    if constant_fps:
        for m in modalites:
            if m == "audio":
                all_audio_samples = make_audio_input(file).float()
            elif m == "video":
                all_video_frames = make_rgb_input(file, video_len, constant_fps)
                all_video_frames = normalize(torch.tensor(all_video_frames)).float()

        return all_video_frames, all_audio_samples
    else:
        # rgb_norm, spec_norm = normalize(np.array(self.make_rgb_input(file, video_len, constant_fps))), normalize(self.make_mfcc_input(file, sampling_rate))
        # return torch.from_numpy(rgb_norm).float(), torch.from_numpy(spec_norm).float()
        raise NotImplementedError()
