from typing import Callable

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence as pseq
from torch.nn.functional import conv2d
from torchvision.transforms.functional import gaussian_blur
import cv2 as cv
from facenet_pytorch import MTCNN

try:
    from extpt.constants import DEVICE, WIDTH, HEIGHT
    from extpt.resnet.resnet import SimpleCNN
    from extpt.jepa.pos_embs import get_1d_sincos_pos_embed
    from extpt.open_clip.transformer import Transformer
except ImportError:
    from constants import DEVICE, WIDTH, HEIGHT
    from resnet.resnet import SimpleCNN
    from jepa.pos_embs import get_1d_sincos_pos_embed
    from open_clip.transformer import Transformer


class CropFace(Callable):
    def __init__(self, size, margin, post_process=False):
        self.size = size
        self.margin = margin
        self.mtcnn = MTCNN(image_size=size, margin=margin, post_process=post_process)
    
    def __call__(self, image: torch.Tensor):
        face = self.mtcnn(image)
        return face


# TODO: unit tests for both of these roi functions
def get_roi_presets(h, w):
    roi_presets = {
        "forehead": {"roi_topleft_y": int(h * 0), "roi_topleft_x": int(w * 0), "roi_height": int(h * 0.2), "roi_width": int(w * 1)},
        "eyebrows": {"roi_topleft_y": int(h * 0.2), "roi_topleft_x": int(w * 0.1), "roi_height": int(h * 0.2), "roi_width": int(w * 0.8)},
        "eyes_nose": {"roi_topleft_y": int(h * 0.35), "roi_topleft_x": int(w * 0.05), "roi_height": int(h * 0.25), "roi_width": int(w * 0.9)},
        "nose_mouth": {"roi_topleft_y": int(h * 0.55), "roi_topleft_x": int(w * 0.1), "roi_height": int(h * 0.25), "roi_width": int(w * 0.8)},
        "close_face": {"roi_topleft_y": int(h * 0.3), "roi_topleft_x": int(w * 0.2), "roi_height": int(h * 0.6), "roi_width": int(w * 0.6)},
    }
    return roi_presets


def get_roi_patch(feature_map_batch, roi_topleft_y, roi_topleft_x, roi_height, roi_width):
    *_, height, width = feature_map_batch.shape
    roi_width_end = roi_topleft_x + roi_width
    roi_height_end = roi_topleft_y + roi_height
    assert roi_topleft_x >= 0 and roi_topleft_y >= 0, f"Top left co-ordinate of requested patch out of image bounds: ({roi_topleft_x}, {roi_topleft_y})"
    assert roi_topleft_x <= width and roi_topleft_y <= height, f"Top left co-ordinate of requested patch out of image bounds[{width}, {height}]: ({roi_topleft_x}, {roi_topleft_y})"
    assert roi_width_end <= width, f"Requested patch width end {roi_width_end} exceeds width of image {width}."
    assert roi_height_end <= height, f"Requested patch height end {roi_height_end} exceeds height of image {height}."

    roi = feature_map_batch[... , roi_topleft_y : roi_height_end, roi_topleft_x : roi_width_end]
    return roi
    

class VideoTransformer(nn.Module):
    def __init__(self, in_channels=64, in_width=109, in_height=109, embed_size=768, num_layers=7, num_heads=8):
        super().__init__()
        self.presets  = get_roi_presets(in_height, in_width)
        self.scnn = SimpleCNN(in_channels)
        # self.simple_tfm = Transformer(1536, 12, 8)
        self.simple_tfm = Transformer(embed_size, num_layers, num_heads)

        # batch = torch.flatten(clip0_rgb_roi_close_face, start_dim=0, end_dim=1)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, 80 + 1, embed_size))
        sincos1d = get_1d_sincos_pos_embed(embed_size, 80, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(sincos1d))

    
    def forward(self, v):
        B, *_ = v.shape
        v_roi_close_face = get_roi_patch(v, **self.presets["close_face"])
        v_roi_forehead = get_roi_patch(v, **self.presets["forehead"])
        v_roi_eyebrows = get_roi_patch(v, **self.presets["eyebrows"])
        v_roi_eyes_nose = get_roi_patch(v, **self.presets["eyes_nose"])
        v_roi_nose_mouth = get_roi_patch(v, **self.presets["nose_mouth"])


        r1 = self.scnn(v_roi_forehead).permute(2, 1, 0)
        r2 = self.scnn(v_roi_eyebrows).permute(2, 1, 0)
        r3 = self.scnn(v_roi_close_face).permute(2, 1, 0)
        r4 = self.scnn(v_roi_eyes_nose).permute(2, 1, 0)
        r5 = self.scnn(v_roi_nose_mouth).permute(2, 1, 0)

        token_batch = pseq([r1, r2, r3, r4, r5]).permute(3, 2, 1, 0)
        token_batch = token_batch.flatten(start_dim=1, end_dim=2)
        cls_tokens = self.cls_token.expand((B, -1, -1))
        token_batch = torch.cat((cls_tokens, token_batch), dim=1)

        # print(token_batch.shape)# each one of the 80 tokens represents a single region in a single frame in a single video
                        # each video therefore has 80 tokens representing both facial and temporal regions of the 
                        # input 
        inp = token_batch + self.pos_embed
        tfm_out = self.simple_tfm(inp)
        return tfm_out


class EdgeDetector(Callable):

    @staticmethod
    def make_kernel(size):

        pass

    def __init__(self, kernel_sz):
        self.kernel_sz = kernel_sz
        self.kernel_5 = torch.FloatTensor([
            [-1, -1, -1, -1, -1],
            [-1,  1,  2,  1, -1],
            [-1,  2,  4,  2, -1],
            [-1,  1,  2,  1, -1],
            [-1, -1, -1, -1, -1]
        ]).repeat(3, 3, 1, 1)
        self.kernel_3 = torch.FloatTensor([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1],
        ]).repeat(3, 3, 1, 1)

    def __call__(self, image: torch.Tensor):
        lowpass = gaussian_blur(image, self.kernel_sz)
        edges = conv2d(image, weight=self.kernel_5, padding="same")
        guassian_high_pass = edges - lowpass
        return guassian_high_pass