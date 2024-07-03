from typing import List
from pathlib import Path
from tqdm import tqdm

from ffprobe import FFProbe
from ffprobe.exceptions import FFProbeError


def get_media_info(path: Path) -> float:
    metadata = FFProbe(str(path))
    l = None
    found_video = False
    found_audio = False
    for stream in metadata.streams:
        if stream.is_video():
            l = stream.duration_seconds()
            found_video = l is not None
            frames = stream.nb_frames
        elif stream.is_audio():
            try:
                l = stream.duration_seconds()
                found_audio = l is not None
                frames = 0
            except FFProbeError as e:
                found_audio = None

    assert found_audio or found_video, f"No audio or video streams found in {path.name}"
    return l, frames, metadata


def perform_diagnostic(dataset_root: Path, file_ext: str="avi"):
    shortest_vid, shortest_vid_len = None, float("inf")
    longest_vid, longest_vid_len = None, 0
    total_runtime = 0 
    total_count = 0
    for p in tqdm(list(dataset_root.rglob(f"*.{file_ext}"))):
        try:
            length, frames, metadata = get_media_info(str(p))
            if length >= longest_vid_len:
                longest_vid_len = length
                longest_vid = p.name
            if length <= shortest_vid_len:
                shortest_vid_len = length
                shortest_vid = p.name
            total_runtime += length
            total_count += 1
        except AttributeError:
            continue
                
    print(metadata)
    print(f"Longest clip: {longest_vid} {longest_vid_len}s\nShortest clip: {shortest_vid} {shortest_vid_len}s\nAverage clip length: {total_runtime/total_count:.2f}")


if __name__ == "__main__":
    from enterface import DATA_DIR as DATA_DIR_ENTERFACE
    from asvp import DATA_DIR as DATA_DIR_ASVP
    from librispeech import DATA_DIR as DATA_DIR_LIBRISPEECH
    from tess import DATA_DIR as DATA_DIR_TESS
    from mspodcast import DATA_DIR as DATA_DIR_MSPODCAST


    # perform_diagnostic(Path(DATA_DIR_ENTERFACE), "avi")
    # perform_diagnostic(Path(DATA_DIR_ASVP), "wav")
    # perform_diagnostic(Path(DATA_DIR_LIBRISPEECH), "wav")
    # perform_diagnostic(Path(DATA_DIR_TESS), "wav")
    perform_diagnostic(Path(DATA_DIR_MSPODCAST), "wav")