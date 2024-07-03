from pathlib import Path
from typing import List, Tuple
try:
    from extpt.datasets.utils import get_label_colours
except ModuleNotFoundError:
    from utils import get_label_colours
    
# Max duration: 5.27s, Min duration: 2.94s, Avg duration: 3.7s
NAME = "ravdess"
MULTILABEL = False
MODALITIES = ["audio", "video"]

LABEL_MAPPINGS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"] 
HEADERS = ["filepath", "duration", "frames", "emo_label", "speaker_id"]
NUM_LABELS = len(LABEL_MAPPINGS)
LABEL_COLORS = get_label_colours(NUM_LABELS)
MAX_SPEC_SEQ_LEN = 125
SPEC_MAX_LEN = 500
SAMPLING_RATE = 48000
SPEC_WINDOW_SZ_MS = 21
SPEC_HOP_LEN_MS   = 10
MAX_AUDIO_TIME_SEC = 3
DATA_DIR = "/root/intelpa-1/datasets/ravdess"
MANIFEST = f"{NAME}.csv"
GPU_DATASET = f"{NAME}_gpu.ds"
ADSMOTE_FEATURES = f"{NAME}_pruned_adsmote.fs"
FACE_MARGIN = 50
DATA_EXT = "mp4"
FORCE_AUDIO_ASPECT = False
VIDEO_FPS = 30

# AUGMENTATION PARAMS
# VIDEO
ROT_DEGREES = 10
FLIP_CHANCE = 0.2
INVERT_CHANCE = 0

# AUDIO
ADSMOTE_GAMMA = 0.25
ADSMOTE_KNNS = 10
ADSMOTE_POLY_SAMPLES = 2


# first 2 items in tuple is filepath, length(s), rest are labels
def manifest_fn(file: Path) -> List[List]:
    tags = file.stem.split("-")
    label = int(tags[2]) - 1
    actor_id = int(tags[6]) - 1
    full_path = str(file.resolve())
    return {
        "filepath": full_path, 
        "label": label, 
        "actor_id": actor_id,
    }