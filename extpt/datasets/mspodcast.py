from pathlib import Path
from typing import List, Tuple
try:
    from extpt.datasets.utils import get_label_colours
except ModuleNotFoundError:
    from utils import get_label_colours
    

NAME = "mspodcast"
MULTILABEL = False
MODALITIES = ["audio"]

LABEL_MAPPINGS = ["angry", "sad", "happy", "surprise", "fear", "disgust", "contempt", "neutral", "other"]
HEADERS = ["filepath", "duration", "frames", "emo_label", "act", "val", "dom", "speaker_id", "gender", "split_set"]
NUM_LABELS = len(LABEL_MAPPINGS)
LABEL_COLORS = get_label_colours(NUM_LABELS)
MAX_SPEC_SEQ_LEN = 125
SPEC_MAX_LEN = 700
SAMPLING_RATE = 16000
SPEC_WINDOW_SZ_MS = 21
SPEC_HOP_LEN_MS   = 10
MAX_AUDIO_TIME_SEC = 3
DATA_DIR = "/root/intelpa-1/datasets/msp_podcast_dataset"
MANIFEST = f"{NAME}.csv"
MANIFEST_TEST = f"{NAME}_test1.csv"
GPU_DATASET = f"{NAME}_gpu.ds"
ADSMOTE_FEATURES = f"{NAME}_pruned_adsmote.fs"
FACE_MARGIN = 50
DATA_EXT = "wav"
FORCE_AUDIO_ASPECT = False
VIDEO_FPS = 25

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
    mapping = {
        "angry": 0,
        "fear": 1,
        "ps": 2,
        "disgust": 3,
        "happy": 4,
        "sad": 5,
        "neutral": 6
    }
    
    tags = file.stem.split("_")
    label = mapping[tags[2]]
    actor_id = tags[0][0]
    full_path = str(file.resolve())
    return {
        "filepath": full_path, 
        "label": label, 
        "actor_id": actor_id,
    }