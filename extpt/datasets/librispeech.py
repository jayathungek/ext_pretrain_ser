from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple
try:
    from extpt.datasets.utils import get_label_colours
except ModuleNotFoundError:
    from utils import get_label_colours


NAME = "librispeech"
MULTILABEL = False
MODALITIES = ["audio"]

LABEL_MAPPINGS = ["neutral"]
NUM_LABELS = len(LABEL_MAPPINGS)
LABEL_COLORS = get_label_colours(NUM_LABELS)
MAX_SPEC_SEQ_LEN = 125
SPEC_MAX_LEN = 1200
SAMPLING_RATE = 16000
SPEC_WINDOW_SZ_MS = 10
SPEC_HOP_LEN_MS   = 5
MAX_AUDIO_TIME_SEC = 5
DATA_DIR = "/root/intelpa-1/datasets/LibriSpeech_dataset/train/wav"
MANIFEST = f"{NAME}.csv"
GPU_DATASET = f"{NAME}_gpu.ds"
ADSMOTE_FEATURES = f"{NAME}_pruned_adsmote.fs"
DATA_EXT = "wav"
FORCE_AUDIO_ASPECT = False


# AUDIO
ADSMOTE_GAMMA = 0.25
ADSMOTE_KNNS = 10
ADSMOTE_POLY_SAMPLES = 2


def manifest_fn(file: Path) -> List[List]:
    import torchaudio

    disqualified = False
    label = 0 
    full_path = str(file.resolve())
    audio_samples, _ = torchaudio.load(full_path)
    
    
    if audio_samples.max() == audio_samples.min():
        #flat signal, discard
        disqualified = True
        
    if not disqualified: 
        return {
            "filepath": full_path, 
            "label": label, 
            "speaker_id": 0,
        }
    else:
        raise ValueError("Audio disqualified, see code above")