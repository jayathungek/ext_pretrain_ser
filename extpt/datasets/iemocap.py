from pathlib import Path
from typing import List
import random
try:
    from extpt.datasets.utils import get_label_colours
except ModuleNotFoundError:
    from utils import get_label_colours
    

NAME = "iemocap"
MULTILABEL = False
# MODALITIES = ["audio", "video"]
MODALITIES = ["audio"]
HEADERS = ["filepath", "duration", "frames", "emo_label", "speaker_id", "session_id", "is_improv", "valence", "activation", "dominance"]
LABEL_MAPPINGS = ["anger", "frustration", "excitement", "happiness", "sadness", "neutral"]
NUM_LABELS = len(LABEL_MAPPINGS)
LABEL_COLORS = get_label_colours(NUM_LABELS)
MAX_SPEC_SEQ_LEN = 125
SPEC_MAX_LEN = 700
SAMPLING_RATE = 16000
SPEC_WINDOW_SZ_MS = 21
SPEC_HOP_LEN_MS   = 10
MAX_AUDIO_TIME_SEC = 7
DATA_DIR = "/root/intelpa-2/datasets/iemocap_dataset/IEMOCAP_full_release"
LABELS_PATH = "sentences/EmoEvaluation"
MANIFEST = f"{NAME}_balanced.csv"
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
# need labels for: M/F, Session num, improv/script, emotion
def manifest_fn(file: Path) -> List[List]:
    mapping = {
        "ang": 0,
        "fru": 1,
        "exc": 2,
        "hap": 3,
        "sad": 4,
        "neu": 5
    }
   
    invalid = False 
    tags = file.stem.split("_")
    session = int(tags[0][3:5]) 
    improv = tags[1].startswith("impro")

    if len(tags) == 4:
        labels_filename = "_".join(tags[:3])
        gender = tags[3][0]
    else:
        labels_filename = "_".join(tags[:2])
        gender = tags[2][0]

    labels_path = f"{DATA_DIR}/Session{session}/{LABELS_PATH}/{labels_filename}.txt"
    with open(labels_path, "r") as fh:
        all_turns = [l.split("	") for l in fh.readlines() if file.stem in l]
    assert len(all_turns) == 1, f"could not find filename {file.stem}"
    labels = [item.strip() for item in all_turns[0]][2:]
    emo_label = labels[0]
    if emo_label == "xxx":
        invalid = True
    if emo_label == "fru" and random.random() < 1/2:
        # remove half of frustration
        invalid = True
    if emo_label == "neu" and random.random() < 1/3:
        # remove a third of neutral
        invalid = True
    attr_labels = labels[1][1:-1].split(", ")
    valence = float(attr_labels[0])
    activation = float(attr_labels[1])
    dominance = float(attr_labels[1])
    full_path = str(file.resolve())
    # print(full_path, emo_label, gender, session, improv, valence, activation, dominance)

    if not invalid:
        return {
            "filepath": full_path, 
            "label": mapping[emo_label], 
            "gender": gender,
            "session": session,
            "improv": improv ,
            "valence": valence,
            "activation": activation,
            "dominance": dominance
        }
    else:
        raise ValueError("Invalid emotion (xxx), skip this file")