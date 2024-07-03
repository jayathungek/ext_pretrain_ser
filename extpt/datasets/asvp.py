from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple
try:
    from extpt.datasets.utils import get_label_colours
except ModuleNotFoundError:
    from utils import get_label_colours


NAME = "asvp"
MULTILABEL = False
MODALITIES = ["audio"]

LABEL_MAPPINGS = ["boredom", "neutral", "happy", "sad", "angry", "fear", "disgust", "surprise", "excited", "pleased", "hurt", "disappointment", "breath"]
HEADERS = ["filepath", "duration", "frames", "emo_label", "speaker_id", "gender", "language"]
NUM_LABELS = len(LABEL_MAPPINGS)
LABEL_COLORS = get_label_colours(NUM_LABELS)
MAX_SPEC_SEQ_LEN = 125
SPEC_MAX_LEN = 60
SAMPLING_RATE = 16000
SPEC_WINDOW_SZ_MS = 10
SPEC_HOP_LEN_MS   = 5
MAX_AUDIO_TIME_SEC = 5
DATA_DIR = "/root/intelpa-1/datasets/asvpd_esd/ASVP-ESD-Update/Audio"
MANIFEST = f"{NAME}.csv"
GPU_DATASET = f"{NAME}_gpu.ds"
ADSMOTE_FEATURES = f"{NAME}_pruned_adsmote.fs"
DATA_EXT = "wav"
FORCE_AUDIO_ASPECT = False


# AUDIO
ADSMOTE_GAMMA = 0.25
ADSMOTE_KNNS = 10
ADSMOTE_POLY_SAMPLES = 2


# first 2 items in tuple is filepath, length(s), rest are labels
# Filename identifiers:
# Modality ( 03 = audio-only).
# Vocal channel (01 = speech, 02 = non speech).
# Emotion ( 01 = boredom,sigh| 02 = neutral,calm| 03 = happy, laugh,gaggle|04 = sad,cry | 05 = angry,grunt,frustration|06 = fearful,scream,panic| 07 = disgust, dislike,contempt|08 = surprised,gasp,amazed| 09 = excited| 10 = pleasure, 11 = pain,groan| 12 = disappointmen,disapproval| 13=breath).
# Emotional intensity (01 = normal, 02 = high).
# Statement (as it’s non scripted this help to refer approximately to data collected from the same period or source base on their rank ).
# Actor ( even numbered acteurs are male, odd numbered actors are female).
# Age(01 = above 65, 02 = between 20~64, 03 = under 20,04= baby).
# Source of downloading (01 -02 = =website ,youtube channel| 03= movies).
# Language(01=Chinese , 02=English ,04 = French , others:Russian and Others ).
# Filename example: 03-01-06-01-02-12-02-01-01-16.wav:
# 
# 1.audio-only (03)
# 2.Speech (01)
# 3.Fearful (06)
# 4.Normal intensity (01)
# 5.Statement (02)
# 6.12th Actorr (12) folder 12 male as its even
# 7.Age(02)
# 8.Source(01)
# 9.language(01)
# 10.similarity with others emotion/sound (16)
#
# All audio file with 77 at the end means files with a high noise environment.
# audio with 66 at the end means mixed voices(there is a limited number of free download on online platform, downloading more will come with mixed voice that can affect the sound)
# for non-speech data:
# Happyness is a collection of (laugh=13,gaggle=23,others=33)
# sadness is a collection of (cry=14, sigh=24,sniffle=34,suffering=44)
# fear is a collection of (scream 16,panic=36)
# angry (rage=15,frustration=25 ,other=35,grunt)
# surprise (surprised=18, amazed=28 ,astonishment=38,others=48)
# disgust(disgust=17, rejection=27)
# pain(moaned)
# boredom(sigh)
def manifest_fn(file: Path) -> List[List]:
    import torchaudio

    disqualified = False
    tags = file.stem.split("-")
    label = int(tags[2]) - 1
    actor_id = int(tags[5])
    gender = 'M' if actor_id % 2 == 0 else 'F'
    language = int(tags[8])
    noise = tags[-1]
    full_path = str(file.resolve())
    audio_samples, _ = torchaudio.load(full_path)
    
    if noise == "77" or noise == "66":
        disqualified = True
    
    if audio_samples.max() == audio_samples.min():
        #flat signal, discard
        disqualified = True
    
    # if language != 2:
    #     disqualified = True
        
    if not disqualified: 
        return {
            "filepath": full_path, 
            "label": label, 
            "actor_id": actor_id,
            "gender": gender,
            "language": language
        }
    else:
        raise ValueError("Audio disqualified, see code above")