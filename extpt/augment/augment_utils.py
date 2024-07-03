import math
from math import dist
from pathlib import Path
import multiprocessing as mp
from dataclasses import dataclass
import random
from functools import partial
from typing import List, Tuple  
import joblib

import torch
import numpy as np
from tqdm import tqdm
import torchyin
import librosa


@dataclass
class WavFeatures:
    name: str
    f0: float  # fundamental frequency of the signal
    rms: float  # root mean square of the signal


def get_field_index(header: List[str], to_find: str) -> int:
    idx = 0
    for item in header:
        if item == to_find:
            return idx
        idx += 1

    assert False, f"{to_find} does not exist in the header: {header}"


def map_feature_space(flist: str, sampling_rate: int, has_header=False, delim=",") -> List[WavFeatures]:
    filepaths = []
    with open(flist, "r") as fh:
        idx = 0
        header = None
        filepath_idx = 0
        for line in fh:
            line = line.strip()
            if idx == 0 and has_header:
                header = line
                filepath_idx = get_field_index(header, "filepath")
            else:
                filepath = line.split(delim)[filepath_idx]
                filepath = Path(filepath).resolve()
                filepaths.append(filepath)
            idx += 1
    cpus = mp.cpu_count()
    pool = mp.Pool(cpus)
    results = list(tqdm(pool.imap_unordered(partial(get_features, sampling_rate=sampling_rate), filepaths), total=len(filepaths)))
    pool.close()
    pool.join()
    failed_files = [r[1].name for r in results if not r[0]]
    if len(failed_files) > 0:
        print(f"failed to extract features from {len(failed_files)} files:")
        for name in failed_files:
            print(name)
    return [w for succ, w in results if succ], failed_files


def get_f0(signal, sampling_rate):
    # fmin: lowest male voice, fmax: highest female voice
    # f0, voiced_flag, voiced_probs = librosa.pyin(signal, sr=sampling_rate, fmin=65, fmax=300)
    # f0 = torch.nan_to_num(torch.from_numpy(f0), nan=0)
    f0 = torchyin.estimate(signal, sample_rate=sampling_rate, pitch_min=65, pitch_max=300)
    gmean_f0 = geo_mean(f0)
    return gmean_f0


def geo_mean(iterable):
    iterable_no_zero = iterable[iterable != 0]
    return torch.exp(torch.mean(torch.log(iterable_no_zero))).item()


def get_rms(signal):
    # In principle, you get Ws=J for energy and W for power, but no units in a wav file
    # just a ratio to the maximum possible value of a wav-file.
    return np.sqrt(np.mean(signal ** 2))


def get_freq_rms(path, sampling_rate):
    s, sr = librosa.load(path, sr=sampling_rate)
    return get_f0(s, sr), get_rms(s)


def get_features(path: str, sampling_rate: int) -> Tuple[bool, WavFeatures]:
    name = path.stem
    try:
        f0, rms = get_freq_rms(path, sampling_rate)
        if math.isnan(f0) or math.isnan(rms): raise ValueError("NaN value found.")
        return True, WavFeatures(name, f0, rms)
    except Exception as e:
        print(f"Could not process file: {name}: {e}")
        return False, WavFeatures(name, 0, 0)


def load_feature_space(filepath: str) -> List[WavFeatures]:
    return joblib.load(filepath)


def save_feature_space(filepath: str, features: List[WavFeatures]):
    joblib.dump(features, filepath)


def get_semitones_distance(source, target):
    return 1200 * np.log2(target / source)  # cite this!!


def get_rms_ratio(source, target):
    return target / source


def normalise_all(points):
    all_f0s, all_rms = zip(*points)
    np_f0s = np.array(all_f0s)
    np_rms = np.array(all_rms)
    f0_min = np_f0s.min()
    f0_max = np_f0s.max()
    rms_min = np_rms.min()
    rms_max = np_rms.max()
    f0_norm = (all_f0s - f0_min) / (f0_max - f0_min)
    rms_norm = (all_rms - rms_min) / (rms_max - rms_min)

    return list(zip(f0_norm, rms_norm)), (f0_min, f0_max, rms_min, rms_max)


def normalise(point, f0_min, f0_max, rms_min, rms_max):
    f0, rms = point
    f0_norm = (f0 - f0_min) / (f0_max - f0_min)
    rms_norm = (rms - rms_min) / (rms_max - rms_min)
    return f0_norm, rms_norm


def denormalise(point, f0_min, f0_max, rms_min, rms_max):
    f0_norm, rms_norm = point
    f0 = (f0_norm * (f0_max - f0_min)) + f0_min
    rms = (rms_norm * (rms_max - rms_min)) + rms_min
    return f0, rms


def get_nearest_neighbours(query_point, all_points, k=1):
    distances = []
    for point in all_points:
        d = dist(query_point, point)
        distances.append((point, d))
    distances = sorted(distances, key=lambda d: d[1])
    return [d[0] for d in distances[1:k + 1]]  # we exclude the first item since that is the query point itself


def interpolate(p1, p2, scaling_factor=1):
    p1_x, p1_y = p1
    p2_x, p2_y = p2

    dx = p2_x - p1_x
    dy = p2_y - p1_y

    scale = random.random() * scaling_factor
    return p1_x + scale * dx, p1_y + scale * dy


def get_wav_features_by_name(name: str, features: List[WavFeatures]):
    for f in features:
        if f.name == name:
            return f
    assert False, f"WavFeatures with name {name} does not exist!"


def adsmote_preprocess(ds_consts) -> list:
    sampling_rate = ds_consts.SAMPLING_RATE
    dataset_filelist = Path(f"{ds_consts.DATA_DIR}/{ds_consts.MANIFEST}").resolve()
    save_filename = f"{dataset_filelist.with_suffix('')}_adsmote.fs"

    print(f"Mapping feature space from file: {dataset_filelist}")
    features, failed_filenames = map_feature_space(dataset_filelist, sampling_rate, delim=",")
    save_feature_space(save_filename, features)
    print(f"Saved mapping to: {save_filename}")
    return failed_filenames
    

def prune_manifest(manifest_file: str, failed_filenames: list):
    manifest_file = Path(manifest_file).resolve()
    def contains_any(line, items):
        for item in items:
            if item in line: return True
        return False

    with open(manifest_file, "r") as fh:
        lines = fh.readlines()

    pruned = [line for line in lines if not contains_any(line, failed_filenames)]
    assert len(pruned) == len(lines) - len(failed_filenames), "Unexpected number of pruned filenames"
    with open(f"{manifest_file.with_suffix('')}_pruned.csv", "w") as fh:
        fh.writelines(pruned)
    
if __name__=="__main__":
    feat = get_features(Path("/root/intelpa-1/datasets/enterface database/subject 33/happiness/sentence 3/s33_ha_3.avi"), 48000)
    print(feat)