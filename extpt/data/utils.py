from pathlib import Path
from typing import List, Union
from tqdm import tqdm
import csv

import torchaudio
import torch
import librosa


def safe_split(total_len: int, fractions: List[float]):
    assert sum(fractions) == 1, f"Split fractions must sum to 1, but got: {sum(split)}"
    split = [
       round(frac * total_len)
       for frac in fractions 
    ]

    # if there is a mismatch, make up the difference by adding it to train samples
    split_sum = sum(split)
    if total_len != split_sum:
        diff = max(total_len, split_sum) - min(total_len, split_sum)
        diff = total_len - sum(split)
        split[0] += diff
    assert sum(split) == total_len, f"Expected sum of split == {total_len}, but got: {sum(split)}" 
    return split


def prune_manifest(manifest: Union[str, List], failed_filenames: list, ds_namespace):
    def contains_any(line, items):
        for item in items:
            if item in line: return True
        return False

    if isinstance(manifest, str):
        manifest_file = Path(manifest).resolve()
        with open(manifest_file, "r") as fh:
            lines = fh.readlines()
    else:
        # if manifest is a list
        manifest_file = Path(f"{ds_namespace.DATA_DIR}/{ds_namespace.MANIFEST}")
        lines = manifest

    pruned = []
    for line in tqdm(lines):
        if not contains_any(line, failed_filenames):
            pruned.append(line)
    assert len(pruned) == len(lines) - len(failed_filenames), "Unexpected number of pruned filenames"
    return pruned


def save_manifest(filename, manifest):
    with open(filename, "w") as fh:
        writer = csv.writer(fh)
        writer.writerows(manifest)
    

def generate_manifest(ds_namespace):
    """
    The first 2 fields need to be filepath, duration.
    For backwards compatibility reasons, the next 2 fields must be
    frames, labels, and must have at least some dummy value. The rest of the 
    fields can be anything.
    """
    from extpt.datasets import get_media_info

    failed_filenames = []
    manifest = []
    print(f"Generating manifest for dataset {ds_namespace.NAME}")
    pbar = tqdm(list(Path(ds_namespace.DATA_DIR).rglob(f"*.{ds_namespace.DATA_EXT}")))
    max_dur = float('-inf')
    min_dur = float('inf')
    avg_dur = 0
    for p in pbar:
        try:
            p = Path(p)
            duration, frames, metadata = get_media_info(p)
            fields = ds_namespace.manifest_fn(p)
            items = [fields["filepath"], duration, frames, fields["label"]]
            del fields["filepath"]
            del fields["label"]
            row = [*items, *list(fields.values())]
            manifest.append(row)

            if duration > max_dur:
                max_dur = duration
            if duration < min_dur:
                min_dur = duration
            avg_dur += duration
            pbar.set_description(f"({len(failed_filenames)} failed)")
            pbar.refresh()
        except Exception as e:
            failed_filenames.append(p.stem)
            continue
    print(f"Max duration: {max_dur:.2f}s, Min duration: {min_dur:.2f}s, Avg duration: {avg_dur:.2f}s")
    return manifest, failed_filenames


def invert_melspec(melspec, n_fft, sr, hop_len):
    spec_ampl = torchaudio.functional.DB_to_amplitude(melspec, 0.01, 0.5)
    audio_signal = librosa.feature.inverse.mel_to_audio(
        spec_ampl.numpy(), sr=sr, n_fft=n_fft, hop_length=hop_len, window="hann")

    audio_signal = torch.from_numpy(audio_signal).unsqueeze(0)
    return audio_signal
    