import sys
import torch
import numpy as np
import random
import torch.nn as nn
import torchvision.transforms as transforms
from typing import Any, Callable, List, Tuple
from torchaudio import sox_effects

try:
    from extpt.augment.augment_utils import (
        normalise_all,
        load_feature_space,
        normalise,
        denormalise,
        get_f0,
        get_rms,
        get_nearest_neighbours,
        interpolate,
        get_semitones_distance,
        get_rms_ratio,
    )
    from extpt.augment.sample_poly import sample_poly
    from extpt.constants import *
except ModuleNotFoundError:
    from augment.augment_utils import (
        normalise_all,
        load_feature_space,
        normalise,
        denormalise,
        get_f0,
        get_rms,
        get_nearest_neighbours,
        interpolate,
        get_semitones_distance,
        get_rms_ratio,
    )
    from augment.sample_poly import sample_poly
    from constants import *


        

class SoxPitchVolEffect:
    sample_rate: int

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate

    def apply(self, signal: torch.Tensor, pitch_distance: float, vol_factor: float) -> Tuple[torch.Tensor, int]:
        if np.isnan(pitch_distance) or np.isnan(vol_factor):
            raise ValueError("One or more parameters are not real numbers!")

        if isinstance(signal, np.ndarray):
            signal = torch.from_numpy(signal).unsqueeze(0)

        effects = [
            ["pitch", f"{pitch_distance}"],
            ["vol", f"{vol_factor}"],
            ["rate", f"{self.sample_rate}"]
        ]
        return sox_effects.apply_effects_tensor(signal, self.sample_rate, effects)


def functional_sox_effect(signal: torch.Tensor, pitch_distance: float, vol_factor: float, sample_rate: float) -> Tuple[torch.Tensor, int]:
    if np.isnan(pitch_distance) or np.isnan(vol_factor):
        raise ValueError("One or more parameters are not real numbers!")

    if isinstance(signal, np.ndarray):
        signal = torch.from_numpy(signal).unsqueeze(0)

    effects = [
        ["pitch", f"{pitch_distance}"],
        ["vol", f"{vol_factor}"],
        ["rate", f"{sample_rate}"]
    ]
    return sox_effects.apply_effects_tensor(signal, sample_rate, effects)

def functional_adsmote_aug( 
    signal,  # np.ndarray or torch.Tensor
    sampling_rate: int,
    f0_min: float,
    f0_max: float,
    rms_min: float,
    rms_max: float,
    feature_space: torch.Tensor,
    knns: int=5,
    n_samples: int=1    
    ) -> torch.Tensor:

    if not isinstance(signal, np.ndarray):
        signal = signal.numpy()
    point = f0_source, rms_source = get_f0(signal, sampling_rate), get_rms(signal)
    point = normalise(point, f0_min, f0_max, rms_min, rms_max)
    nearest_neighbors = get_nearest_neighbours(point, feature_space, k=knns)
    if knns > 2:
        # we need at least 3 points to create a 2d simplex
        samples = sample_poly(nearest_neighbors, n_samples)
    else:
        # Basic SMOTE method
        s = nearest_neighbors[0]
        synth = interpolate(point, s, scaling_factor=1)
        samples = [synth]

    aug_signals = []
    for other in samples:
        f0_target, rms_target = denormalise(other, f0_min, f0_max, rms_min, rms_max)
        cents_shift = get_semitones_distance(f0_source, f0_target)
        rms_shift = get_rms_ratio(rms_source, rms_target)
        aug_signal, _ = functional_sox_effect(signal, cents_shift, rms_shift, sampling_rate)
        aug_signals.append(aug_signal)
    return aug_signals

# video augmentation part
class VideoAugmentor(nn.Module):
    def __init__(self, rot_degrees=15, flip_chance=0.2, invert_chance=0.2) -> None:
        super().__init__()
        self.rot_degrees = rot_degrees
        self.flip_chance = flip_chance
        self.invert_chance = invert_chance

    def generate_transforms(self):
        # generate fixed transform based off the random state of the 3 parameters:
        # rot_degrees [-deg, +deg], flip_chance[0, 1], invert_chance[0, 1]
        rot_degrees = random.randint(-self.rot_degrees, self.rot_degrees)
        should_flip = 1 if random.random() < self.flip_chance else 0
        should_invert = 1 if random.random() < self.invert_chance else 0
        return nn.Sequential(
            transforms.RandomRotation(degrees=[rot_degrees, rot_degrees]),
            transforms.RandomHorizontalFlip(p=should_flip),
            transforms.RandomInvert(p=should_invert)
        )

    def forward(self, rgb_batch):
        # in: (batch, channels, frames, h, w), spec, frame_count, labels
        # out: (batch, frames, channels, h, w)
        rgb_batch = rgb_batch.permute(0, 2, 1, 3, 4)
        aug_batch = []
        for frames in rgb_batch:
            aug = self.generate_transforms()
            aug_frames = aug(frames)
            aug_batch.append(aug_frames.unsqueeze(0))
        
        aug_batch = torch.cat(aug_batch)
        aug_batch = aug_batch.permute(0, 2, 1, 3, 4)
        return aug_batch


class AdSmoteAugment:
    def __init__(
        self, 
        gamma: float,
        features_path: str,
        sampling_rate: int,
        knns: int,
        samples: int
    ):
        self.samples = samples
        self.gamma = gamma
        self.knns = knns
        self.tfm = SoxPitchVolEffect(sampling_rate)
        # self.get_mel_fn = get_mel_fn
        self.sampling_rate = sampling_rate
        feature_space, min_maxes = normalise_all([(wav.f0, wav.rms) for wav in load_feature_space(features_path)])
        self.feature_space = feature_space
        self.f0_min, self.f0_max, self.rms_min, self.rms_max = min_maxes

    def augment_one(self, signal,  # np.ndarray or torch.Tensor
                    sampling_rate: int) -> torch.Tensor:
        if not isinstance(signal, np.ndarray):
            signal = signal.numpy()
        point = f0_source, rms_source = get_f0(signal, sampling_rate), get_rms(signal)
        point = normalise(point, self.f0_min, self.f0_max, self.rms_min, self.rms_max)
        nearest_neighbors = get_nearest_neighbours(point, self.feature_space, k=self.knns)
        if self.knns > 2:
            # we need at least 3 points to create a 2d simplex
            samples = sample_poly(nearest_neighbors, self.samples)
        else:
            # Basic SMOTE method
            s = nearest_neighbors[0]
            synth = interpolate(point, s, scaling_factor=1)
            samples = [synth]

        # aug_specs = []
        aug_signals = []
        for other in samples:
            f0_target, rms_target = denormalise(other, self.f0_min, self.f0_max, self.rms_min, self.rms_max)
            cents_shift = get_semitones_distance(f0_source, f0_target)
            rms_shift = get_rms_ratio(rms_source, rms_target)
            aug_signal, _ = self.tfm.apply(signal, cents_shift, rms_shift)

            # aug_spec = self.get_mel_fn(
            #     aug_signal)  # make sure this returns the right data type(torch.Tensor, np.ndrray, tf.Tensor etc
            # aug_specs.append(aug_spec)
            aug_signals.append(aug_signal)
        return aug_signals

    def augment_batch(self, batch):
        augmented_batch = []
        num_real = round(self.gamma * len(batch))
        assert num_real > 0, "Too few real points! Increase batch size or gamma."
        for idx in range(num_real):
            augmented_batch.append(batch[idx])

        failed_signals = 0
        last_successful_aug = None
        while len(augmented_batch) < len(batch):
            random_idx = random.randrange(0, num_real)
            rgb_item, signal, frame_count, label = batch[random_idx]
            try:
                aug_items = self.augment_one(signal, self.sampling_rate)
                labelled_aug_items = [(rgb_item, i, frame_count, label) for i in aug_items]
                last_successful_aug = labelled_aug_items
                augmented_batch += labelled_aug_items
            except ValueError as e:
                # Augment fn sometimes randomly fails?? sellotape fix until
                # I figure out why.
                if last_successful_aug:
                    augmented_batch += last_successful_aug
                failed_signals += 1

        augmented_batch = augmented_batch[:len(batch)]
        assert len(batch) == len(augmented_batch), "Augmentation should not change batch size!"
        if failed_signals > 0:
            print(f"Failed to process {failed_signals} audio signals in batch")
        return augmented_batch

# combined
class MultimodalAugment:
    def __init__(self, ds_consts):
        self.video_aug = VideoAugmentor(
            rot_degrees=ds_consts.ROT_DEGREES,
            flip_chance=ds_consts.FLIP_CHANCE, 
            invert_chance=ds_consts.INVERT_CHANCE
        )
        self.audio_aug = AdSmoteAugment(
            gamma=ds_consts.ADSMOTE_GAMMA,
            knns=ds_consts.ADSMOTE_KNNS,
            sampling_rate=ds_consts.SAMPLING_RATE,
            samples=ds_consts.ADSMOTE_POLY_SAMPLES,
            features_path=f"{ds_consts.DATA_DIR}/{ds_consts.ADSMOTE_FEATURES}"
        )

    def __call__(self, batch) -> Any:

        batch = self.audio_aug.augment_batch(batch)
        batch = [(self.video_aug(v.unsqueeze(0)).squeeze(0), a, f, l) for v, a, f, l in batch]
        return batch
    