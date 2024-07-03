import sys
import csv
from typing import List
from types import ModuleType
import multiprocessing as mp
from multiprocessing import get_context
from pathlib import Path
import random
from tqdm import tqdm
from functools import partial

import torch.nn.functional as F
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms as vtrans
from torchaudio import transforms as atrans

try:
    from extpt.tokenizer import make_input
    from extpt.constants import *
    from extpt.helpers import * 
    from extpt.datasets import enterface
    from extpt.augment.augment_utils import normalise_all, load_feature_space
    from extpt.augment.specaugment import Specaugment
    from extpt.data.utils import safe_split
except ModuleNotFoundError:
    from tokenizer import make_input
    from constants import *
    from helpers import * 
    from datasets import enterface
    from augment.augment_utils import normalise_all, load_feature_space
    from augment.specaugment import Specaugment
    from data.utils import safe_split


    
class EmoDataset(Dataset):
    def __init__(self, dataset_const_namespace, nlines, shuffle_manifest=False, leave_speaker_out=None):
        super(EmoDataset, self).__init__()
        self.constants = dataset_const_namespace
        self.modalities = dataset_const_namespace.MODALITIES
        manifest_filepath = Path(self.constants.DATA_DIR) / f"{self.constants.MANIFEST}"
        if shuffle_manifest:
            self.dataset = pd.read_csv(manifest_filepath).sample(frac=1).head(nlines)
        else:
            self.dataset = pd.read_csv(manifest_filepath, nrows=nlines)

        if leave_speaker_out is not None:
            speaker_ids = leave_speaker_out["speaker_ids"]
            train = leave_speaker_out["is_train_set"]
            speaker_id_col = 4
            for speaker_id in speaker_ids:
                assert (self.dataset.iloc[:, speaker_id_col].astype(str).str.contains(speaker_id)).any(), f"Speaker with ID {speaker_id} not found in dataset {self.constants.NAME}."
            
            if train:
                self.dataset = self.dataset[~self.dataset.iloc[:, speaker_id_col].astype(str).isin(speaker_ids)]
            else:
                self.dataset = self.dataset[self.dataset.iloc[:, speaker_id_col].astype(str).isin(speaker_ids)]

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, item):
        header_names = self.constants.HEADERS
        filename = self.dataset.iloc[item].iloc[0]
        duration = self.dataset.iloc[item].iloc[1]
        frames_count = self.dataset.iloc[item].iloc[2]
        label = self.dataset.iloc[item].iloc[3]
        other_labels = {}
        for i, header_name in enumerate(header_names[4:], start=4):
            other_labels[header_name] = self.dataset.iloc[item].iloc[i]
        # speaker_id = int(str(Path(filename).stem).split('_')[0][1:])
        rgb_frames, audio_samples = make_input(
                                        filename, 
                                        float(duration), 
                                        self.modalities,
                                        constant_fps=True,
                                    )
        return (
            rgb_frames,
            audio_samples,
            frames_count,
            label,
            other_labels
        )
    
class EmoDatasetNeutralAware(Dataset):
    def __init__(self, dataset_const_namespace, nlines, shuffle_manifest=False, leave_speaker_out=None):
        super(EmoDatasetNeutralAware, self).__init__()
        self.constants = dataset_const_namespace
        self.modalities = dataset_const_namespace.MODALITIES
        manifest_filepath = Path(self.constants.DATA_DIR) / f"{self.constants.MANIFEST}"
        if shuffle_manifest:
            self.dataset = pd.read_csv(manifest_filepath).sample(frac=1).head(nlines)
        else:
            self.dataset = pd.read_csv(manifest_filepath, nrows=nlines)

        if leave_speaker_out is not None:
            speaker_id = leave_speaker_out["speaker_id"]
            train = leave_speaker_out["is_train_set"]
            speaker_id_col = 5
            assert (self.dataset.iloc[:, speaker_id_col].astype(str).str.contains(speaker_id)).any(), f"Speaker with ID {speaker_id} not found in dataset {self.constants.NAME}."
            if train:
                self.dataset = self.dataset[~self.dataset.iloc[:, speaker_id_col].astype(str).str.contains(speaker_id)]
            else:
                self.dataset = self.dataset[self.dataset.iloc[:, speaker_id_col].astype(str).str.contains(speaker_id)]

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, item):
        header_names = ["filepath_neu", "filepath", "duration", "frames", "emo_label", "speaker_id"]
        filename_neutral = self.dataset.iloc[item].iloc[0]
        filename = self.dataset.iloc[item].iloc[1]
        duration = self.dataset.iloc[item].iloc[2]
        frames_count = self.dataset.iloc[item].iloc[3]
        label = self.dataset.iloc[item].iloc[4]
        other_labels = {}
        for i, header_name in enumerate(header_names[5:], start=5):
            other_labels[header_name] = self.dataset.iloc[item].iloc[i]


        rgb_frames, audio_samples_emo = make_input(
                                        filename, 
                                        float(duration), 
                                        self.modalities,
                                        constant_fps=True,
                                    )
        rgb_frames, audio_samples_neu = make_input(
                                        filename_neutral, 
                                        float(duration), 
                                        self.modalities,
                                        constant_fps=True,
                                    )
        return (
            audio_samples_neu,
            audio_samples_emo,
            frames_count,
            label,
            other_labels
        )


class CollateContrastiveAudio:
    def __init__(
        self, 
        dataset_namespace: ModuleType,
        augmentor=None,
        ):
        """
        force_audio_aspect: force resize of audio spectrogram to video shape
        """
        self.dataset_constants = dataset_namespace
        self.force_audio_shape = self.dataset_constants.FORCE_AUDIO_ASPECT
        hop_len = int(self.dataset_constants.SAMPLING_RATE * self.dataset_constants.SPEC_HOP_LEN_MS / 1000)
        win_len = int(self.dataset_constants.SAMPLING_RATE * self.dataset_constants.SPEC_WINDOW_SZ_MS/ 1000)
        self.audio_transform = nn.Sequential(
            atrans.MelSpectrogram(
                sample_rate=self.dataset_constants.SAMPLING_RATE,
                n_fft=1024,
                n_mels=NUM_MELS,
                win_length=win_len,
                hop_length=hop_len,
                normalized=True
            ),
            atrans.AmplitudeToDB(),
            vtrans.Resize((HEIGHT, WIDTH), antialias=True) if self.force_audio_shape else nn.Identity()
        )
        self.audio_augmentor = augmentor 

    def __call__(self, batch):
        
        spec_tensor = []
        spec_tensor_aug = []
        label_list = []
        other_labels_list = []

        for _, audio_samples, frames_count, label, other_labels in batch:



            spec = self.audio_transform(audio_samples.unsqueeze(0)).squeeze(0)
            spec_tmp = spec + abs(spec.min())
            spec_norm = spec_tmp / spec_tmp.max()
            spec_norm = spec_norm.permute(1, 0)
            
            spec_aug_norm = spec_norm.clone()
            if self.audio_augmentor:
                spec_aug_norm = self.audio_augmentor(spec_aug_norm)

            spec_tensor.append(spec_norm)
            spec_tensor_aug.append(spec_aug_norm)

            label = torch.tensor([label], dtype=torch.long).unsqueeze(0)
            label_list.append(label)
            
            other_labels_list.append(other_labels)


        label_batch_tensor = torch.LongTensor(len(batch), self.dataset_constants.NUM_LABELS)
        torch.cat(label_list, out=label_batch_tensor)
        
        spec_batch_tensor = pad_sequence(spec_tensor, batch_first=False)[:self.dataset_constants.SPEC_MAX_LEN, :, :]
        spec_batch_tensor = spec_batch_tensor.permute(1, 0, 2)
        spec_batch_tensor_aug = pad_sequence(spec_tensor_aug, batch_first=False)[:self.dataset_constants.SPEC_MAX_LEN, :, :]
        spec_batch_tensor_aug = spec_batch_tensor_aug.permute(1, 0, 2)

        to_pad = self.dataset_constants.SPEC_MAX_LEN - spec_batch_tensor.shape[1]
        to_pad_aug = self.dataset_constants.SPEC_MAX_LEN - spec_batch_tensor_aug.shape[1]
        spec_batch_tensor = F.pad(spec_batch_tensor, (0, 0, 0, to_pad), value=0)
        spec_batch_tensor_aug = F.pad(spec_batch_tensor_aug, (0, 0, 0, to_pad_aug), value=0)

        return {
            "audios": spec_batch_tensor,
            "audios_aug": spec_batch_tensor_aug,
            "labels": label_batch_tensor,
            "other_labels": other_labels_list
        } 

    
class CollateContrastiveMSP:
    def __init__(
        self, 
        dataset_namespace: ModuleType,
        augmentor=None,
        ):
        """
        force_audio_aspect: force resize of audio spectrogram to video shape
        """
        self.dataset_constants = dataset_namespace
        self.force_audio_shape = self.dataset_constants.FORCE_AUDIO_ASPECT
        hop_len = int(self.dataset_constants.SAMPLING_RATE * self.dataset_constants.SPEC_HOP_LEN_MS / 1000)
        win_len = int(self.dataset_constants.SAMPLING_RATE * self.dataset_constants.SPEC_WINDOW_SZ_MS/ 1000)
        self.audio_transform = nn.Sequential(
            atrans.MelSpectrogram(
                sample_rate=self.dataset_constants.SAMPLING_RATE,
                n_fft=1024,
                n_mels=NUM_MELS,
                win_length=win_len,
                hop_length=hop_len,
                normalized=True
            ),
            atrans.AmplitudeToDB(),
            vtrans.Resize((HEIGHT, WIDTH), antialias=True) if self.force_audio_shape else nn.Identity()
        )
        self.audio_augmentor = augmentor 

    def __call__(self, batch):
        
        spec_tensor = []
        spec_tensor_aug = []
        act_val_dom_list = []
        label_list = []
        other_labels_list = []

        for _, audio_samples, frames_count, label, other_labels in batch:


            act, val, dom = other_labels['act'], other_labels['val'], other_labels['dom']

            spec = self.audio_transform(audio_samples.unsqueeze(0)).squeeze(0)
            spec_tmp = spec + abs(spec.min())
            spec_norm = spec_tmp / spec_tmp.max()
            spec_norm = spec_norm.permute(1, 0)
            
            spec_aug_norm = spec_norm.clone()
            if self.audio_augmentor:
                spec_aug_norm = self.audio_augmentor(spec_aug_norm)

            spec_tensor.append(spec_norm)
            spec_tensor_aug.append(spec_aug_norm)

            label = torch.tensor([label], dtype=torch.long).unsqueeze(0)
            label_list.append(label)
            act_val_dom_tensor = torch.FloatTensor([act, val, dom]).unsqueeze(0)
            act_val_dom_list.append(act_val_dom_tensor)
            
            other_labels_list.append(other_labels)


        label_batch_tensor = torch.LongTensor(len(batch), self.dataset_constants.NUM_LABELS)
        act_val_dom_batch_tensor = torch.cat(act_val_dom_list)
        torch.cat(label_list, out=label_batch_tensor)
        
        spec_batch_tensor = pad_sequence(spec_tensor, batch_first=False)[:self.dataset_constants.SPEC_MAX_LEN, :, :]
        spec_batch_tensor = spec_batch_tensor.permute(1, 0, 2)
        spec_batch_tensor_aug = pad_sequence(spec_tensor_aug, batch_first=False)[:self.dataset_constants.SPEC_MAX_LEN, :, :]
        spec_batch_tensor_aug = spec_batch_tensor_aug.permute(1, 0, 2)

        to_pad = self.dataset_constants.SPEC_MAX_LEN - spec_batch_tensor.shape[1]
        to_pad_aug = self.dataset_constants.SPEC_MAX_LEN - spec_batch_tensor_aug.shape[1]
        spec_batch_tensor = F.pad(spec_batch_tensor, (0, 0, 0, to_pad), value=0)
        spec_batch_tensor_aug = F.pad(spec_batch_tensor_aug, (0, 0, 0, to_pad_aug), value=0)

        return {
            "audios": spec_batch_tensor,
            "audios_aug": spec_batch_tensor_aug,
            "labels": label_batch_tensor,
            "act_val_dom": act_val_dom_batch_tensor,
            "other_labels": other_labels_list
        } 

class CollateNoncontrastive:
    def __init__(
        self, 
        dataset_namespace: ModuleType,
        augmentor=None,
        ):
        """
        force_audio_aspect: force resize of audio spectrogram to video shape
        """
        self.dataset_constants = dataset_namespace
        self.force_audio_shape = self.dataset_constants.FORCE_AUDIO_ASPECT
        # this one works for enterface
        # atrans.MelSpectrogram(
        #     sample_rate=self.dataset_constants.SAMPLING_RATE,
        #     n_fft=1024,
        #     n_mels=128,
        #     win_length=None,
        #     hop_length=512,
        # ),
        hop_len = int(self.dataset_constants.SAMPLING_RATE * self.dataset_constants.SPEC_HOP_LEN_MS / 1000)
        win_len = int(self.dataset_constants.SAMPLING_RATE * self.dataset_constants.SPEC_WINDOW_SZ_MS/ 1000)
        self.audio_transform = nn.Sequential(
            atrans.MelSpectrogram(
                sample_rate=self.dataset_constants.SAMPLING_RATE,
                n_fft=1024,
                n_mels=NUM_MELS,
                win_length=win_len,
                hop_length=hop_len,
            ),
            atrans.AmplitudeToDB(),
            vtrans.Resize((HEIGHT, WIDTH), antialias=True) if self.force_audio_shape else nn.Identity()
        )
        # features_path = f"{self.dataset_constants.DATA_DIR}/{self.dataset_constants.ADSMOTE_FEATURES}"
        # feature_space, min_maxes = normalise_all([(wav.f0, wav.rms) for wav in load_feature_space(features_path)])
        # self.feature_space = feature_space
        # self.f0_min, self.f0_max, self.rms_min, self.rms_max = min_maxes
        self.augmentor = augmentor

    def __call__(self, batch):
        
        rgb_tensor_list = []
        spec_tensor_list = []
        label_list = []
        speaker_id_list = []

        if self.augmentor:
            batch = self.augmentor(batch)
        
        for rgb_frames, audio_samples, frames_count, label, speaker_id, *_ in batch:

            if rgb_frames is not None:
                start1, start2 = get_clip_start_frames(frames_count, FRAMES)
                end1, end2 = start1 + FRAMES, start2 + FRAMES
                rgb = rgb_frames[start1 : end1]
                rgb_tensor_list.append(rgb.unsqueeze(0).permute(0, 2, 1, 3, 4))

            if audio_samples is not None:
                spec_norm = self.audio_transform(audio_samples.unsqueeze(0)).squeeze(0)
                spec_norm = spec_norm.permute(1, 0)
                spec_tensor_list.append(spec_norm)

            label = torch.tensor([label], dtype=torch.long).unsqueeze(0)
            label_list.append(label)

            speaker_id_list.append(speaker_id)


        label_batch_tensor = torch.LongTensor(len(batch), self.dataset_constants.NUM_LABELS)
        torch.cat(label_list, out=label_batch_tensor)

        if len(rgb_tensor_list) > 0:
            rgb_batch_tensor = torch.FloatTensor(len(batch), CHANS, FRAMES, HEIGHT, WIDTH)
            torch.cat(rgb_tensor_list, out=rgb_batch_tensor)
        else:
            rgb_batch_tensor = None

        if len(spec_tensor_list) > 0:
            spec_batch_tensor = pad_sequence(spec_tensor_list, batch_first=False)[:self.dataset_constants.SPEC_MAX_LEN, :, :]
            spec_batch_tensor = spec_batch_tensor.permute(1, 0, 2)
            to_pad = self.dataset_constants.SPEC_MAX_LEN - spec_batch_tensor.shape[1]
            spec_batch_tensor = F.pad(spec_batch_tensor, (0, 0, 0, to_pad), value=0)
        else:
            spec_batch_tensor = None

        assert not (rgb_batch_tensor is None and spec_batch_tensor is None), "At least one modality of [audio, video] must be present"

        return {
            "clip0": (rgb_batch_tensor, spec_batch_tensor),
            "labels": label_batch_tensor,
            "speaker_ids": speaker_id_list
        } 

class Collate_Constrastive:
    def __init__(
        self, 
        dataset_namespace: ModuleType,
        augmentor=None,
        ):
        """
        force_audio_aspect: force resize of audio spectrogram to video shape
        """
        self.dataset_constants = dataset_namespace
        self.force_audio_shape = self.dataset_constants.FORCE_AUDIO_ASPECT
        hop_len = int(self.dataset_constants.SAMPLING_RATE * self.dataset_constants.SPEC_HOP_LEN_MS / 1000)
        win_len = int(self.dataset_constants.SAMPLING_RATE * self.dataset_constants.SPEC_WINDOW_SZ_MS/ 1000)
        self.audio_transform = nn.Sequential(
            atrans.MelSpectrogram(
                sample_rate=self.dataset_constants.SAMPLING_RATE,
                n_fft=1024,
                n_mels=NUM_MELS,
                win_length=win_len,
                hop_length=hop_len,
                normalized=True
            ),
            atrans.AmplitudeToDB(),
            vtrans.Resize((HEIGHT, WIDTH), antialias=True) if self.force_audio_shape else nn.Identity()
        )
        self.audio_augmentor = augmentor 

    def __call__(self, batch):
        
        rgb_tensor_list1 = []
        rgb_tensor_list2 = []
        spec_tensor_list1 = []
        spec_tensor_list2 = []
        label_list = []
        other_labels_list = []

        for rgb_frames, audio_samples, frames_count, label, other_labels in batch:

            if rgb_frames is not None:
                start1, start2 = get_clip_start_frames(frames_count, FRAMES)
                end1, end2 = start1 + FRAMES, start2 + FRAMES
                rgb1 = rgb_frames[start1 : end1]
                rgb2 = rgb_frames[start2 : end2]
                rgb_tensor_list1.append(rgb1.unsqueeze(0).permute(0, 2, 1, 3, 4))
                rgb_tensor_list2.append(rgb2.unsqueeze(0).permute(0, 2, 1, 3, 4))


            if audio_samples is not None:
                spec1 = self.audio_transform(audio_samples.unsqueeze(0)).squeeze(0)
                spec_tmp = spec1 + abs(spec1.min())
                spec1_norm = spec_tmp / spec_tmp.max()
                # spec1 = spec1.unsqueeze(1).repeat(1, 3, 1, 1) # repeat twice in the frames axis so 3d convolution returns a non-zero value in this axis
                
                
                # audio_samples2 = None
                # while audio_samples2 is None:
                #     try:
                #         audio_samples2 = functional_adsmote_aug(
                #             audio_samples,
                #             self.dataset_constants.SAMPLING_RATE,
                #             self.f0_min,
                #             self.f0_max,
                #             self.rms_min,
                #             self.rms_max,
                #             self.feature_space
                #         )[0] # returns a list of transformed signals
                #     except ValueError as e:
                #         print(e)
                
                # spec2 = self.audio_transform(audio_samples2.unsqueeze(0)) 
                # spec2 = spec2.repeat(1, 3, 1, 1)
                # spec2_tmp = spec2 + abs(spec2.min())
                # spec2_norm = spec2_tmp / spec2_tmp.max()




                spec1_norm = spec1_norm.permute(1, 0)
                spec2_norm = spec1_norm.clone()
                spec_tensor_list1.append(spec1_norm)
                spec_tensor_list2.append(spec2_norm)


            label = torch.tensor([label], dtype=torch.long).unsqueeze(0)
            label_list.append(label)
            other_labels_list.append(other_labels)


        label_batch_tensor = torch.LongTensor(len(batch), self.dataset_constants.NUM_LABELS)
        torch.cat(label_list, out=label_batch_tensor)
        if len(rgb_tensor_list1) > 0:
            rgb_batch_tensor1 = torch.FloatTensor(len(batch), CHANS, FRAMES, HEIGHT, WIDTH)
            rgb_batch_tensor2 = torch.FloatTensor(len(batch), CHANS, FRAMES, HEIGHT, WIDTH)
            torch.cat(rgb_tensor_list1, out=rgb_batch_tensor1)
            torch.cat(rgb_tensor_list2, out=rgb_batch_tensor2)
        else:
            rgb_batch_tensor1, rgb_batch_tensor2 = None, None
        
        if len(spec_tensor_list1) > 0:
            spec_batch_tensor1 = pad_sequence(spec_tensor_list1, batch_first=False)[:self.dataset_constants.SPEC_MAX_LEN, :, :]
            spec_batch_tensor2 = pad_sequence(spec_tensor_list2, batch_first=False)[:self.dataset_constants.SPEC_MAX_LEN, :, :]
            spec_batch_tensor1 = spec_batch_tensor1.permute(1, 0, 2)
            spec_batch_tensor2 = spec_batch_tensor2.permute(1, 0, 2)
            to_pad = self.dataset_constants.SPEC_MAX_LEN - spec_batch_tensor1.shape[1]
            spec_batch_tensor1 = F.pad(spec_batch_tensor1, (0, 0, 0, to_pad), value=0)
            spec_batch_tensor2 = F.pad(spec_batch_tensor2, (0, 0, 0, to_pad), value=0)
            if self.audio_augmentor:
                spec_batch_tensor2 = self.audio_augmentor(spec_batch_tensor2)
        else:
            spec_batch_tensor1, spec_batch_tensor2 = None, None
        # if not self.force_audio_shape:
        #     padding = (0, 0, 0, self.dataset_constants.MAX_SPEC_SEQ_LEN - spec_tensor_list1[0].shape[0])
        #     padding = (0, 0, 0, self.dataset_constants.MAX_SPEC_SEQ_LEN - spec_tensor_list1[0].shape[0])
        #     spec_tensor_list1[0] = nn.ConstantPad2d(padding, 0)(spec_tensor_list1[0])
        #     spec_tensor_list2[0] = nn.ConstantPad2d(padding, 0)(spec_tensor_list2[0])
        #     spec_batch_tensor1 = pad_sequence(spec_tensor_list1, batch_first=True)
        #     spec_batch_tensor2 = pad_sequence(spec_tensor_list2, batch_first=True)
        #     spec_batch_tensor1 = spec_batch_tensor1.swapaxes(1, 2)
        #     spec_batch_tensor2 = spec_batch_tensor2.swapaxes(1, 2)
        #     spec_batch_tensor1 = spec_batch_tensor1.unsqueeze(1).repeat(1, 3, 1, 1)
        #     spec_batch_tensor2 = spec_batch_tensor2.unsqueeze(1).repeat(1, 3, 1, 1)

        return {
            "clip0": (rgb_batch_tensor1, spec_batch_tensor1),
            "clip1": (rgb_batch_tensor2, spec_batch_tensor2),
            "labels": label_batch_tensor,
            "other_labels": other_labels_list
        } 


class Collate_Constrastive_NeutralAware:
    def __init__(
        self, 
        dataset_namespace: ModuleType,
        augmentor=None,
        ):
        """
        force_audio_aspect: force resize of audio spectrogram to video shape
        """
        self.dataset_constants = dataset_namespace
        self.force_audio_shape = self.dataset_constants.FORCE_AUDIO_ASPECT
        hop_len = int(self.dataset_constants.SAMPLING_RATE * self.dataset_constants.SPEC_HOP_LEN_MS / 1000)
        win_len = int(self.dataset_constants.SAMPLING_RATE * self.dataset_constants.SPEC_WINDOW_SZ_MS/ 1000)
        self.audio_transform = nn.Sequential(
            atrans.MelSpectrogram(
                sample_rate=self.dataset_constants.SAMPLING_RATE,
                n_fft=1024,
                n_mels=NUM_MELS,
                win_length=win_len,
                hop_length=hop_len,
                normalized=True
            ),
            atrans.AmplitudeToDB(),
            vtrans.Resize((HEIGHT, WIDTH), antialias=True) if self.force_audio_shape else nn.Identity()
        )
        self.audio_augmentor = augmentor 

    def __call__(self, batch):
        
        spec_tensor_neu = []
        spec_tensor_emo = []
        label_list = []
        other_labels_list = []

        for audio_samples_neu, audio_samples_emo, frames_count, label, other_labels in batch:



            spec_neu = self.audio_transform(audio_samples_neu.unsqueeze(0)).squeeze(0)
            spec_neu_tmp = spec_neu + abs(spec_neu.min())
            spec_neu_norm = spec_neu_tmp / spec_neu_tmp.max()
            spec_neu_norm = spec_neu_norm.permute(1, 0)

            spec_emo = self.audio_transform(audio_samples_emo.unsqueeze(0)).squeeze(0)
            spec_emo_tmp = spec_emo + abs(spec_emo.min())
            spec_emo_norm = spec_emo_tmp / spec_emo_tmp.max()
            spec_emo_norm = spec_emo_norm.permute(1, 0)

            # spec_neu_norm = spec_emo_norm.clone()

            spec_tensor_neu.append(spec_neu_norm)
            spec_tensor_emo.append(spec_emo_norm)


            label = torch.tensor([label], dtype=torch.long).unsqueeze(0)
            label_list.append(label)
            other_labels_list.append(other_labels)


        label_batch_tensor = torch.LongTensor(len(batch), self.dataset_constants.NUM_LABELS)
        torch.cat(label_list, out=label_batch_tensor)
        rgb_batch_tensor1, rgb_batch_tensor2 = None, None
        
        spec_batch_tensor_neu = pad_sequence(spec_tensor_neu, batch_first=False)[:self.dataset_constants.SPEC_MAX_LEN, :, :]
        spec_batch_tensor_neu = spec_batch_tensor_neu.permute(1, 0, 2)
        spec_batch_tensor_emo = pad_sequence(spec_tensor_emo, batch_first=False)[:self.dataset_constants.SPEC_MAX_LEN, :, :]
        spec_batch_tensor_emo = spec_batch_tensor_emo.permute(1, 0, 2)

        to_pad_neu = self.dataset_constants.SPEC_MAX_LEN - spec_batch_tensor_neu.shape[1]
        to_pad_emo = self.dataset_constants.SPEC_MAX_LEN - spec_batch_tensor_emo.shape[1]
        spec_batch_tensor_neu = F.pad(spec_batch_tensor_neu, (0, 0, 0, to_pad_neu), value=0)
        spec_batch_tensor_emo = F.pad(spec_batch_tensor_emo, (0, 0, 0, to_pad_emo), value=0)
        if self.audio_augmentor:
            spec_batch_tensor_emo = self.audio_augmentor(spec_batch_tensor_emo)

        return {
            "clip0": (rgb_batch_tensor1, spec_batch_tensor_emo),
            "clip1": (rgb_batch_tensor2, spec_batch_tensor_neu),
            "labels": label_batch_tensor,
            "other_labels": other_labels_list
        } 

def load_data(dataset_const_namespace, 
              train_collate_func,
              val_collate_func,
              batch_sz=16,
              train_val_test_split=[0.8, 0.1, 0.1], 
              nlines=None,
              seed=None,
              shuffle_manifest=False,
              leave_speaker_out=None
              ):
            
        
    if leave_speaker_out is not None:
        leave_speaker_out = [str(l) for l in leave_speaker_out]
        train_set = EmoDataset(dataset_const_namespace, 
                               nlines=nlines, 
                               shuffle_manifest=shuffle_manifest, 
                               leave_speaker_out={
                                   "speaker_ids": leave_speaker_out,
                                   "is_train_set": True
                               }
                    )
        val_set = EmoDataset(dataset_const_namespace, 
                             nlines=nlines, 
                             shuffle_manifest=shuffle_manifest, 
                             leave_speaker_out={
                                 "speaker_ids": leave_speaker_out,
                                 "is_train_set": False
                             }
                    )
        
        train_dl = DataLoader(train_set, 
                            batch_size=batch_sz, 
                            shuffle=True, 
                            num_workers=NUM_DATA_WORKERS,
                            collate_fn=train_collate_func)            

        val_dl = DataLoader(val_set, 
                            batch_size=batch_sz, 
                            shuffle=False, 
                            num_workers=NUM_DATA_WORKERS,
                            collate_fn=train_collate_func)            
        
        test_dl = None
        split_seed = 0
    else:
        assert sum(train_val_test_split) == 1, "Train, val and test fractions should sum to 1!"  # Always a good idea to use static asserts when processing arguments that are passed in by a user!
        dataset = EmoDataset(dataset_const_namespace, nlines=nlines, shuffle_manifest=shuffle_manifest)
        
        # This code generates the actual number of items that goes into each split using the user-supplied fractions
        tr_va_te = safe_split(len(dataset), train_val_test_split)

        if seed:
            split_seed = seed
        else:
            split_seed = random.randint(0, sys.maxsize)
        generator = torch.Generator().manual_seed(split_seed)
        train_split, val_split, test_split = random_split(dataset, tr_va_te, generator=generator)
        
        # Use Pytorch DataLoader to load each split into memory. It's important to pass in our custom collate function, so it knows how to interpret the 
        # data and load it. num_workers tells the DataLoader how many CPU threads to use so that data can be loaded in parallel, which is faster
        if len(train_split) > 0:
            train_dl = DataLoader(train_split, 
                                batch_size=batch_sz, 
                                shuffle=True, 
                                num_workers=NUM_DATA_WORKERS,
                                collate_fn=train_collate_func)            
        else:
            train_dl = None

        if len(val_split) > 0:
            val_dl = DataLoader(val_split, 
                                batch_size=batch_sz, 
                                shuffle=False, 
                                num_workers=NUM_DATA_WORKERS,
                                collate_fn=val_collate_func)
        else:
            val_dl = None

        if len(test_split) > 0:
            test_dl = DataLoader(test_split,
                                batch_size=batch_sz,
                                shuffle=False,
                                num_workers=NUM_DATA_WORKERS,
                                collate_fn=val_collate_func)
        else:
            test_dl = None

    return train_dl, val_dl, test_dl, split_seed


def load_data_neutral_aware(dataset_const_namespace, 
              train_collate_func,
              val_collate_func,
              batch_sz=16,
              train_val_test_split=[0.8, 0.1, 0.1], 
              nlines=None,
              seed=None,
              shuffle_manifest=False,
              leave_speaker_out=None
              ):
            
        
    if leave_speaker_out is not None:
        leave_speaker_out = str(leave_speaker_out)
        train_set = EmoDatasetNeutralAware(dataset_const_namespace, 
                               nlines=nlines, 
                               shuffle_manifest=shuffle_manifest, 
                               leave_speaker_out={
                                   "speaker_id": leave_speaker_out,
                                   "is_train_set": True
                               }
                    )
        val_set = EmoDatasetNeutralAware(dataset_const_namespace, 
                             nlines=nlines, 
                             shuffle_manifest=shuffle_manifest, 
                             leave_speaker_out={
                                 "speaker_id": leave_speaker_out,
                                 "is_train_set": False
                             }
                    )
        
        train_dl = DataLoader(train_set, 
                            batch_size=batch_sz, 
                            shuffle=True, 
                            num_workers=NUM_DATA_WORKERS,
                            collate_fn=train_collate_func)            

        val_dl = DataLoader(val_set, 
                            batch_size=batch_sz, 
                            shuffle=False, 
                            num_workers=NUM_DATA_WORKERS,
                            collate_fn=train_collate_func)            
        
        test_dl = None
        split_seed = 0
    else:
        assert sum(train_val_test_split) == 1, "Train, val and test fractions should sum to 1!"  # Always a good idea to use static asserts when processing arguments that are passed in by a user!
        dataset = EmoDatasetNeutralAware(dataset_const_namespace, nlines=nlines, shuffle_manifest=shuffle_manifest)
        
        # This code generates the actual number of items that goes into each split using the user-supplied fractions
        tr_va_te = safe_split(len(dataset), train_val_test_split)

        if seed:
            split_seed = seed
        else:
            split_seed = random.randint(0, sys.maxsize)
        generator = torch.Generator().manual_seed(split_seed)
        train_split, val_split, test_split = random_split(dataset, tr_va_te, generator=generator)
        
        # Use Pytorch DataLoader to load each split into memory. It's important to pass in our custom collate function, so it knows how to interpret the 
        # data and load it. num_workers tells the DataLoader how many CPU threads to use so that data can be loaded in parallel, which is faster
        if len(train_split) > 0:
            train_dl = DataLoader(train_split, 
                                batch_size=batch_sz, 
                                shuffle=True, 
                                num_workers=NUM_DATA_WORKERS,
                                collate_fn=train_collate_func)            
        else:
            train_dl = None

        if len(val_split) > 0:
            val_dl = DataLoader(val_split, 
                                batch_size=batch_sz, 
                                shuffle=False, 
                                num_workers=NUM_DATA_WORKERS,
                                collate_fn=val_collate_func)
        else:
            val_dl = None

        if len(test_split) > 0:
            test_dl = DataLoader(test_split,
                                batch_size=batch_sz,
                                shuffle=False,
                                num_workers=NUM_DATA_WORKERS,
                                collate_fn=val_collate_func)
        else:
            test_dl = None

    return train_dl, val_dl, test_dl, split_seed

