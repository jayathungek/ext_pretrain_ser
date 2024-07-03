import math
from pathlib import Path
from typing import List

import numpy as np
import torch
from pprint import pformat
from torchmetrics.classification import MulticlassRecall, MulticlassConfusionMatrix


class ConfusionMatrixMeter:
    def __init__(self, num_classes, human_readable_labels) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.text_labels = human_readable_labels
        self.reset()
        
    def reset(self):
        self.count = 0
        self.recall = MulticlassRecall(self.num_classes, average=None)
        self.conf_matrix = MulticlassConfusionMatrix(num_classes=self.num_classes)
        self.class_recalls = torch.zeros(self.num_classes).cpu()

    def get_uar(self, pred, target):
        recalls_for_each_class = self.recall(pred, target)
        return recalls_for_each_class
    
    def save_confusion_matrix(self, name):
        fig, _ = self.conf_matrix.plot(labels=self.text_labels)
        fig.savefig(f"{name}_confusion_matrix.png")
        
    def update(self, preds, targets):
        self.count += 1
        self.class_recalls += self.recall(preds, targets)
        self.conf_matrix.update(preds, targets)
    
    def get_state(self):
        state = self.conf_matrix.metric_state['confmat']
        return pformat([[i.item() for i in row] for row in list(state)])

    def uweighted_avg_recall(self, average_across_classes=True):
        uar = self.class_recalls / self.count
        if average_across_classes:
            uar = uar.mean().item()
        else:
            uar = [v.item() for v in list(uar)]
        return uar
        


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        n = float(n)
        self.sum += val * n
        self.count += n
    
    def avg(self):
        return (self.sum / self.count)


def frame_to_audio_sample(frame: int, fps: int, audio_sr: int) -> int:
    timestamp = frame * (1 / fps)
    sample = int(audio_sr * timestamp)
    return sample


def _get_clip_start_frames(total_frames: int, clip_length: int):
    max_dist = total_frames - (2 * clip_length)
    frames = np.arange(max_dist, dtype=float)
    probs = frames/frames.sum()
    probs = -probs + probs[-1]
    dist = np.random.choice(frames, 1, p = probs)[0]
    s1_choices = np.arange(0, total_frames - (2 * clip_length) - dist) 
    s1 = np.random.choice(s1_choices, 1)[0]
    s2 = s1 + clip_length + dist
    return int(s1), int(s2)

def get_clip_start_frames(total_frames: int, clip_length: int):
    max_dist = total_frames - clip_length
    if max_dist == 0:
        return 0, 0 # clip len is the same as total_frames, just return the same frame twice
    frames = np.arange(max_dist, dtype=float)
    probs = frames/frames.sum()
    probs = -probs + probs[-1]
    dist = np.random.choice(frames, 1, p = probs)[0]
    s1_choices = np.arange(0, total_frames - clip_length - dist) 
    s1 = np.random.choice(s1_choices, 1)[0]
    s2 = s1 + dist
    return int(s1), int(s2)


def is_jupyter():
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is None:
            # we have IPython installed but not running from IPython
            return False
        else:
            return True
    except:
        # We do not even have IPython installed
        return False

def gather_hparams():
    try:
        import extpt.constants as c
    except ImportError:
        import constants as c
    params = {}
    for k, v in c.__dict__.items():
        if not k.startswith("__"):
            v = v if not v.__class__.__name__ == "bytes" else str(v)
            params[k] = v
    return params
    
def get_best_checkpoint(checkpoints: List[str]):
    lowest = math.inf
    best = None
    for chkpt in checkpoints:
        checkpt_path = Path(chkpt)
        sections = checkpt_path.stem.split("_") 
        loss_val = float(sections[-1])
        if loss_val < lowest:
            lowest = loss_val
            best = checkpt_path.absolute()
    return best

if __name__ == "__main__":
    gather_hparams()
