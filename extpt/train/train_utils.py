import json
import pickle
from pathlib import Path
from logging import Logger
from typing import List

import torch.nn as nn

from extpt.data import CollateContrastiveAudio, CollateContrastiveMSP

from extpt.clip import SSAST_MSP
from extpt.visualise import Visualiser
from extpt.augment.specaugment import Specaugment
from extpt.loss import AVDGuidedContrastiveLoss
from extpt.constants import DEVICE


#TODO: implement project-wide logger with loglevel set by cmd flag
logger = Logger(name="train_utils_logger")

def get_sorted_checkpoints(model_dir: str, sortby="metric"):
    model_dir = Path(model_dir).resolve()
    k = lambda x: float(x.stem.split('_')[-1 if sortby == "metric" else 1]) 
    all_checkpoints = list(model_dir.glob('*.pth'))
    if len(all_checkpoints) > 0:
        metric = all_checkpoints[0].stem.split("_")[3]
        sorted_checkpoints =  sorted(
            all_checkpoints,
            key=k,
            reverse= True if metric == "acc" else False
        )
        return sorted_checkpoints
    else:
        return []
    

def clean_model_dir(model_dir: str, confirmed=True):
    """
    Keeps the *.pth file in a given folder that has the best 
    loss value and deletes the rest of them  
    """
    sorted_checkpoints = get_sorted_checkpoints(model_dir)
    if len(sorted_checkpoints) < 2: return # only 1 checkpoint. Nothing to do, it's already the best one
    
    # always keep the best and first checkpoint
    other_checkpoints = sorted_checkpoints[1:]

    for path in other_checkpoints:
        if confirmed:
            path.unlink() 
            logger.info(f"rm {str(path)}")
        else:
            logger.info(f"[Dry run]: rm {str(path)}")
    

def get_train_objects(model_variant: str, mode: str, dataset_namespace, emb_dim=1024):
    if mode == "pretrain":
        augmentor = Specaugment()
        loss_func = AVDGuidedContrastiveLoss(dataset_namespace)
        collate_func = CollateContrastiveMSP(dataset_namespace, augmentor)
        model = SSAST_MSP(dataset_namespace, pt_model_name=None, emb_dim=1024)
    elif mode == "finetune":
        loss_func = nn.CrossEntropyLoss()
        collate_func = CollateContrastiveAudio(dataset_namespace)
        model = SSAST_MSP(dataset_namespace, pt_model_name=model_variant, emb_dim=emb_dim)
    
    model = nn.DataParallel(model).to(DEVICE)
    return model, loss_func, collate_func
    

def save_visualisation(model, dataloaders, ds_namespace, best_chkpt, save_dir, visualise="both"):
    visualiser = Visualiser(ds_namespace)
    get_emb_func = visualiser.get_embeddings_and_labels_msp

    if visualise == "both":
        embeddings, labels, other_labels = get_emb_func(dataloaders["train"], model)
        struct_train = {"embeddings": embeddings, "labels": labels, "other_labels": other_labels}

        embeddings, labels, other_labels = get_emb_func(dataloaders["val"], model)
        struct_val = {"embeddings": embeddings, "labels": labels, "other_labels": other_labels}

        with open(f"{save_dir}/{best_chkpt['CHECKPOINT_NAME']}_trainset.pkl", "wb") as fh:
            pickle.dump(struct_train, fh)

        with open(f"{save_dir}/{best_chkpt['CHECKPOINT_NAME']}_valset.pkl", "wb") as fh:
            pickle.dump(struct_val, fh)

    elif visualise == "train":
        embeddings, labels, other_labels = get_emb_func(dataloaders["train"], model)
        struct = {"embeddings": embeddings, "labels": labels, "other_labels": other_labels}
        with open(f"{save_dir}/{best_chkpt['CHECKPOINT_NAME']}_trainset.pkl", "wb") as fh:
            pickle.dump(struct, fh)

    elif visualise == "val":
        embeddings, labels, other_labels = get_emb_func(dataloaders["val"], model)
        struct = {"embeddings": embeddings, "labels": labels, "other_labels": other_labels}
        with open(f"{save_dir}/{best_chkpt['CHECKPOINT_NAME']}_valset.pkl", "wb") as fh:
            pickle.dump(struct, fh)

    with open(f"{save_dir}/hparams.json", "w") as fh:
        json.dump(best_chkpt, fh)

def get_submodule(module: nn.Module, indices: List[int]=[]):
    modules = list(module.modules())
    for i in indices:
        module = modules[i]
        modules = list(modules[i].modules())
        
    return module
    
    
def is_better(value1, value2, metric="loss"):
    return value1 <= value2 if metric == "loss" else value1 >= value2