import math
import pickle
from pprint import pformat
from pathlib import Path
from typing import List
import json
import warnings
warnings.filterwarnings('ignore')

import wandb
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lrsch

from extpt.data import load_data, Collate_Constrastive, CollateNoncontrastive
from extpt.constants import *
from extpt.visualise import Visualiser
from extpt.metrics import WandbCtx
from extpt.augment.augment import MultimodalAugment
from extpt.helpers import gather_hparams
from extpt.astfm.emo_ast import EmoAST, pretrain_one_epoch, train_supervised, val_supervised


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


def save_visualisation(model, dataloader, ds_namespace, best_chkpt, save_dir):
    visualiser = Visualiser(ds_namespace)
    embeddings, labels, speaker_ids = visualiser.get_embeddings_and_labels(dataloader, model)
    struct = {"embeddings": embeddings, "labels": labels, "speaker_ids": speaker_ids}
    with open(f"{save_dir}/{best_chkpt['CHECKPOINT_NAME']}_trainset.pkl", "wb") as fh:
        pickle.dump(struct, fh)

    with open(f"{save_dir}/hparams.json", "w") as fh:
        json.dump(best_chkpt, fh)

def start_training(parsed_args, dataset_to_use, pretrain_mdl_path=None):
    pretraining = pretrain_mdl_path is None
    if pretraining:
        emoast = EmoAST(
                    dataset_to_use,
                    fshape=16, tshape=16, fstride=16, tstride=16,
                    input_fdim=128, input_tdim=dataset_to_use.SPEC_MAX_LEN,
                    pretrain_stage=True, load_pretrained_mdl_path=SSAST_MODEL_CHKPT, freeze_first=FREEZE_FIRST)
        augmentor = MultimodalAugment(dataset_to_use) if APPLY_AUG else None
        train_collate_func = CollateNoncontrastive(dataset_to_use, augmentor=augmentor)
        train_dl, _, _, split_seed  = load_data(dataset_to_use, 
                                            batch_sz=BATCH_SZ,
                                            train_val_test_split=SPLIT,
                                            seed=parsed_args.data_seed,
                                            train_collate_func=train_collate_func,
                                            nlines=NUM_DATASET_RECORDS,
                                            val_collate_func=None)
    else:
        emoast = EmoAST(
                    dataset_to_use,
                    fshape=16, tshape=16, fstride=16, tstride=16,
                    input_fdim=128, input_tdim=dataset_to_use.SPEC_MAX_LEN,
                    pretrain_stage=False, load_pretrained_mdl_path=pretrain_mdl_path, freeze_first=0)
        augmentor = MultimodalAugment(dataset_to_use) if APPLY_AUG else None
        train_collate_func = CollateNoncontrastive(dataset_to_use, augmentor=augmentor)
        val_collate_func = CollateNoncontrastive(dataset_to_use, augmentor=None)
        train_dl, val_dl, _, split_seed  = load_data(dataset_to_use, 
                                            batch_sz=BATCH_SZ,
                                            train_val_test_split=SPLIT,
                                            seed=parsed_args.data_seed,
                                            train_collate_func=train_collate_func,
                                            nlines=NUM_DATASET_RECORDS,
                                            val_collate_func=val_collate_func)
        
    emoast = nn.DataParallel(emoast).to(DEVICE)

                                    
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    print(f"split_seed: {split_seed}")
    optimizer = torch.optim.AdamW(emoast.parameters(), betas=(0.9, 0.999), lr=LR, weight_decay=1.0 / 16)
    scheduler = lrsch.SequentialLR(
        optimizer=optimizer,
        schedulers=[
        lrsch.ConstantLR(optimizer=optimizer, factor=1, total_iters=WARMUP_EPOCHS),
        lrsch.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=T_0)
    ], milestones=MILESTONES)


    best = gather_hparams() 
    best["OPTIM"] = str(optimizer.__class__.__name__)
    best["OPTIM_LR"] = optimizer.defaults.get('lr')
    best["OPTIM_MOMENTUM"] = optimizer.defaults.get('momentum')
    best["OPTIM_WEIGHT_DECAY"] = optimizer.defaults.get('weight_decay')
    best["BEST_EPOCH"] = 0
    best["SPLIT_SEED"] = split_seed
    best["V_AUG_ROT_DEGREES"] = dataset_to_use.ROT_DEGREES if APPLY_AUG else None
    best["V_AUG_FLIP_CHANCE"] = dataset_to_use.FLIP_CHANCE if APPLY_AUG else None
    best["V_AUG_INVERT_CHANCE"] = dataset_to_use.INVERT_CHANCE if APPLY_AUG else None
    best["A_AUG_ADSMOTE_GAMMA"] = dataset_to_use.ADSMOTE_GAMMA if APPLY_AUG else None
    best["A_AUG_ADSMOTE_KNNS"] = dataset_to_use.ADSMOTE_KNNS if APPLY_AUG else None
    best["A_AUG_ADSMOTE_POLY_SAMPLES"] = dataset_to_use.ADSMOTE_POLY_SAMPLES if APPLY_AUG else None
    best["VAL"] = {"loss": None}
    best["TRAIN"] = {"loss": None}
    best["CHECKPOINT_NAME"] = None

    experiment_name = parsed_args.name
    save_path = Path(f"saved_models/{experiment_name}")
    if save_path.exists() and save_path.is_dir():
        print(f"Experiment exists, searching for checkpoints...")
        checkpoints = list(save_path.glob("*pth"))
        if len(checkpoints) > 0:
            best_checkpoint = get_best_checkpoint(checkpoints)
            save_items = torch.load(best_checkpoint)
            print(f"Loading from checkpoint {best_checkpoint}")
            best["CHECKPOINT_NAME"] = str(best_checkpoint.stem)
            model_state_dict = save_items["model_state_dict"]
            optimizer_dict = save_items["optimizer_dict"]
            scaler_dict = save_items["scaler_dict"]
            scheduler_dict = save_items["scheduler_dict"]
            emoast.load_state_dict(model_state_dict)
            optimizer.load_state_dict(optimizer_dict)
            scaler.load_state_dict(scaler_dict)
            scheduler.load_state_dict(scheduler_dict)
        else:
            print("No checkpoints found, training from scratch.")
    else:
        save_path.mkdir(exist_ok=False)

    fpath_params = save_path / "hparams.json"
    with fpath_params.open("w") as fh:
        json.dump(best, fh)

    try:
        with WandbCtx(PROJECT_NAME, run_name=experiment_name, config=best, enabled=parsed_args.log_wandb) as run:
            frozen_layers = emoast.module.retrieve_first_n_layers(emoast.module.freeze_first)
            if len(frozen_layers) > 0:
                print(f"Will freeze first {len(frozen_layers)} layers during training.")
            for epoch in range(EPOCHS):
                if pretraining:
                    train_loss = pretrain_one_epoch(emoast, train_dl, optimizer, scaler, frozen_layers=frozen_layers)
                    scheduler.step()
                    print(f"Epoch {epoch + 1}: train_loss {train_loss:.5f}\n")
                    if parsed_args.log_wandb:
                        wandb.log({
                            "train_loss": train_loss,
                        })

                    if best["TRAIN"]["loss"] is None or (best["TRAIN"]["loss"] is not None and train_loss <= best["TRAIN"]["loss"]): 
                        fname = f"clipmbt_train_loss_{train_loss:.5f}"
                        fpath_chkpt = save_path / f"{fname}.pth"
                        best["T_0"] = T_0
                        best["BEST_EPOCH"] = epoch + 1
                        best["TRAIN"]["loss"] = train_loss
                        best["CHECKPOINT_NAME"] = fname
                        save_items = {
                            "model_state_dict": emoast.state_dict(),
                            "optimizer_dict": optimizer.state_dict(),
                            "scaler_dict": scaler.state_dict(),
                            "scheduler_dict": scheduler.state_dict()
                        }
                        torch.save(save_items, fpath_chkpt)
                else:
                    loss_fn = nn.CrossEntropyLoss()
                    train_loss, train_acc = train_supervised(emoast, train_dl, optimizer, scaler, loss_fn)
                    val_loss, val_acc = val_supervised(emoast, val_dl, loss_fn)
                    scheduler.step()
                    print(f"Epoch {epoch + 1}: {train_loss=:.5f}, {train_acc=:.5f}, {val_loss=:.5f}, {val_acc=:.5f}\n")
                    if parsed_args.log_wandb:
                        wandb.log({
                            "train_loss": train_loss,
                            "train_acc": train_acc,
                            "val_loss": val_loss,
                            "val_acc": val_acc,
                        })

                    if best["VAL"]["loss"] is None or (best["VAL"]["loss"] is not None and val_loss <= best["VAL"]["loss"]): 
                        fname = f"clipmbt_train_loss_{train_loss:.5f}"
                        fpath_chkpt = save_path / f"{fname}.pth"
                        best["T_0"] = T_0
                        best["BEST_EPOCH"] = epoch + 1
                        best["TRAIN"]["loss"] = train_loss
                        best["VAL"]["loss"] = val_loss
                        best["CHECKPOINT_NAME"] = fname
                        save_items = {
                            "model_state_dict": emoast.state_dict(),
                            "optimizer_dict": optimizer.state_dict(),
                            "scaler_dict": scaler.state_dict(),
                            "scheduler_dict": scheduler.state_dict()
                        }
                        torch.save(save_items, fpath_chkpt)
                    
    except (Exception, KeyboardInterrupt) as e:
        import traceback
        print(f"Fatal error: {traceback.format_exc()}")
    finally:
        print(pformat(best))
        print(f"Saving visualisation for best checkpoint {best['CHECKPOINT_NAME']}")
        save_visualisation(emoast, train_dl, dataset_to_use, best, save_path)
    