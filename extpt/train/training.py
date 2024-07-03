from pprint import pformat
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

import wandb
import torch
import torch.optim.lr_scheduler as lrsch

from extpt.data import load_data
from extpt.train.train_utils import get_train_objects, save_visualisation, is_better
from extpt.constants import *
from extpt.metrics import WandbCtx
from extpt.helpers import gather_hparams, get_best_checkpoint
from extpt.clip import (
    pretrain_avd_guided_contrastive,
    train_supervised_msp,
    val_supervised_msp,
)
    
        
def start_training(parsed_args, dataset_to_use):
    model, loss_func, train_collate_func = get_train_objects(parsed_args.pretrained_name, parsed_args.mode, dataset_to_use, parsed_args.embed_dim)
    val_collate_func = train_collate_func
    if parsed_args.test_only:
        split = TEST_SPLIT 
    else:
        if parsed_args.mode == "pretrain":
            split = SPLIT
        else:
            split = SPLIT_FINETUNE

    train_dl, val_dl, _, split_seed  = load_data(dataset_to_use, 
                                        batch_sz=BATCH_SZ,
                                        train_val_test_split=split,
                                        seed=parsed_args.data_seed,
                                        train_collate_func=train_collate_func,
                                        nlines=NUM_DATASET_RECORDS,
                                        val_collate_func=val_collate_func,
                                        leave_speaker_out=parsed_args.leave_speaker_out
                                        )
                                    
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    print(f"split_seed: {split_seed}")
    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.999), lr=LR, weight_decay=1.0 /16)
    scheduler = lrsch.SequentialLR(
        optimizer=optimizer,
        schedulers=[
        lrsch.ConstantLR(optimizer=optimizer, factor=1, total_iters=WARMUP_EPOCHS),
        lrsch.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=T_0)
    ], milestones=MILESTONES)


    best = gather_hparams() 
    best["LOSS"] = str(loss_func.__class__.__name__)
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
    best["VAL"] = {"acc": None, "loss": None, "conf_matrix": None}
    best["TRAIN"] = {"acc": None, "loss": None}
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
            model.load_state_dict(model_state_dict)
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
            for epoch in range(EPOCHS):
                if parsed_args.mode == "pretrain":
                    val_confm = None
                    split_to_use = "train"
                    metric_to_use = "loss"
                    train_loss = pretrain_avd_guided_contrastive(model, train_dl, optimizer, scaler, loss_fn=loss_func)
                    val_loss = 0
                    train_acc = 0
                    val_acc = 0
                    print(f"Epoch {epoch + 1}: train_loss {train_loss:.5f}\n")

                elif parsed_args.mode == "finetune":
                    split_to_use = "val"
                    metric_to_use = "acc"
                    if not parsed_args.test_only:
                        train_loss, train_acc, _ = train_supervised_msp(model, train_dl, optimizer, scaler, loss_fn=loss_func, freeze_pretrained_weights=parsed_args.freeze_pt_weights)
                        val_loss, val_acc, val_confm = val_supervised_msp(model, val_dl, loss_fn=loss_func)
                    else:
                        print(f"ONLY validation run")
                        train_loss, train_acc = 0, 0
                        val_loss, val_acc, val_confm = val_supervised_msp(model, val_dl, loss_fn=loss_func)
                        
                    val_uar = val_confm.uweighted_avg_recall()
                    val_uar_classes = val_confm.uweighted_avg_recall(average_across_classes=False)
                    print(f"Epoch {epoch + 1}: {train_loss=:.5f}, {val_loss=:.5f}, {train_acc=:.5f}, {val_acc=:5f}\n{val_uar=:5f}\n{val_uar_classes=}\n{val_confm.get_state()}")


                if split_to_use == "val":
                    metric_value = val_loss if metric_to_use == "loss" else val_acc
                elif split_to_use == "train":
                    metric_value = train_loss if metric_to_use == "loss" else train_acc
                    
                scheduler.step() 
                if parsed_args.log_wandb:
                    wandb.log({
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "train_acc": train_acc,
                        "val_acc": val_acc,
                    })

                if parsed_args.checkpoint_freq == 0:
                    if best[split_to_use.upper()][metric_to_use] is None or \
                        (best[split_to_use.upper()][metric_to_use] is not None and \
                        is_better(metric_value, best[split_to_use.upper()][metric_to_use], metric=metric_to_use)): 

                        fname = f"clipmbt_{epoch}_{split_to_use}_{metric_to_use}_{metric_value:.5f}"
                        fpath_chkpt = save_path / f"{fname}.pth"
                        best["T_0"] = T_0
                        best["BEST_EPOCH"] = epoch + 1
                        best["VAL"]["loss"] = val_loss
                        best["TRAIN"]["loss"] = train_loss
                        best["VAL"]["acc"] = val_acc
                        best["TRAIN"]["acc"] = train_acc
                        best["CHECKPOINT_NAME"] = fname
                        if val_confm is not None:
                            best["VAL"]["conf_matrix"] = val_confm.get_state()
                            val_confm.save_confusion_matrix(str(save_path / "val"))
                        save_items = {
                            "model_state_dict": model.state_dict(),
                            "ast_state_dict": model.module.audio_backbone.state_dict(),
                            "optimizer_dict": optimizer.state_dict(),
                            "scaler_dict": scaler.state_dict(),
                            "scheduler_dict": scheduler.state_dict(),
                            "best": best
                        }
                        torch.save(save_items, fpath_chkpt)
                elif epoch % parsed_args.checkpoint_freq == 0:
                    fname = f"clipmbt_{epoch}_{split_to_use}_{metric_to_use}_{metric_value:.5f}"
                    fpath_chkpt = save_path / f"{fname}.pth"
                    best["T_0"] = T_0
                    best["BEST_EPOCH"] = epoch + 1
                    best["VAL"]["loss"] = val_loss
                    best["TRAIN"]["loss"] = train_loss
                    best["VAL"]["acc"] = val_acc
                    best["TRAIN"]["acc"] = train_acc
                    best["CHECKPOINT_NAME"] = fname
                    if val_confm is not None:
                        best["VAL"]["conf_matrix"] = val_confm.get_state()
                        val_confm.save_confusion_matrix(str(save_path / "val"))
                    save_items = {
                        "model_state_dict": model.state_dict(),
                        "ast_state_dict": model.module.audio_backbone.state_dict(),
                        "optimizer_dict": optimizer.state_dict(),
                        "scaler_dict": scaler.state_dict(),
                        "scheduler_dict": scheduler.state_dict(),
                        "best": best
                    }
                    torch.save(save_items, fpath_chkpt)
    except (Exception, KeyboardInterrupt) as e:
        import traceback
        print(f"Fatal error: {traceback.format_exc()}")
    finally:
        print(pformat(best))
        dloaders = {
            "train": train_dl,
            "val": val_dl
        }
        print(f"Saving visualisation for best checkpoint {best['CHECKPOINT_NAME']} (model tested on sets: {parsed_args.visualise})")
        save_visualisation(model, dloaders, dataset_to_use, best, save_path, parsed_args.visualise)