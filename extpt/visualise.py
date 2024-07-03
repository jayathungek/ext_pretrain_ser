from typing import List, Tuple
from types import ModuleType
from tqdm import tqdm
import pickle

import torch
from umap import UMAP
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from extpt.data import load_data, Collate_Constrastive
    from extpt.constants import *
    from extpt.clip import CLIPMBT
except ModuleNotFoundError:
    from data import load_data, Collate_Constrastive
    from constants import *
    from clip import CLIPMBT


# TODO: Implement Calinski-Harabasz Index algorithm to check the clustering performance 
# TODO: Replace tsne with umap 
def reduce_dimensions(embeddings: np.ndarray, perplexity) -> np.ndarray:
    reducer =UMAP(n_components=2, metric="cosine", random_state=42)
    reducer.fit(embeddings)
    tsne_points = reducer.transform(embeddings)
    return tsne_points


class Visualiser:
    def __init__(self, dataset_const_namespace: ModuleType) -> None:
        self.ds_constants = dataset_const_namespace
        self.modalities = len(self.ds_constants.MODALITIES)

    def label_to_human_readable(self, label_tensor: torch.tensor) -> List[str]:
        assert len(label_tensor.shape) == 1, f"tensor {label_tensor} has shape {label_tensor.shape}"
        if self.ds_constants.MULTILABEL:
            labels_readable = [self.ds_constants.LABEL_MAPPINGS[i] for i, item in 
                                enumerate(label_tensor.tolist()) if item == 1]
        else:
            labels_readable = [self.ds_constants.LABEL_MAPPINGS[i] for i in label_tensor.tolist()]

        if len(labels_readable) == 0:
            labels_readable = ["None"]

        return labels_readable

    def load_model(self, best_checkpoint):
        save_items = torch.load(best_checkpoint)
        cmbt = CLIPMBT(enterface) 
        cmbt_state_dict = save_items["cmbt_state_dict"]
        cmbt = nn.DataParallel(cmbt).cuda()
        cmbt = cmbt.module.load_state_dict(cmbt_state_dict)
        return cmbt

    def get_embeddings_and_labels_msp(self, dataloader: DataLoader, model: nn.Module) -> Tuple[np.ndarray, List[List[str]]]:
        outputs = []
        labels = []
        other_labels = []
        with torch.no_grad():
            for data in tqdm(dataloader):
                labels_batch = data["labels"]
                other_labels_batch = data["other_labels"]
                audios = data["audios"]
                audios_aug = data["audios_aug"]

                avd, embeddings = model(audios.cuda())
                    
                outputs.append(embeddings)

                for label in labels_batch:
                    flattened_labels = label.flatten()
                    labels.append(self.label_to_human_readable(flattened_labels))
                
                    
                other_labels += other_labels_batch

        sample_output = outputs[0][0]
        output_tensor = torch.FloatTensor(len(dataloader), *sample_output.shape).to(DEVICE)
        outputs = torch.cat(outputs, out=output_tensor)
        outputs = outputs.detach().cpu().numpy()
        print(outputs.shape)
        assert outputs.shape[0] == len(labels), f"Have {outputs.shape[0]} embeddings and {len(labels)} labels!"
        assert outputs.shape[0] == len(other_labels), f"Have {outputs.shape[0]} embeddings and {len(other_labels)} speaker IDs!"
        return outputs, labels, other_labels

    def get_embeddings_and_labels(self, dataloader: DataLoader, model: nn.Module) -> Tuple[np.ndarray, List[List[str]]]:
        outputs = []
        labels = []
        other_labels = []
        with torch.no_grad():
            for data in tqdm(dataloader):
                labels_batch = data["labels"]
                other_labels_batch = data["other_labels"]
                clip0_rgb, clip0_spec = data["clip0"]


                # use the first frame of the video
                # _, embeddings = model(clip0_spec.cuda(), clip0_rgb.cuda())
                if self.modalities == 2:
                    embeddings = model(clip0_spec.cuda(), clip0_rgb.cuda())
                elif self.modalities ==1:
                    _, embeddings = model(clip0_spec.cuda(), None)
                else:
                    raise ValueError("Num modalities must be 1 or 2")
                    
                # use the audio batch
                # clip0_features, _, _ = model(clip0_spec.cuda())
                outputs.append(embeddings)

                for label in labels_batch:
                    flattened_labels = label.flatten()
                    labels.append(self.label_to_human_readable(flattened_labels))
                
                    
                other_labels += other_labels_batch

        sample_output = outputs[0][0]
        output_tensor = torch.FloatTensor(len(dataloader), *sample_output.shape).to(DEVICE)
        outputs = torch.cat(outputs, out=output_tensor)
        outputs = outputs.detach().cpu().numpy()
        print(outputs.shape)
        assert outputs.shape[0] == len(labels), f"Have {outputs.shape[0]} embeddings and {len(labels)} labels!"
        assert outputs.shape[0] == len(other_labels), f"Have {outputs.shape[0]} embeddings and {len(other_labels)} speaker IDs!"
        return outputs, labels, other_labels


    def do_inference_and_save_embeddings(self, model_path: str, split_seed: int=None, use_train: bool=False) -> Tuple[np.ndarray, List[List[str]]]:
        pkl_path = f"saved_models/{CHKPT_NAME}{'_TRAIN_POINTS' if use_train else ''}.pkl"
        collate_fn = Collate_Constrastive(enterface, force_audio_aspect=True)
        train_dl, val_dl, _, _ = load_data(self.ds_constants, 
                                            batch_sz=BATCH_SZ,
                                            train_val_test_split=SPLIT,
                                            collate_func=collate_fn,
                                            seed=split_seed)

        split_to_use = train_dl if use_train else val_dl
        model = self.load_model(model_path)
        embeddings, labels = self.get_embeddings_and_labels(split_to_use, model)
        struct = {"embeddings": embeddings, "labels": labels}
        with open(pkl_path, "wb") as fh:
            pickle.dump(struct, fh)


if __name__ == "__main__":
    from datasets import enterface

    CHKPT_NAME = "saved_models/utt_video.offset.contrastive.labeled1.0/clipmbt_val_loss_2.95574.pth"
    v = Visualiser(enterface)
    v.do_inference(CHKPT_NAME, save_action="attn_map")
    # v.do_inference_and_save_embeddings(f"saved_models/{CHKPT_NAME}.pth", split_seed=6816054967144557969, use_train=True)