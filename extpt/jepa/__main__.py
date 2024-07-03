import torch

from extpt.jepa.vision_transformer import load_pretrained_videovit
from extpt.constants import *
from extpt.datasets import enterface
from extpt.data import load_data, Collate_Constrastive
 

if __name__ == "__main__":

    collate_fn = Collate_Constrastive(enterface)
    train_dl, val_dl, _, _ = load_data(enterface, 
                                        nlines=1,
                                        batch_sz=1,
                                        train_val_test_split=[1.0, 0.0, 0.0],
                                        train_collate_func=collate_fn,
                                        val_collate_func=collate_fn,
                                        shuffle_manifest=True,
                                        seed=None)


    data = next(iter(train_dl))
    clip0_rgb, clip0_spec = data["clip0"]
    with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=USE_AMP), torch.backends.cuda.sdp_kernel(enable_flash=False), torch.no_grad():
        vvit = load_pretrained_videovit(f"{SAVED_MODELS_PATH}/vitl16.pth")
        out = vvit(clip0_rgb)
        print(out.shape)


