import torch
import torch.nn as nn

from extpt.resnet.resnet import get_pretrianed_resnet
from extpt.constants import *
from extpt.datasets import enterface
from extpt.data import load_data, Collate_Constrastive
 

if __name__ == "__main__":
    res = get_pretrianed_resnet(PRETRAINED_CHKPT_RESNET)
    res = nn.DataParallel(res).to(DEVICE)
    collate_fn = Collate_Constrastive(enterface)
    train_dl, val_dl, _, _ = load_data(enterface, 
                                        nlines=None,
                                        batch_sz=32,
                                        train_val_test_split=[1.0, 0.0, 0.0],
                                        train_collate_func=collate_fn,
                                        val_collate_func=collate_fn,
                                        shuffle_manifest=False,
                                        seed=None)


    data = next(iter(train_dl))
    clip0_rgb, clip0_spec = data["clip0"]
    clip0_rgb = clip0_rgb[:, : , 0, :, :]
    with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=USE_AMP), torch.backends.cuda.sdp_kernel(enable_flash=False), torch.no_grad():
        out = res(clip0_rgb, mode="video")
        print(out.shape)

