import sys
import argparse
from typing import List

from extpt.datasets import enterface, iemocap, tess, asvp, mspodcast
from extpt.train.training import start_training
from extpt.train.train_utils import clean_model_dir
from extpt.augment.augment import MultimodalAugment
    

def parse_args_train(args):
    parser = argparse.ArgumentParser(prog="start")
    parser.add_argument("-n", "--name")
    parser.add_argument("-v", "--visualise", default="both", choices=["train", "val", "both"])
    parser.add_argument("-w", "--log-wandb", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("-s", "--data-seed", default=None, type=int)
    parser.add_argument("-f", "--checkpoint-freq", default=0, type=int)
    parser.add_argument("-l", "--leave-speaker-out", default=None, nargs="+")
    parser.add_argument("-d", "--dataset", default="enterface", type=str)
    parser.add_argument("-c", "--clean", action="store_true")
    parser.add_argument("-t", "--test-only", action="store_true")
    parser.add_argument("-p", "--pretrained-name", default=None, type=str)
    parser.add_argument("-m", "--mode", default="pretrain", choices=["pretrain", "finetune"])
    parser.add_argument("-e", "--embed-dim", default=1024, type=int)
    parser.add_argument("-z", "--freeze-pt-weights", action="store_true")

    parsed = parser.parse_args(args)
    if parsed.name is None:
        print("Need model name: -n OR --name")
        exit(1)
    if parsed.mode == "finetune" and parsed.pretrained_name is None:
        print("Need name of existing pretrained model in saved_models/ when finetuning: -p or --pretrained-name")
        exit(1)
    if parsed.dataset == "enterface":
        parsed.dataset = enterface
    elif parsed.dataset == "iemocap":
        parsed.dataset = iemocap
    elif parsed.dataset == "tess":
        parsed.dataset = tess
    elif parsed.dataset == "asvp":
        parsed.dataset = asvp
    elif parsed.dataset == "mspodcast":
        parsed.dataset = mspodcast
    else:        
        print(f"Dataset name {parsed.dataset} not known, aborting...")
        exit(2)
    return parsed


def parse_args_clean(args):
    parser = argparse.ArgumentParser(prog="clean")
    parser.add_argument("dir")
    parser.add_argument("-c", "--confirmed", action="store_true")
    parsed = parser.parse_args(args)
    return parsed
    

def run(args):
    parser = argparse.ArgumentParser(prog="extpt.train")
    parser.add_argument("command")
    parsed =  parser.parse_args([args[0]])
    subcommand_args = args[1:]
    if parsed.command == "start":
        parsed_train = parse_args_train(subcommand_args)
        dataset_to_use = parsed_train.dataset
        start_training(parsed_train, dataset_to_use=dataset_to_use)
        clean_model_dir(f"saved_models/{parsed_train.name}", confirmed=parsed_train.clean)
    elif parsed.command == "clean":
        parsed_clean = parse_args_clean(subcommand_args)
        clean_model_dir(parsed_clean.dir, parsed_clean.confirmed)
    elif parsed.command == "scratchpad":
        scratchpad()



def scratchpad():
    """
    To be used as a testing ground where one can import modules and
    run them to debug etc. Access by passing the arg "scratchpad" to
    clipmbt.train
    """
    import torch
    from torch.nn import CosineSimilarity
    from extpt.loss import NeutralAwareLoss, SupConLoss
    from torch.nn.functional import normalize
    from extpt.datasets import tess

    neutrals = torch.tensor([
        [0.5, 0.5], 
        [0.5, 0.5], 
    ])
    emos = torch.tensor([
        [-0.5, -0.5], 
        [-0.5, -0.5], 
    ])

    nal = NeutralAwareLoss()
    l = nal(neutrals.cuda(), emos.cuda())
    print(l)

    exit()
    l = NeutralAwareLoss()
    s = SupConLoss(tess)
    # a = torch.tensor([
    #     [-0.83483301, -0.16904167, 0.52390721],
    #     [ 0.70374682, -0.18682394, -0.68544673],
    #     [ 0.15465702,  0.32303224,  0.93366556],
    #     [ 0.53043332, -0.83523217, -0.14500935],
    #     [ 0.68285685, -0.73054075,  0.00409143],
    #     [ 0.76652431,  0.61500886,  0.18494479]
    # ])
    # b = torch.tensor([
    #     [-0.83455951, -0.16862266, 0.52447767],
    #     [ 0.70374682, -0.18682394, -0.68544673],
    #     [ 0.15465702,  0.32303224,  0.93366556],
    #     [ 0.53043332, -0.83523217, -0.14500935],
    #     [ 0.68285685, -0.73054075,  0.00409143],
    #     [ 0.76652431,  0.61500886,  0.18494479]
    # ])
    a = normalize(torch.rand(20, 256), dim=1)
    b = normalize(torch.rand(20, 256), dim=1)
    conc = torch.cat((a.unsqueeze(1), b.unsqueeze(1)), dim=1)
    loss_l = l(a.cuda(), b.cuda())
    loss_s = s(conc.cuda())
    print(loss_l, loss_s)

    
if __name__ == "__main__":
    run(sys.argv[1:])