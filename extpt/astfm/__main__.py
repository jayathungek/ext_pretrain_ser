import sys
import argparse

from extpt.astfm.training import start_training
from extpt.datasets import enterface, asvp, librispeech, iemocap
from extpt.astfm.training import start_training
from extpt.train.train_utils import clean_model_dir
from extpt.train.train_utils import get_sorted_checkpoints

def parse_args_train(args):
    parser = argparse.ArgumentParser(prog="start")
    parser.add_argument("-n", "--name")
    parser.add_argument("-w", "--log-wandb", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("-s", "--data-seed", default=None, type=int)
    parser.add_argument("-p", "--pretrained-model", default=None, type=str)
    parser.add_argument("-d", "--dataset", default="enterface", type=str)
    parsed = parser.parse_args(args)
    if parsed.name is None:
        print("Need model name: -n OR --name")
        exit(1)
    if parsed.dataset == "enterface":
        parsed.dataset = enterface
    elif parsed.dataset == "asvp":
        parsed.dataset = asvp
    elif parsed.dataset == "librispeech":
        parsed.dataset = librispeech
    elif parsed.dataset == "iemocap":
        parsed.dataset = iemocap
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
    program = "clipmbt.ast"
    parser = argparse.ArgumentParser(prog=program)
    parser.add_argument("command")
    parsed =  parser.parse_args([args[0]])
    subcommand_args = args[1:]
    if parsed.command == "pretrain":
        parsed_train = parse_args_train(subcommand_args)
        dataset_to_use = parsed_train.dataset
        print(f"Pretraining on dataset: {dataset_to_use}")
        start_training(parsed_train,  dataset_to_use=dataset_to_use)
        clean_model_dir(f"saved_models/{parsed_train.name}", confirmed=True)
    elif parsed.command == "finetune":
        parsed_train = parse_args_train(subcommand_args)
        dataset_to_use = parsed_train.dataset
        pretrained_model_path = get_sorted_checkpoints(f"saved_models/{parsed_train.pretrained_model}")[0]
        assert pretrained_model_path is not None, "Need pretrained model path to finetune!"

        print(f"Finetuning on dataset: {dataset_to_use}")
        start_training(parsed_train, dataset_to_use=dataset_to_use, pretrain_mdl_path=pretrained_model_path)
        clean_model_dir(f"saved_models/{parsed_train.name}", confirmed=True)
    elif parsed.command == "clean":
        parsed_clean = parse_args_clean(subcommand_args)
        clean_model_dir(parsed_clean.dir, parsed_clean.confirmed)
    else:
        print(f"Unrecognised command for {program}: {parsed.command}")



    
if __name__ == "__main__":
    run(sys.argv[1:])