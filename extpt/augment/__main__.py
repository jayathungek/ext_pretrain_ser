import sys
import argparse
from extpt.augment.augment_utils  import adsmote_preprocess, prune_manifest


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("-d", "--dataset-name")
    parser.add_argument("-m", "--write-manifest", default=False, action=argparse.BooleanOptionalAction)
    parsed = parser.parse_args(args)
    if parsed.task == "preprocess" and parsed.dataset_name is None:
        print("Need manifest file to preprocess: -d OR --dataset-name")
        exit(1)
    return parsed


if __name__ == "__main__":
    parsed_args = parse_args(sys.argv[1:])
    if parsed_args.task == "preprocess":
        if parsed_args.dataset_name == "enterface":
            from datasets import enterface
            failed = adsmote_preprocess(enterface)
            if parsed_args.write_manifest:
                prune_manifest(f"{enterface.DATA_DIR}/{enterface.MANIFEST}", failed)
                
        else:
            raise ValueError(f"No such dataset found: {parsed_args.dataset_name}")
    # elif parsed_args.task == "test":
    #     augmetor = MultimodalAugment(enterface)
    #     collate_fn = Collate_Constrastive(enterface, augmentor=augmetor)
    #     train_dl, val_dl, test_dl, split_seed  = load_data(enterface, 
    #                                         batch_sz=5,
    #                                         train_val_test_split=SPLIT,
    #                                         seed=None,
    #                                         nlines=100,
    #                                         collate_func=collate_fn)
        
    #     batch = next(iter(train_dl))
    #     clip0_rgb, clip0_spec = batch["clip0"]
    #     print(clip0_rgb.shape, clip0_spec.shape)
    #     # v_aug = v(clip0_rgb)
    #     # first_frame = clip0_rgb
    #     # aug = v(first_frame)
