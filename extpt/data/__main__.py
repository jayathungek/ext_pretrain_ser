import sys
import argparse

from extpt.data.utils import generate_manifest, save_manifest


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("-d", "--dataset-name")
    # parser.add_argument("-m", "--write-manifest", default=False, action=argparse.BooleanOptionalAction)
    parsed = parser.parse_args(args)
    if parsed.task == "make_manifest" and parsed.dataset_name is None:
        print("Need manifest file to preprocess: -d OR --dataset-name")
        exit(1)
    return parsed


def handle_manifest(dataset_constant_namespace):
    manifest, failed = generate_manifest(dataset_constant_namespace)
    save_filename = f"{dataset_constant_namespace.DATA_DIR}/{dataset_constant_namespace.MANIFEST}" 
    save_manifest(save_filename, manifest)
    print(f"Wrote {len(manifest)} lines to {save_filename} -- {len(failed)} failed")


if __name__ == "__main__":
    parsed_args = parse_args(sys.argv[1:])
    if parsed_args.task == "make_manifest":
        if parsed_args.dataset_name == "enterface":
            from extpt.datasets import enterface
            handle_manifest(enterface)
        elif parsed_args.dataset_name == "asvp":
            from extpt.datasets import asvp
            handle_manifest(asvp)
        elif parsed_args.dataset_name == "librispeech":
            from extpt.datasets import librispeech
            handle_manifest(librispeech)
        elif parsed_args.dataset_name == "iemocap":
            from extpt.datasets import iemocap
            handle_manifest(iemocap)
        elif parsed_args.dataset_name == "tess":
            from extpt.datasets import tess
            handle_manifest(tess)
        elif parsed_args.dataset_name == "ravdess":
            from extpt.datasets import ravdess
            handle_manifest(ravdess)
        else:
            raise ValueError(f"No such dataset: {parsed_args.dataset_name}")