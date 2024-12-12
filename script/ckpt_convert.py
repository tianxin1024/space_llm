# https://github.com/NVIDIA/FasterTransformer/blob/main/docs/gpt_guide.md

import argparse
import configparser
import datetime
import json
import multiprocessing
import pathlib
import re
import shutil
import sys

import numpy as np
import torch  # pytype: disable=import-error

import ipdb

# from examples.pytorch.gpt.utils.gpt import DEFAULT_START_TAG, DEFAULT_END_TAG, OPENAI_GPT2_START_ID, OPENAI_GPT2_END_ID
# from examples.pytorch.utils import torch2np, safe_transpose, cpu_map_location, gpu_map_location, WEIGHT2DTYPE


def cpu_map_location(storage, loc):
    return storage.cpu()

def _get_checkpoint_name(checkpoint_dir):

    checkpoint_dir = pathlib.Path(checkpoint_dir)
    patterns = [
        "model_optim_rng.pt",  # older megatron checkpoints
        "*last.ckpt",  # newer format of checkpoints
        "llama2_7b.bin",
    ]
    for pattern in patterns:
        model_files = sorted(list(checkpoint_dir.rglob(pattern)))
        if model_files:
            return model_files[0].name

    raise ValueError(f"Could not find checkpoint files in {checkpoint_dir}")

def convert_checkpoint(args):
    saved_dir = pathlib.Path(args.saved_dir) / f"{args.infer_gpu_num:d}-gpu"
    # if saved_dir.exists():
    #     print(f"[ERROR] Remove {saved_dir} target directory before running conversion")
    #     sys.exit(1)
    # saved_dir.mkdir(parents=True)

    if args.vocab_path:
        shutil.copy(args.vocab_path, (saved_dir / "vocab.json").as_posix())
    if args.merges_path:
        shutil.copy(args.merges_path, (saved_dir / "merges.txt").as_posix())

    load_checkpoints_to_cpu = bool(args.load_checkpoints_to_cpu)
    print(load_checkpoints_to_cpu)
    # map_location_fn = cpu_map_location if load_checkpoints_to_cpu else gpu_map_location
    map_localtion_fn = cpu_map_location

    ipdb.set_trace()
    checkpoints_dir = pathlib.Path(args.in_file)
    checkpoint_name = _get_checkpoint_name(checkpoints_dir)

    # load position_embedding from rank 0
    checkpoints_paths = sorted(checkpoints_dir.rglob(checkpoint_name))
    if not checkpoints_paths:
        print(f"[ERROR] Cannot find checkpoint in {checkpoints_dir}.")
        exit(1)
    model_00 = torch.load(checkpoints_paths[0].as_posix(), map_location=map_localtion_fn)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--saved-dir", "-saved_dir", "-o", help="folder name of output files", required=True)
    parser.add_argument(
        "--in-file", "-in_file", "-i", help="file name of input checkpoint file", required=True
    )
    parser.add_argument(
        "--infer-gpu-num", "-infer_gpu_num", "-i_g", type=int, help="How many gpus for inference", required=True
    )
    # -h_n and -t_g are needed when megatron_ckpt_version = 0, for example the public megatron 345M gpt model
    parser.add_argument(
        "--head-num",
        "-head_num",
        "-h_n",
        type=int,
        help="The number of heads, only needed when weight doesn't contain structure hyperparameters"
    )
    parser.add_argument(
        "--trained-tensor-parallel-size",
        "-trained_tensor_parallel_size",
        "-t_g",
        type=int,
        help="the tensor parallel size for training"
    )
    parser.add_argument(
        "--processes",
        "-processes",
        "-p",
        type=int,
        default=16,
        help="How many processes to spawn for conversion",
    )
    parser.add_argument(
        "--weight-data-type", "-weight_data_type", choices=["fp32", "fp16"], default="fp32", help=""
    )
    parser.add_argument(
        "--load-checkpoints-to-cpu",
        "-load_checkpoints_to_cpu",
        "-cpu",
        type=int,
        choices=[0, 1],
        default=1,
        help="Whether to load model weights to CPU",
    )
    parser.add_argument(
        "--vocab-path",
        type=str,
        help="Path to vocabulary file to embed in FasterTransformer checkpoint",
        required=False,
    )
    parser.add_argument(
        "--merges-path", type=str, help="Path to merges file to embed in FasterTransformer checkpoint", required=False
    )

    args = parser.parse_args()
    print("\n=============== Argument ===============")
    for key in vars(args):
        print(f"{key}: {vars(args)[key]}")
    print("========================================")

    start_time = datetime.datetime.now()
    convert_checkpoint(args)
    run_time = datetime.datetime.now() - start_time
    print(f"[INFO] Spent {run_time} (h:m:s) to convert the model")



if __name__ == "__main__":
    main()
