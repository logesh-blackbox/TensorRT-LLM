import argparse
import os
import subprocess
import sys
import typing
from glob import glob
from pathlib import Path

import torch.multiprocessing as mp

RESOURCES_DIR = Path(__file__).parent.parent.parent.parent.parent / "examples" / "chatglm6b"
ENGINE_TARGET_PATH = Path(__file__).parent.parent / "models" / "rt_engine" / "chatglm6b"


def run_command(command: typing.Sequence[str], *, cwd=None, **kwargs) -> None:
    command = [str(i) for i in command]
    print(f"Running: cd {str(cwd or Path.cwd())} && {' '.join(command)}")
    subprocess.check_call(command, cwd=cwd, **kwargs)


def build_engine(weight_dir: Path, engine_dir: Path, *args) -> None:
    print("Additional parameters: " + " ".join(args[0]))
    arg = _ecb.parse_arguments()
    arg.model_dir = str(weight_dir)
    arg.output_dir = str(engine_dir)
    arg.max_batch_size = 2
    arg.max_beam_width = 2
    for item in args[0]:
        key, value = item.split(" ")
        if key[2:] in dir(arg):
            arg.__setattr__(key, value)
        else:
            print("Error parameter name:", key)
            return
    _ecb.build(0, arg)


def build_engines(model_cache: typing.Optional[str] = None, world_size: int = 1) -> None:
    # Clone the model directory
    hf_dir = RESOURCES_DIR / "pyTorchModel"
    ft_dir = RESOURCES_DIR / "ftModel"
    trt_dir = RESOURCES_DIR / "trtModel"

    run_command(["pip", "install", "-r", str(RESOURCES_DIR) + "/requirements.txt"],
                cwd=RESOURCES_DIR)

    if not hf_dir.exists():
        hf_dir.mkdir()

    if not list(hf_dir.glob("*")):
        run_command(["git", "clone", "https://huggingface.co/THUDM/chatglm-6b", hf_dir],
                    cwd=RESOURCES_DIR)

    if not (RESOURCES_DIR / "lm.npy").exists():
        print("Exporting weight of LM")
        run_command(["cp", str(hf_dir) / "modeling_chatglm.py",
                     str(hf_dir) / "modeling_chatglm.py-backup"],
                    cwd=RESOURCES_DIR)
        run_command(["cp", str(RESOURCES_DIR) / "modeling_chatglm.py",
                     str(hf_dir) / "modeling_chatglm.py"],
                    cwd=RESOURCES_DIR)
        run_command(["python3", str(RESOURCES_DIR) / "exportLM.py"],
                    cwd=RESOURCES_DIR)
        assert (RESOURCES_DIR / "lm.npy").exists()
        run_command(["mv", str(hf_dir) / "modeling_chatglm.py-backup",
                     str(hf_dir) / "modeling_chatglm.py"],
                    cwd=RESOURCES_DIR)

    if not list(ft_dir.glob("*")):
        print("\nConverting weight")
        run_command(
            ["python3", str(RESOURCES_DIR) / "hf_chatglm6b_convert.py", "-i",
             hf_dir, "-o", ft_dir, "--storage-type", "fp16"],
            cwd=RESOURCES_DIR)

    if not list(trt_dir.glob("*")):
        print("\nBuilding engine")
        arg_list = ["--dtype", "float16", "--use_gpt_attention_plugin",
                    "float16", "--use_gemm_plugin", "float16"]
        build_engine(ft_dir, trt_dir, arg_list)

    if not ENGINE_TARGET_PATH.exists():
        ENGINE
