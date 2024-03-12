from setuptools import setup, find_packages

setup(
    name="my-project",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "transformers",
    ],
    entry_points={
        "console_scripts": [
            "build_engines = my_project.build_engines:main",
        ],
    },
)


[build-system]
requires = ["setuptools>=42", "wheel"]
build-backend = "setuptools.build_meta"

[tool.black]
line-length = 88

[tool.isort]
profile = "black"

[tool.mypy]
warn_unused_ignores = true
disallow_untyped_defs = true

[tool.pytest]
addopts = "--cov=my_project"


[build_engines]
model_cache = /path/to/model/cache
only_fp8 = false


import argparse
import logging
import os
import pathlib
import subprocess
import sys

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

def build_engine(weight_dir: pathlib.Path, engine_dir: pathlib.Path, *args):
    build_args = [
        sys.executable, "examples/gptj/build.py",
        "--model_dir", str(weight_dir),
        "--output_dir", str(engine_dir),
        "--dtype", "float16",
        "--logits_dtype", "float16",
        "--use_gemm_plugin", "float16",
        "--use_layernorm_plugin", "float16",
        "--max_batch_size", "32",
        "--max_input_len", "40",
        "--max_output_len", "20",
        "--max_beam_width", "2",
        "--log_level", "error",
    ] + list(args)
    subprocess.run(build_args, check=True)

def build_engines(model_cache: str, only_fp8: bool):
    resources_dir = pathlib.Path(__file__).parent.resolve().parent
    models_dir = resources_dir / "models"
    model_name = "gpt-j-6b"

    # Clone or update the model directory without lfs
    hf_dir = models_dir / model_name
    if hf_dir.exists():
        assert hf_dir.is_dir()
        subprocess.run(["git", "pull"], cwd=hf_dir, check=True)
    else:
        if os.name == "nt":
            url_prefix = ""
        else:
            url_prefix = "file://"
        model_url = url_prefix + str(
            pathlib.Path(model_cache) / model_name
        ) if model_cache else "https://huggingface.co/EleutherAI/gpt-j-6b"
        subprocess.run(
            [
                "git",
                "clone",
                model_url,
                "--single-branch",
                "--no-local",
                model_name,
            ],
            cwd=hf_dir.parent,
            env={
                **os.environ, "GIT_LFS_SKIP_SMUDGE": "1"
            },
            check=True,
        )

    assert (hf_dir.is_dir())

    # Download the model file
    model_file_name = "pytorch_model.bin"
    if model_cache:
        if os.name == "nt":
            subprocess.run(
                [
                    "copy",
                    str(
                        pathlib.Path(model_cache) / model_name / model_file_name),
                    model_file_name,
                ],
                cwd=hf_dir,
                check=True,
            )
        else:
            subprocess.run(
                [
                    "rsync",
                    "-av",
                    str(
                        pathlib.Path(model_cache) / model_name / model_file_name),
                    ".",
                ],
                cwd=hf_dir,
                check=True,
            )
    else:
        subprocess.run(
            ["git", "lfs", "pull", "--include", model_file_name],
            cwd=hf_dir,
            check=True,
        )

    assert ((hf_dir / model_file_name).is_file())

    engine_dir = models_dir / "rt_engine" / model_name

    # TODO add Tensor and Pipeline parallel
