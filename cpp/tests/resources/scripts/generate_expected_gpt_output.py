# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import pathlib
import numpy as np
import csv
import argparse

import run


def generate_output(engine: str,
                    input_file: str,
                    tokenizer_path: str,
                    output_npy: str,
                    output_csv: str,
                    max_output_len: int = 8,
                    num_beams: int = 1):
    """
    Generate output using the specified engine, input file, tokenizer path,
    output NPY file, output CSV file, maximum output length, and number of beams.

    Args:
        engine (str): The engine to use for generation.
        input_file (str): The path to the input file.
        tokenizer_path (str): The path to the tokenizer.
        output_npy (str): The path to the output NPY file.
        output_csv (str): The path to the output CSV file.
        max_output_len (int, optional): The maximum output length. Defaults to 8.
        num_beams (int, optional): The number of beams. Defaults to 1.
    """
    engine_dir = os.path.join(
        pathlib.Path(__file__).parent.resolve().parent,
        'models',
        'rt_engine',
        'gpt2',
        engine,
        'tp1-pp1-gpu'
    )

    run.generate(engine_dir=engine_dir,
                 input_file=input_file,
                 tokenizer_path=tokenizer_path,
                 output_npy=output_npy,
                 output_csv=output_csv,
                 max_output_len=max_output_len,
                 num_beams=num_beams)


def generate_outputs(num_beams):
    """
    Generate GPT2 FP32 and FP16 outputs with the specified number of beams.

    Args:
        num_beams (int): The number of beams.
    """
    print('Generating GPT2 FP32 outputs')
    if num_beams == 1:
        generate_output(engine='fp32-default',
                        input_file=str(pathlib.Path(__file__).parent / 'data' / 'input_tokens.npy'),
                        tokenizer_path=str(pathlib.Path(__file__).parent / 'models' / 'gpt2'),
                        output_npy=str(pathlib.Path(__file__).parent / 'data' / 'gpt2' / 'sampling' / 'output_tokens_fp32.npy'),
                        output_csv=str(pathlib.Path(__file__).parent / 'data' / 'gpt2' / 'sampling' / 'output_tokens_fp32.csv'),
                        max_output_len=8,
                        num_beams=num_beams)
    generate_output(engine='fp32-plugin',
                    input_file=str(pathlib.Path(__file__).parent / 'data' / 'input_tokens.npy'),
                    tokenizer_path=str(pathlib.Path(__file__).parent / 'models' / 'gpt2'),
                    output_npy=str(pathlib.Path(__file__).parent / 'data' / 'gpt2' / 'output_tokens_fp32_plugin.npy'),
                    output_csv=str(pathlib.Path(__file__).parent / 'data' / 'gpt2' / 'output_tokens_fp32_plugin.csv'),
                    max_output_len=8,
                    num_beams=num_beams)

    print('Generating GPT2 FP16 outputs')
    if num_beams == 1:
        generate_output(engine='fp16-default',
                        input_file=str(pathlib.Path(__file__).parent / 'data' / 'input_tokens.npy'),
                        tokenizer_path=str(pathlib.Path(__file__).parent / 'models' / 'gpt2'),
                        output_npy=str(pathlib.Path(__file__).parent / 'data' / 'gpt2' / 'beam_search_1' / 'output_tokens_fp16.npy'),
                
