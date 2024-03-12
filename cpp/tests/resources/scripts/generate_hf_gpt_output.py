#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import run_hf

def generate_hf_output(data_type, output_name):
    model = 'gpt2'
    resources_dir = Path(__file__).parent.resolve().parent
    models_dir = resources_dir / 'models'
    model_dir = models_dir / model

    data_dir = resources_dir / 'data'
    input_file = data_dir / 'input_tokens.npy'
    output_dir = data_dir / model / 'huggingface'

    run_hf.generate(model_dir=str(model_dir),
                    data_type=data_type,
                    input_file=str(input_file),
                    output_npy=str(output_dir / (output_name + '.npy')),
                    output_csv=str(output_dir / (output_name + '.csv')))

def generate_hf_outputs():
    generate_hf_output(data_type='fp32', output_name='output_tokens_fp32_huggingface')
    generate_hf_output(data_type='fp16', output_name='output_tokens_fp16_huggingface')

if __name__ == '__main__':
    generate_hf_outputs()
