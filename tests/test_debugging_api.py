# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import unittest

import numpy as np
import torch
from polygraphy.backend.trt import EngineFromNetwork, TrtRunner
from torch import nn

import tensorrt_llm
from tensorrt_llm import Module, Tensor


class TorchMLP(nn.Module):

    def __init__(self, hidden_size, ffn_hidden_size, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.fc1 = nn.Linear(hidden_size, ffn_hidden_size, bias=bias)
        self.fc2 = nn.Linear(ffn_hidden_size, hidden_size, bias=bias)

    def forward(self, x):
        x = x.float()
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


class TensorRTMLP(Module):

    def __init__(self, hidden_size, ffn_hidden_size, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.fc1 = Tensor(
