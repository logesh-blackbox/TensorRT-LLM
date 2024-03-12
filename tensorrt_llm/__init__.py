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

import tensorrt_llm.functional as functional  # Functional API for building models
import tensorrt_llm.models as models  # Pre-defined models
import tensorrt_llm.quantization as quantization  # Quantization utilities
import tensorrt_llm.runtime as runtime  # Runtime utilities
import tensorrt_llm.tools as tools  # Miscellaneous utilities

# Internal imports
from ._common import _init, default_net, default_trtnet, precision  # Configuration and utilities
from ._utils import mpi_rank, mpi_world_size, str_dtype_to_trt  # MPI utilities and data type conversions
from .builder import Builder, BuilderConfig  # Model builder and configuration
from .functional import Tensor  # Tensor class for model building
from .logger import logger  # Logger
from .mapping import Mapping  # Mapping between TensorRT and Torch data types
from .module import Module  # Torch-like module for TensorRT engines
from .network import Network, net_guard  # Network class for model building
from .parameter import Parameter  # Parameter class for model building

# Export all public names
__all__ = [
    'logger',
    'str_dtype_to_trt',
    'str_dtype_to_torch',
    'mpi_rank',
    'mpi_world_size',
    'constant',
    'default_net',
    'default_trtnet',
    'precision',
    'net_guard',
    'Network',
    'Mapping',
    'Builder',
    'BuilderConfig',
    'Tensor',
    'Parameter',
    'runtime',
    'Module',
    'functional',
    'models',
    'quantization',
    'tools',
]

# Initialize logging
_init(log_level="error")
