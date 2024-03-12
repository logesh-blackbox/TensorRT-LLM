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

import unittest

import numpy as np
import torch
from parameterized import parameterized
from polygraphy.backend.trt import EngineFromNetwork, TrtRunner

import tensorrt_llm
from tensorrt_llm import Tensor


class TestFunctional(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    @parameterized.expand([('float32', )])
    def test_gelu(self, dtype):
        # torch gelu does not support float16
        # test data
        x_shape = (12, 12, 96, 96)
        x_data = torch.rand(x_shape,
                            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype))

        # construct trt network
        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        with tensorrt_llm.net_guard(net):
            # Import the default TRT network
            network = tensorrt_llm.default_trtnet()

            # Define the input tensor
            x = Tensor(name='x',
                       shape=x_shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))

            # Compute the GELU activation using the functional API
            output = tensorrt_llm.functional.gelu(x).trt_tensor

            # Set the output tensor name
            output.name = 'output'

            # Mark the output tensor
            network.mark_output(output)

        # Build the TensorRT engine
        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network))

        # Create a TRT runner
        with TrtRunner(build_engine) as runner:
            # Perform inference
            outputs = runner.infer(feed_dict={'x': x_data.numpy()})

        # Perform the reference PyTorch computation
        ref = torch.nn.functional.gelu(x_data)

        # Compare the results
        np.testing.assert_allclose(ref.cpu().numpy(),
                                   outputs['output'],
                                   atol=1e-3)
