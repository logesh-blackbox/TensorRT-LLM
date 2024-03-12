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
from transformers.models.bloom.modeling_bloom import build_alibi_tensor

import tensorrt_llm
from tensorrt_llm import Tensor


class TestFunctional(unittest.TestCase):
    """
    Test class for functional methods in tensorrt_llm.
    """

    def setUp(self):
        """
        Set up the logger to only show error messages during testing.
        """
        tensorrt_llm.logger.set_level('error')

    def create_random_bool_mask(self, batch_size, seq_len):
        """
        Create a random boolean mask with given batch size and sequence length.
        """
        mask = torch.zeros(size=[batch_size, seq_len], dtype=torch.bool)
        seq_lens = torch.randint(low=1, high=seq_len + 1, size=[batch_size])

        for b in range(batch_size):
            mask[b, :seq_lens[b]] = True

        return mask

    @parameterized.expand([(1, 64, 32), (16, 1, 64), (24, 20, 500),
                           (32, 128, 60), (64, 32, 1024), (80, 12, 20),
                           (112, 4, 389)])
    def test_alibi_biases(self, num_heads, batch_size, seq_len):
        """
        Test the generate_alibi_biases function with different parameters.

        :param num_heads: Number of attention heads.
        :param batch_size: Batch size.
        :param seq_len: Sequence length.
        """
        # construct trt network
        builder = tensorrt_llm.Builder()
        net = builder.create_network()

        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()

            # create a fake key tensor
            trt_key = Tensor(name='fake_key',
                             shape=(seq_len, ),
                             dtype=tensorrt_llm.str_dtype_to_trt('int32'))

            # get the key length
            key_len = tensorrt_llm.functional.shape(trt_key, 0)

            # generate alibi slopes
            slopes = tensorrt_llm.functional.generate_alibi_slopes(
                num_heads=num_heads)

            # generate alibi biases
            output = tensorrt_llm.functional.generate_alibi_biases(
                slopes, key_len).trt_tensor

            # mark the output tensor
            output.name = 'output'
            network.mark_output(output)

        # build the TensorRT engine
        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network))

        # create a TRT runner
        with TrtRunner(build_engine) as runner:
            # infer the network with a random key tensor
            outputs = runner.infer(
                feed_dict={
                    'fake_key': np.empty(shape=(seq_len, ), dtype=np.int32)
                })

        # get the TRT alibi output
        trt_alibi_output = outputs['output']

        # create a reference alibi tensor using the transformers library
        binary_mask = self.create_random_bool_mask(batch_size, seq_len)
        ref = build_alibi_tensor(binary_mask, num_heads,
                                 torch.float32).cpu().numpy()

        # reshape the reference tensor
        ref = ref.reshape(batch_size, num_heads, 1, seq_len)

        # create a binary mask for the valid regions
        binary_mask = binary_mask.cpu().numpy().reshape(batch_size, 1, 1,
                                                        seq_len)

        # mask the reference tensor with the binary mask
        ref *= binary_mask

        # repeat the TRT alibi output to match the shape of the reference tensor
        trt_alibi_output = np.repeat(trt_alibi_output, batch_size, axis=0)

        # mask the TRT alibi output with the binary mask
        trt_alibi_output *= binary_mask

        # compare the TRT alibi output with the reference tensor
        np.testing.assert_allclose(ref, trt_alibi_output, atol=1e-3)


if __name__ == "__main__":
    unittest.main()

