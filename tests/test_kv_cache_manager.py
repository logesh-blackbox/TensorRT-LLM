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

import torch

import tensorrt_llm
from tensorrt_llm.runtime.kv_cache_manager import (Block, BlocksManager,
                                                   GenerationSequence,
                                                   KVCacheManager)


class TestKVCacheManager(unittest.TestCase):
    _sizeof = {torch.float32: 4, torch.float16: 2, torch.int8: 1}

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    def test_block(self):
        block = Block(block_idx=0, k_ptrs=[123321], v_ptrs=[321123])
        block.add_link()
        self.assertEqual(block.ref_count, 1)

        block.add_link()
        self.assertEqual(block.ref_count, 2)
        self.assertTrue(block.has_link())

        block.remove_link()
        self.assertEqual(block.ref_count, 1)

        block.remove_link()
        self.assertEqual(block.ref_count, 0)
        self.assertFalse(block.has_link())

        self.assertEqual(block.get_k_ptr(0), 123321)
        self.assertEqual(block.get_v_ptr(0), 321123)

    def test_sequence(self):
        seq = GenerationSequence(seq_idx=1, batch_idx=0)
        self.assertEqual(seq.get_batch_idx(), 0)
        self.assertEqual(seq.get_seq_idx(), 1)

        seq1 = GenerationSequence(seq_idx=1, batch_idx=1)
        seq2 = GenerationSequence(seq_idx=1, batch_idx=0)
        seq3 = GenerationSequence(seq_idx=0, batch_idx=0)

        self.assertNotEqual(seq, seq1)
        self.assertEqual(seq, seq2)
        self.assertNotEqual(seq, seq3)

    def allocate_blocks(self, manager, sequences, block_len):
        for _ in range(block_len):
            for seq in sequences:
                self.assertTrue(manager.has_free_block())
                manager.allocate(seq)
        # All blocks should be allocated by now
        self.assertFalse(manager.has_free_block())

    def verify_pointer_array(self,
                             manager,
                             sequences,
                             block_len,
                             total_blocks,
                             max_blocks_per_seq,
                             block_elts,
                             memory_pool,
                             pool_idx=0):
        pointers = manager.get_pointer_array(pool_idx, beam_width=1)

        self.assertEqual(pointers.shape,
                         torch.Size([len(sequences), 1, 2, max_blocks_per_seq]))

        # Check if pointer array is correct
        for seq in sequences:
            for block in range(block_len):
                linear_block_idx = (block * len(sequences) +
                                    seq.get_batch_idx())
                self.assertEqual(pointers[seq.get_batch_idx()][0][0][block], memory_pool.data_ptr() + \
                                 linear_block_idx * block_elts * self._sizeof[memory_pool.dtype])
                self.assertEqual(pointers[seq.get_batch_idx()][0][1][block], memory_pool.data_ptr() + \
                                 (linear_block_idx * block_elts + total_blocks * block_elts) * \
                                    self._sizeof[memory_pool.dtype])

    def free_blocks(self, manager, sequences, block_len):
        for seq in sequences:
            manager.free(seq)
            # We don't have double references to the blocks for now
            self.assertEqual(len(manager.free_blocks),
                             (seq.get_batch_idx() + 1) * block_len)

    def full_allocate_free_test(self, manager, sequences, block_len,
                                total_blocks, max_blocks_per_seq, block_elts,
                                memory_pool):
        self.allocate_blocks(manager, sequences, block_len)

        self.verify_pointer_array(manager, sequences, block_len, total_blocks,
                                  max_blocks_per_seq, block_elts, memory_pool)

        self.free_blocks(manager, sequences, block_len)

    def test_blocks_manager_single_pool(self):
        max_seq = 32
        max_blocks_per_seq = 32
        block_elts = 64
        memory_pool = torch.zeros(max_seq,
                                  2,
                                  max_blocks_per_seq,
                                  block_elts,
                                  dtype=torch.float,
                                  device='cuda')

        sequences = [
            GenerationSequence(seq_idx=idx, batch_idx=idx)
            for idx in range(max_seq)
        ]

        manager = BlocksManager(memory_pools=[memory_pool],
                                blocks=max_seq * max_blocks_per
