import os
import sys
import tempfile
import unittest
from itertools import product

import numpy as np
import pytest
import tensorrt as trt
import torch
from parameterized import parameterized
from transformers import GPTJConfig, GPTJForCausalLM

import tensorrt_llm
from tensorrt_llm import Builder
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin.plugin import ContextFMHAType

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.util import getSMVersion


class TestGPTJ(unittest.TestCase):

    def _gen_hf_gpt_j(self, hidden_act, n_layer, max_length, dtype):
        gpt_config = GPTJConfig(activation_function=hidden_act,
                                n_layer=n_layer,
                                max_length=max_length,
                                torch_dtype=dtype,
                                n_embd=4096,
                                n_head=16,
                                rotary_dim=64)
        hf_gpt = GPTJForCausalLM(gpt_config).cuda().to(
            tensorrt_llm._utils.str_dtype_to_torch(dtype)).eval()
        return gpt_config, hf_gpt

    def _gen_tensorrt_llm_network(self, network, hf_gpt, gpt_config, batch_size,
                                  beam_width, input_len, output_len, fp16,
                                  tensor_parallel):
        num_layers = gpt_config.n_layer
        num_heads = gpt_config.n_head
        hidden_size = gpt_config.n_embd
        vocab_size = gpt_config.vocab_size
        hidden_act = gpt_config.activation_function
        n_positions = gpt_config.n_positions
        rotary_dim = gpt_config.rotary_dim

        with net_guard(network):
            kv_dtype = trt.float16 if fp16 else trt.float32
            # Initialize model
            tensorrt_llm_gpt = tensorrt_llm.models.GPTJForCausalLM(
                num_layers=num_layers,
                num_heads=num_heads,
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                hidden_act=hidden_act,
                max_position_embeddings=n_positions,
                rotary_dim=rotary_dim,
                dtype=kv_dtype,
                mapping=tensorrt_llm.Mapping(world_size=tensor_parallel,
                                             tp_size=tensor_parallel),
            )
            inputs = tensorrt_llm_gpt.prepare_inputs(batch_size,
                                                     input_len,
                                                     output_len,
                                                     use_cache=True,
                                                     max_beam_width=beam_width)
            load_from_hf_gpt_j(tensorrt_llm_gpt, hf_gpt, fp16=fp16)

            # Prepare
            network.set_named_parameters(tensorrt_llm_gpt.named_parameters())

            tensorrt_llm_gpt(*inputs)

        return network

    def _gen_tensorrt_llm_runtime(self,
                                  dtype,
                                  world_size,
                                  rank,
                                  gpt_config,
                                  hf_gpt,
                                  use_attention_plugin,
                                  batch_size,
                                  beam_width,
                                  input_len,
                                  output_len,
                                  use_refit,
                                  use_ln_gemm_plugin,
                                  context_fmha_flag=ContextFMHAType.disabled,
                                  enable_remove_input_padding=False,
                                  use_rope_plugin=False):
        tensorrt_llm.logger.set_level('error')
        mapping = tensorrt_llm.Mapping(world_size, rank, tp_size=world_size)

        runtime = None
        builder = Builder()
        fp16 = (dtype == 'float16')

        with tempfile.TemporaryDirectory() as tmpdirname:
            network = builder.create_network()
            if use_attention_plugin:
                network.plugin_config.set_gpt_attention_plugin(dtype)
            if use_ln_gemm_plugin:
                network.plugin_config.set_gemm_plugin(dtype)
                network.plugin_config.set_layernorm_plugin(dtype)
            if enable_remove_input_padding:
                network.plugin_config.enable_remove_input_padding()
            if use_rope_plugin:
                network.plugin_config.set_rope_plugin(dtype)
            network.plugin_config.set_context_fmha(context_fmha_flag)

            self._gen_tensorrt_llm_network(network, hf_gpt, gpt_config,
                                           batch_size, beam_width, input_len,
                                           output_len, fp16, world_size)

            builder_config = builder.create_builder_config(
                name='gptj',
                precision=dtype,
                timing_cache='model.cache',
                tensor_parallel=world_size,  # TP only
                use_refit=use_refit,
            )
            engine_buffer = builder.build_engine(network, builder_config)
            assert engine_buffer is not None
            runtime = tensorrt_llm.runtime.generation._Runtime(
                engine_buffer, mapping)

            ok = builder.save_timing_cache(builder_config, 'model.cache')
            assert ok, "Failed to save timing cache."
