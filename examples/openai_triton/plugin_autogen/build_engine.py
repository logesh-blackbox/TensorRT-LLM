import argparse
import math
# include plugins
# yapf: disable
import sys
import time
from pathlib import Path
from typing import List, OrderedDict

import tensorrt as trt

# from plugin import LAYER_NAME, FmhaLayer, get_engine_name
import tensorrt_llm
from tensorrt_llm import Module, str_dtype_to_trt
from tensorrt_llm.builder import Builder, BuilderConfig
from tensorrt_llm.functional import Tensor
from tensorrt_llm.logger import logger
from tensorrt_llm.network import net_guard

sys.path.append('./tmp')
from functional import fused_attention_kernel # isort:skip
# yapf: enable


def get_engine_name(head_size: int, dtype: str) -> str:
    return f'fmha_{head_size}_{dtype}.engine'


class FmhaLayer(Module):

    def __init__(self, num_heads: int, head_size: int, softmax_scale: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.softmax_scale = softmax_scale
        self.dtype = str_dtype_to_trt('float16')

    def forward(self, Q: Tensor, K: Tensor, V: Tensor):
        inputs = [Q, K, V]
        Out, L, M = fused_attention_kernel(self.softmax_scale, self.num_heads,
                                           *[p.trt_tensor for p in inputs])
        Out.mark_output('out', self.dtype)
        L.mark_output('L', self.dtype)
        M.mark_output('M', self.dtype)
        return Out, L, M

    def prepare_inputs(self, max_batch_size: int, max_len: int) -> List[Tensor]:
        '''

        @brief: Prepare inputs Tensors for the model, the given sizes are used to
            determine the ranges of the dimensions of when using TRT dynamic shapes.

        @return: a list contains values which can be fed into the self.forward()
        '''

        bs_range = [1, (max_batch_size + 1) // 2, max_batch_size]
        max_len_range = [1, (max_len + 1) // 2, max_len]

        dynamic_shape = [-1, self.num_heads, -1, self.head_size]
        Q = Tensor(name='Q',
                   dtype=trt.float16,
                   shape=dynamic_shape,
                   dim_range=OrderedDict([
                       ('batch_size', [bs_range]),
                       ('num_heads', [self.num_heads]),
                       ('seq_len', [max_len_range]),
                       ('head_size', [self.head_size]),
                   ]))
        K = Tensor(name='K',
                   dtype=trt.float16,
                   shape=dynamic_shape,
                   dim_range=OrderedDict([
                       ('batch_size', [bs_range]),
                       ('num_heads', [self.num_heads]),
                       ('seq_len', [max_len_range]),
                       ('head_size', [self.head_size]),
                   ]))
        V = Tensor(name='V',
                   dtype=trt.float16,
                   shape=dynamic_shape,
                   dim_range=OrderedDict([
                       ('batch_size', [bs_range]),
                       ('num_heads', [self.num_heads]),
                       ('seq_len', [max_len_range]),
                       ('head_size', [self.head_size]),
                   ]))
        return [Q, K, V]


def build_engine(builder: Builder, builder_config: BuilderConfig,
                 engine_name: str, args: argparse.Namespace) -> trt.IHostMemory:
    '''
    @brief: Build a TensorRT engine.
    @param args: The cmd line arguments.
    @return: The built or refitted engine.
    '''

    # Initialize Module
    softmax_scale = 1.0 / math.sqrt(args.head_size)
    layer = FmhaLayer(args.num_heads, args.head_size, softmax_scale)

    # Module -> Network
    network = builder.create_network()
    network.trt_network.name = engine_name
    with net_guard(network):
        # Prepare
        inputs = layer.prepare_inputs(args.max_batch_size, args.max_seq_len)
        # Forward
        logger.debug(f'model inputs: {inputs}')
        layer(*inputs)

        print('dot:')
        print(network.to_dot())

        layer = network.get_layer_by_name(
            "FmhaLayer/PLUGIN_V2_fused_attention_kernelPlugin_2").as_layer()
        print('layer', layer.plugin.plugin_type)
        print('layer', layer.plugin.plugin_version)
        print('layer', layer.plugin.plugin_namespace)

    # Network -> Engine
    engine = builder.build_engine(network, builder_config)
    config_path = Path(args.output_dir) / 'config.json'
    builder.save_config(builder_config, str(config_path))
    return engine


def build(args):
    tensorrt_llm.logger.set_level(args.log_level)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    builder = Builder()
    cache = None
    builder_config = builder.create_builder_config(
        name='fmha_triton',
        precision=args.dtype,
        timing_cache=args.timing_cache if cache is None else cache)

    engine_name = get_engine_name(
