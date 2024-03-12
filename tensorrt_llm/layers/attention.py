import math
from typing import List, Optional

import numpy as np
import tensorrt as trt

from .._common import default_net, precision
from ..functional import (AttentionMaskType, PositionEmbeddingType,
                          RotaryScalingType, Tensor, bert_attention, cast, clip,
                          concat, constant, expand_mask, generate_alibi_biases,
                          generate_alibi_slopes, gpt_attention, matmul, round,
                          shape, slice, softmax, split)
from ..module import Module
from ..parameter import Parameter
from ..quantization import QuantMode
from ..quantization.layers import FP8Linear, FP8RowLinear
from .linear import ColumnLinear, RowLinear


class AttentionParams:

    def __init__(self,
                 sequence_length: Tensor = None,
                 context_lengths: Tensor = None,
                 host_context_lengths: Tensor = None,
                 max_context_length: int = None,
                 host_request_types: Tensor = None,
                 encoder_input_lengths: Tensor = None,
                 encoder_max_input_length: Tensor = None):
        self.sequence_length = sequence_length
        self.context_lengths = context_lengths
        self.host_context_lengths = host_context_lengths
        # max allowed context lengths. Required to
        # compute scratch memory size.
        self.max_context_length = max_context_length
        self.host_request_types = host_request_types

        self.encoder_input_lengths = encoder_input_lengths
        self.encoder_max_input_length = encoder_max_input_length

    def is_valid_cross_attn(self, do_cross_attention):
        if self.encoder_input_lengths is None:
            return False
        if self.encoder_max_input_length is None:
            return False

    def is_valid(self, gpt_attention_plugin, remove_input_padding):
        if gpt_attention_plugin:
            if self.sequence_length is None:
                return False
            if self.context_lengths is None:
                return False
            if self.host_request_types is None:
                return False
            if self.max_context_length is None:
                return False

        if remove_input_padding:
            if self.host_context_lengths is None:
                return False
            if not gpt_attention_plugin:
                return False

        return True


class KeyValueCacheParams:

    def __init__(self,
                 past_key_value: List[Tensor] = None,
                 host_past_key_value_lengths: Tensor = None,
                 kv_cache_block_pointers: List[Tensor] = None,
                 cache_indirection: Tensor = None,
                 past_key_value_length: Tensor = None):
        self.past_key_value = past_key_value
        self.host_past_key_value_lengths = host_past_key_value_lengths
        self.kv_cache_block_pointers = kv_cache_block_pointers
        self.cache_indirection = cache_indirection
        # self.past_key_value_length = past_key_value_length

    def get_first_past_key_value(self):
        if self.past_key_value is None:
            return None
        return self.past_key_value[0]

    def get_first_kv_cache_block_pointers(self):
        if self.kv_cache_block_pointers is None:
            return None
        return self.kv_cache_block_pointers[0]

    def is_valid(self, gpt_attention_plugin):
        if gpt_attention_plugin:
            if self.host_past_key_value_lengths is None:
                return False
            if self.cache_indirection is None:
                return False

        return True


class Attention(Module):

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 num_kv_heads=None,
                 max_position_embeddings=1024,
                 num_layers=1,
                 apply_query_key_layer_scaling=False,
                 attention_mask_type=AttentionMaskType.padding,
                 bias=True,
                 dtype=None,
                 position_embedding_type=PositionEmbeddingType.learned_absolute,
                 rotary_embedding_base=10000.0,
                 rotary_embedding_scaling=None,
                 use_int8_kv_cache=False,
                 rotary_embedding_percentage=1.0,
                 tp_group=None,
                 tp_size=1,
                 tp_rank=0,
                 multi_block_mode=False,
                 quant_mode: QuantMode = QuantMode(0),
                 q_scaling=1.0,
                 cross_attention=False,
                 relative_attention=False,
                 max_distance=0,
                 num_buckets=0,
                 instance_id: int = 0):
        super().__init__()

        self.cross_attention = cross_attention
        self.attention_mask_type = attention_mask_type
        self.attention_head_size = hidden_size // num_attention_heads
        assert num_attention_heads % tp_size == 0, \
        "num_attention_heads must be divisible by tp_size"
        self.num_attention_heads = num_attention_heads // tp_size
        self.num_attention_kv_heads = (
            num_kv_heads + tp_size - 1
        ) // tp_size if num_kv_heads is not None else self.num_attention_heads
        self.hidden_size = hidden_size // tp_size
        self.max_position_embeddings = max_position_embeddings
        self.tp_size
