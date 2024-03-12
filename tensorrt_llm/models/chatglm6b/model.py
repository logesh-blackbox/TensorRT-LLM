import math
from collections import OrderedDict

import numpy as np
import tensorrt as trt

from ..._common import default_net
from ..._utils import pad_vocab_size, str_dtype_to_trt
from ...functional import (PositionEmbeddingType, Tensor, assertion, concat,
                           constant, gather_last_token_logits, gpt_attention,
                           shape, split)
from ...layers import (MLP, AttentionMaskType, AttentionParams, ColumnLinear,
                       Embedding, KeyValueCacheParams, LayerNorm, RowLinear)
from ...mapping import Mapping
from ...module import Module, ModuleList
from ...parameter import Parameter
from ...quantization import QuantMode


class ChatGLMAttention(Module):

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 max_position_embeddings,
                 num_layers=1,
                 apply_query_key_layer_scaling=False,
                 bias=True,
                 dtype=None,
                 position_embedding_type:
                 PositionEmbeddingType = PositionEmbeddingType.learned_absolute,
                 use_int8_kv_cache=False,
                 tp_group=None,
                 tp_size=1,
                 multi_block_mode=False,
                 multi_query_mode=False):
        super().__init__()

        self.attention_mask_type = AttentionMaskType.bidirectional
        self.attention_head_size = hidden_size // num_attention_heads
        self.num_attention_heads = num_attention_heads // tp_size
        self.num_attention_kv_heads = 1 if multi_query_mode else self.num_attention_heads
        self.hidden_size = hidden_size // tp_size
        self.max_position_embeddings = max_position_embeddings

        self.num_layers = num_layers
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.norm_factor = math.sqrt(self.attention_head_size)
        self.q_scaling = 1
        if self.apply_query_key_layer_scaling:
            self.norm_factor *= self.num_layers
            self.q_scaling *= self.num_layers

        self.multi_block_mode = multi_block_mode
        self.multi_query_mode = multi_query_mode

        self.rotary_embedding_dim = 0
        self.position_embedding_type = position_embedding_type
        self.dtype = dtype

        self.use_int8_kv_cache = use_int8_kv_cache
        if self.use_int8_kv_cache:
            self.kv_orig_quant_scale = Parameter(shape=(1, ), dtype='float32')
            self.kv_quant_orig_scale = Parameter(shape=(1, ), dtype='float32')
        else:
            self.register_parameter('kv_orig_quant_scale', None)
            self.register_parameter('kv_quant_orig_scale', None)

        # Note: in multi_query_mode, only query heads are split between multiple GPUs,
        # while key/value head are not split as there is only one head per key/value.
        # The output feature size is therefore (h/tp + 2) * d, where h is num_heads,
        # d is head_size, and tp is tensor_parallel_size.
        # In ColumnLinear op, the output dim is calculated by (h + 2*tp) * d / tp,
        # which matches the desired output size (h/tp + 2) * d after splitting
        self.qkv = ColumnLinear(hidden_size,
                                hidden_size *
                                3 if not multi_query_mode else hidden_size +
                                2 * tp_size * self.attention_head_size,
                                bias=bias,
                                dtype=dtype,
                                tp_group=tp_group,
                                tp_size=tp_size,
                                gather_output=False)
        self.dense = RowLinear(hidden_size,
                               hidden_size,
                               bias=bias,
                               dtype=dtype,
                               tp_group=tp_group,
                               tp_size=tp_size)

    def forward(self,
                hidden_states: Tensor,
                position_embedding,
                use_cache=False,
                kv_cache_params=None,
                attention_params=None):

        if not default_net().plugin_config.gpt_attention_plugin:
            raise ValueError(
                'ChatGLM is only supported with GPTAttention plugin')

        assert isinstance(hidden_states, Tensor)
        qkv = self.qkv(hidden_states)

        # attention

        qkv = qkv.view(
            concat([
                shape(qkv, 0),
                shape(qkv, 1), self.num_attention_heads, 3,
                self.attention_head_size
            ]))
        query, key, value = split(qkv, 1, dim=3)
        query = query.view(
            concat([
                shape(qkv, 0),
                shape(qkv, 1), self.num_attention_heads,
                self.attention_head_size
            ]))
        key = key.view(
            concat([
                shape(qkv, 0),
                shape(qkv, 1), self.num_attention_heads,
                self.attention_head_size
            ]))
        value = value.view(
            concat([
                shape(qkv, 0),
                shape(qkv, 1), self.num_attention_heads,
                self.attention_head_size
            ]))
        zero = constant(
            np.ascontiguousarray(
                np.zeros([1, 1, 1, 1],

