import tensorrt as trt
from typing import Optional, Tuple

from ..._common import default_net
from ..._utils import pad_vocab_size, str_dtype_to_trt
from ...functional import Tensor, gather_last_token_logits
from ...layers import (Attention, AttentionMaskType, AttentionParams,
                       ColumnLinear, Embedding, GatedMLP, KeyValueCacheParams,
                       RmsNorm)
from ...mapping import Mapping
from ...module import Module, ModuleList
from ..generation_mixin import GenerationMixin


class BaichuanDecoderLayer(Module):
    def __init__(self,
                 hidden_size: int,
                 num_attention_heads: int,
                 max_position_embeddings: int,
                 position_embedding_type: Optional[str],
                 dtype: Optional[trt.DataType] = None,
                 hidden_act: str = 'silu',
                 mlp_hidden_size: Optional[int] = None,
                 tp_group: Optional[str] = None,
                 tp_size: int = 1,
                 tp_rank: int = 0):
        super().__init__()
        self.input_layernorm = RmsNorm(normalized_shape=hidden_size,
                                       dtype=dtype)

        self.attention = Attention(
            hidden_size,
            num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            dtype=dtype,
            attention_mask_type=AttentionMaskType.causal,
            bias=False,
            position_embedding_type=position_embedding_type,
            tp_group=tp_group,
            tp_size=tp_size,
            tp_rank=tp_rank)
        if not mlp_hidden_size:
            mlp_hidden_size = hidden_size * 4
        self.mlp = GatedMLP(hidden_size=hidden_size,
                            ffn_hidden_size=mlp_hidden_size,
                            hidden_act=hidden_act,
                            dtype=dtype,
                            bias=False,
                            tp_group=tp_group,
                            tp_size=tp_size)
        self.post_layernorm = RmsNorm(normalized_shape=hidden_size, dtype=dtype)

    def forward(self,
                hidden_states: Tensor,
                attention_mask: Optional[Tensor] = None,
                use_cache: bool = False,
                kv_cache_params: Optional[KeyValueCacheParams] = None,
                attention_params: Optional[AttentionParams] = None) -> Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attention_output = self.attention(hidden_states,
                                          attention_mask=attention_mask,
                                          use_cache=use_cache,
                                          kv_cache_params=kv_cache_params,
                                          attention_params=attention_params)

        if use_cache:
            attention_output, presents = attention_output

        hidden_states = residual + attention_output

        residual = hidden_states
        hidden_states = self.post_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states
        if use_cache:
            return (hidden_states, presents)
        return hidden_states


class BaichuanModel(Module):
    def __init__(self,
                 num_layers: int,
                 num_heads: int,
                 hidden_size: int,
                 vocab_size: int,
                 hidden_act: str,
                 max_position_embeddings: int,
                 position_embedding_type: Optional[str],
                 dtype: Optional[trt.DataType],
                 mlp_hidden_size: Optional[int] = None,
                 mapping: Mapping = Mapping()):
        super().__init__()
        self.num_layers = num_layers
        self.vocab_embedding = Embedding(vocab_size, hidden_size, dtype=dtype)

        self.layers = ModuleList([
            BaichuanDecoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_heads,
                max_position_embeddings=max_position_embeddings,
                position_embedding_type=position_embedding_type,
                dtype=dtype,
                hidden_act=hidden_act,
                mlp_hidden_size=mlp_hidden_size,
                tp_group=mapping.tp_group,
                tp_size=mapping.tp_size,
                tp_rank=mapping.tp_rank) for _ in range(num_layers)
        ])

        self.ln_f = RmsNorm(normalized_shape=hidden_size, dtype=dtype)

    def forward(self,
                input_ids: Tensor,
                position_ids: Optional[Tensor] = None,
                use_cache: bool = False,
                attention_mask: Optional[Tensor] = None,
                kv_cache_params: Optional[KeyValueCacheParams] = None,
                attention_params: Optional[AttentionParams] = None) -> Tensor:

        hidden_states = self.vocab_embedding(input_ids)

        if kv_cache_params.past_key_value is None:
            kv_cache_params.past_key_value = tuple([None] * len(self.layers))

        if use_cache:
            presents = []

        for layer, past, pointer in zip(
                self.layers, kv_cache_params.past_key_value,
                kv_cache_params.kv_cache_block_pointers):
            hidden_states = layer(
                hidden_states,
                use_cache=use_cache,
                attention_mask=attention_mask,
                kv_
