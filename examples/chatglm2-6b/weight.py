import time
import numpy as np
import torch
import transformers

import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.models import ChatGLM2HeadModel
from tensorrt_llm.quantization import QuantMode


def extract_layer_idx(name):
    for s in name.split('.'):
        if s.isdigit():
            return s
    return None


def split(v, tp_size, idx, dim=0):
    if tp_size == 1:
        return v
    if len(v.shape) == 1:
        return np.split(v, tp_size)[idx]
    else:
        return np.split(v, tp_size, axis=dim)[idx]


def load_from_hf_chatglm2_6B(tensorrt_llm_model,
                             hf_model,
                             rank=0,
                             tensor_parallel=1,
                             dtype="float32",
                             multi_query_mode=False):
    tensorrt_llm.logger.info('Loading weights from HF ChatGLM2...')
    time.time()

    quant_mode = getattr(tensorrt_llm_model, 'quant_mode', QuantMode(0))
    if quant_mode.is_int8_weight_only():
        plugin_weight_only_quant_type = torch.int8
    elif quant_mode.is_int4_weight_only():
        plugin_weight_only_quant_type = torch.quint4x2
    use_weight_only = quant_mode.is_weight_only()

    str_dtype_to_torch(dtype)
    tensorrt_llm_model.encoder.final_layernorm.weight.value = hf_model.transformer.encoder.final_layernorm.weight.detach().cpu().numpy()
    tensorrt_llm_model.embedding.weight.value = hf_model.transformer.embedding.word_embeddings.weight.detach().cpu().numpy()
    tensorrt_llm_model.lm_head.weight.value = hf_model.transformer.output_layer.weight.detach().cpu().numpy()

    def load_quant_weight(src, value_dst, scale_dst,
                          plugin_weight_only_quant_type):
        v = np.ascontiguousarray(src.transpose())
        processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
            torch.tensor(v), plugin_weight_only_quant_type)
        value_dst.value = processed_torch_weights.view(
            dtype=torch.float32).numpy()
        scale_dst.value = torch_weight_scales.numpy()

    for i in range(28):
        tensorrt_llm_model.encoder.layers[
            i].input_layernorm.weight.value = hf_model.transformer.encoder.layers[
                i].input_layernorm.weight.detach().cpu().numpy()
        tensorrt_llm_model.encoder.layers[
            i].post_attention_layernorm.weight.value = hf_model.transformer.encoder.layers[
                i].post_attention_layernorm.weight.detach().cpu().numpy()
        tensorrt_llm_model.encoder.layers[
            i].self_attention.qkv.bias.value = hf_model.transformer.encoder.layers[
                i].self_attention.query_key_value.bias.detach().cpu().numpy()
        h_to_4h_weight = hf_model.transformer.encoder.layers[
            i].mlp.dense_h_to_4h.weight.detach().cpu()
        h_to_4h_weight = torch.split(h_to_4h_weight,
                                     h_to_4h_weight.shape[0] // 2, 0)
        h_to_4h_weight = torch.concat(h_to_4h_weight[::-1], 0).numpy()
        if use_weight_only:

            load_quant_weight(
                src=h_to_4h_weight,
                value_dst=tensorrt_llm_model.encoder.layers[i].mlp.fc.weight,
                scale_dst=tensorrt_llm_model.encoder.layers[i].mlp.fc.
                per_channel_scale,
                plugin_weight_only_quant_type=plugin_weight_only_quant_type)
            load_quant_weight(
                src=hf_model.transformer.encoder.layers[i].mlp.dense_4h_to_h.
                weight.detach().cpu().numpy(),
                value_dst=tensorrt_llm_model.encoder.layers[i].mlp.proj.weight,
                scale_dst=tensorrt_llm_model.encoder.layers[i].mlp.proj.
                per_channel_scale,
                plugin_weight_only_quant_type=plugin_weight_only_quant_type)
            load_quant_weight(
                src=hf_model.
