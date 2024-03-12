import time

import numpy as np
import torch

import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_torch, torch_to_numpy
from tensorrt_llm.quantization import QuantMode


def extract_layer_idx(name):
    """
    Extract the layer index from the name of a parameter.

    Args:
        name (str): The name of the parameter.

    Returns:
        int: The layer index, or None if the name does not contain a layer index.
    """
    for s in name.split('.'):
        if s.isdigit():
            return int(s)
    return None


def split(v, tp_size, idx, dim=0):
    """
    Split a numpy array into multiple parts along a given dimension.

    Args:
        v (numpy.ndarray): The numpy array to split.
        tp_size (int): The number of parts to split the array into.
        idx (int): The index of the part to return.
        dim (int, optional): The dimension along which to split the array. Defaults to 0.

    Returns:
        numpy.ndarray: The specified part of the split array.
    """
    if v.size == 1:
        return v
    if v.ndim == 1:
        return np.split(v, tp_size)[idx]
    else:
        return np.split(v, tp_size, axis=dim)[idx]


def load_from_hf_baichuan(tensorrt_llm_baichuan,
                          hf_baichuan,
                          model_version,
                          rank=0,
                          tensor_parallel=1,
                          dtype="float32"):
    """
    Load the weights from a Hugging Face Baichuan model to a TensorRT LLM Baichuan model.

    Args:
        tensorrt_llm_baichuan (tensorrt_llm.TensorRTLLMBaichuan): The TensorRT LLM Baichuan model to load the weights into.
        hf_baichuan (torch.nn.Module): The Hugging Face Baichuan model to load the weights from.
        model_version (str): The version of the Hugging Face Baichuan model.
        rank (int, optional): The rank of the current process. Defaults to 0.
        tensor_parallel (int, optional): The number of tensor parallel processes. Defaults to 1.
        dtype (str, optional): The data type of the weights. Defaults to "float32".

    Returns:
        None
    """
    assert model_version is not None
    tensorrt_llm.logger.info(
        f'Loading weights from HF Baichuan {model_version}...')
    tik = time.time()

    quant_mode = getattr(tensorrt_llm_baichuan, 'quant_mode', QuantMode(0))
    if quant_mode.is_int8_weight_only():
        plugin_weight_only_quant_type = torch.int8
    elif quant_mode.is_int4_weight_only():
        plugin_weight_only_quant_type = torch.quint4x2
    use_weight_only = quant_mode.is_weight_only()

    model_params = dict(hf_baichuan.named_parameters())
    for k, v in model_params.items():
        torch_dtype = str_dtype_to_torch(dtype)
        v = torch_to_numpy(v.to(torch_dtype).detach().cpu())
        if 'model.embed_tokens.weight' in k:
            tensorrt_llm_baichuan.vocab_embedding.weight.value = v
        elif 'model.norm.weight' in k:
            tensorrt_llm_baichuan.ln_f.weight.value = v
        elif 'lm_head.weight' in k:
            if model_version.startswith('v2'):
                # baichuan v2 models use NormHead
                tensorrt_llm.logger.info(
                    f'Normalizing lm_head.weight for {model_version}')
                original_v = model_params[k]
                v = torch_to_numpy(
                    torch.nn.functional.normalize(original_v).to(
                        torch_dtype).detach().cpu())
            tensorrt_llm_baichuan.lm_head.weight.value = np.ascontiguousarray(
                split(v, tensor_parallel, rank))
        else:
            layer_idx = extract_layer_idx(k)
            if layer_idx is None:
                continue
            if layer_idx >= tensorrt_llm_baichuan._num_layers:
                continue
            if 'input_layernorm.weight' in k:
                tensorrt_llm_baichuan.layers[
                    layer_idx].input_layernorm.weight.value = v
            elif 'post_attention_layernorm.weight' in k:
                dst = tensorrt_llm_baichuan.layers[layer_idx].post_layernorm.weight
                dst.value = v
            elif 'self_attn.W_pack.weight' in k:
                dst = tensorrt_llm_baichuan.layers[layer_idx].attention.qkv.weight
                q_emb = v.shape[0] // 3
                model_emb = v.shape[1]
                v = v.reshape(3, q_emb, model_emb)
                split_v = [
                    split(v[i], tensor_parallel, rank, dim=1)
                    for i in range(3)
                ]
                split_v = [
                    split_v[i].reshape(q_emb // tensor_parallel, model_emb)
                    for i in range(3)
                ]
                if use_weight_only:
                    v = np.stack(
                        [np.transpose(split_v[i], (1,
