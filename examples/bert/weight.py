import numpy as np
import torch

def extract_layer_idx(name):
    ss = name.split('.')
    for s in ss:
        if s.isdigit():
            return int(s)
    return None

def split(v, tp_size, idx, dim=0):
    if tp_size == 1:
        return v
    if len(v.shape) == 1:
        return np.ascontiguousarray(np.split(v, tp_size)[idx])
    elif len(v.shape) == 2:
        return np.ascontiguousarray(np.split(v, tp_size, axis=dim)[idx])
    return None

def load_from_hf_bert(tensorrt_llm_bert,
                      hf_bert,
                      hf_bert_config,
                      rank=0,
                      tensor_parallel=1,
                      fp16=False):
    """
    Loads the weights from a Hugging Face BERT model to a TensorRT LLM BERT model.

    Args:
    tensorrt_llm_bert (TensorRTLLM): The TensorRT LLM BERT model to load the weights into.
    hf_bert (HuggingFaceBERT): The Hugging Face BERT model to load the weights from.
    hf_bert_config (HuggingFaceBERTConfig): The configuration of the Hugging Face BERT model.
    rank (int, optional): The rank of the current process. Defaults to 0.
    tensor_parallel (int, optional): The number of tensor parallel processes. Defaults to 1.
    fp16 (bool, optional): Whether to use float16 precision. Defaults to False.
    """
    qkv_weight = [[None, None, None]
                  for _ in range(hf_bert_config.num_hidden_layers)]

    qkv_bias = [[None, None, None]
                for _ in range(hf_bert_config.num_hidden_layers)]

    for k, v in hf_bert.state_dict().items():
        torch_dtype = torch.float16 if fp16 else torch.float32
        v = v.to(torch_dtype).cpu().numpy()
        if 'embeddings.word_embeddings.weight' in k:
            tensorrt_llm_bert.embedding.vocab_embedding.weight.value = v
        elif 'embeddings.position_embeddings.weight' in k:
            tensorrt_llm_bert.embedding.position_embedding.weight.value = v
        elif 'embeddings.token_type_embeddings.weight' in k:
            tensorrt_llm_bert.embedding.token_embedding.weight.value = v
        elif 'embeddings.LayerNorm.weight' in k:
            tensorrt_llm_bert.embedding.embedding_ln.weight.value = v
        elif 'embeddings.LayerNorm.bias' in k:
            tensorrt_llm_bert.embedding.embedding_ln.bias.value = v
        else:
            layer_idx = extract_layer_idx(k)
            if layer_idx is None:
                continue
            if 'attention.output.dense.weight' in k:
                tensorrt_llm_bert.layers[
                    layer_idx].attention.dense.weight.value = split(v,
                                                                  tensor_parallel,
                                                                  rank,
                                                                  dim=1)
            elif 'attention.output.dense.bias' in k:
                tensorrt_llm_bert.layers[layer_idx].attention.dense.bias.value = v
            elif 'attention.output.LayerNorm.weight' in k:
                tensorrt_llm_bert.layers[layer_idx].input_layernorm.weight.value = v
            elif 'attention.output.LayerNorm.bias' in k:
                tensorrt_llm_bert.layers[layer_idx].input_layernorm.bias.value = v
            elif 'intermediate.dense.weight' in k:
                tensorrt_llm_bert.layers[layer_idx].mlp.fc.weight.value = split(
                    v, tensor_parallel, rank)
            elif 'intermediate.dense.bias' in k:
                tensorrt_llm_bert.layers[layer_idx].mlp.fc.bias.value = split(
                    v, tensor_parallel, rank)
            elif 'output.dense.weight' in k:
                tensorrt_llm_bert.layers[layer_idx].mlp.proj.weight.value = split(
                    v, tensor_parallel, rank, dim=1)
            elif 'output.dense.bias' in k:
                tensorrt_llm_bert.layers[layer_idx].mlp.proj.bias.value = v
            elif 'output.LayerNorm.weight' in k:
                tensorrt_llm_bert.layers[layer_idx].post_layernorm.weight.value = v
            elif 'output.LayerNorm.bias' in k:
                tensorrt_llm_bert.layers[layer_idx].post_layernorm.bias.value = v
            elif 'attention.self.query.weight' in k:
                qkv_weight[layer_idx][0] = v
            elif 'attention.self.query.bias' in k:
                qkv_bias[layer_idx][0] = v
            elif 'attention.self.key.weight' in k:
                qkv_weight[layer_idx][1] = v
            elif 'attention.self.key.bias' in k:
                qkv_bias[layer_idx][1] = v
            elif 'attention.self.value.weight' in k:
                qkv_weight[layer_idx][2] = v
            elif 'attention.self.value.bias' in k:
                qkv_bias[layer_idx][2] = v
            else:
                continue

    for layer_idx in range
