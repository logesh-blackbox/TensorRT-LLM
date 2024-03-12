import argparse
import configparser
import dataclasses
import os
from pathlib import Path

import torch
import torch.multiprocessing as multiprocessing
from convert import split_and_save_weight
from smoothquant import capture_activation_range, smooth_gemm
from tqdm import tqdm
from transformers import BloomForCausalLM, BloomTokenizerFast
from transformers.models.bloom.modeling_bloom import BloomBlock

from tensorrt_llm._utils import str_dtype_to_torch, torch_to_numpy


@dataclasses.dataclass(frozen=True)
class ProgArgs:
    out_dir: str
    in_file: str
    tensor_parallelism: int = 1
    processes: int = 4
    calibrate_kv_cache: bool = False
    smoothquant: float = None
    model: str = "bloom"
    storage_type: str = "fp32"
    dataset_cache_dir: str = None
    load_model_on_cpu: bool = False
    convert_model_on_cpu: bool = False

    @staticmethod
    def parse(args=None) -> 'ProgArgs':
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('--out-dir',
                            '-o',
                            type=str,
                            help='file name of output directory',
                            required=True)
        parser.add_argument('--in-file',
                            '-i',
                            type=str,
                            help='file name of input checkpoint file',
                            required=True)
        parser.add_argument('--tensor-parallelism',
                            '-tp',
                            type=int,
                            help='Requested tensor parallelism for inference',
                            default=1)
        parser.add_argument(
            "--processes",
            "-p",
            type=int,
            help=
            "How many processes to spawn for conversion (default: 4). Set it to a lower value to reduce RAM usage.",
            default=4)
        parser.add_argument(
            "--calibrate-kv-cache",
            "-kv",
            action="store_true",
            help=
            "Generate scaling factors for KV cache. Used for storing KV cache in int8."
        )
        parser.add_argument(
            "--smoothquant",
            "-sq",
            type=float,
            default=None,
            help="Set the α parameter (see https://arxiv.org/pdf/2211.10438.pdf)"
            " to Smoothquant the model, and output int8 weights."
            " A good first try is 0.5. Must be in [0, 1]")
        parser.add_argument(
            "--model",
            default="bloom",
            type=str,
            help="Specify Bloom variants to convert checkpoints correctly",
            choices=["bloom"])
        parser.add_argument("--storage-type",
                            "-t",
                            type=str,
                            default="float32",
                            choices=["float32", "float16", "bfloat16"])
        parser.add_argument("--dataset-cache-dir",
                            type=str,
                            default=None,
                            help="cache dir to load the hugging face dataset")
        parser.add_argument("--load-model-on-cpu", action="store_true")
        parser.add_argument("--convert-model-on-cpu", action="store_true")
        return ProgArgs(**vars(parser.parse_args(args)))


def reorder_torch_qkv_weight_or_bias(v, model, is_bias=False):
    """ Reorder the qkv weight.

    Note that the shape of the fused QKV weights in HF is different from the
    shape that TRT-LLM requires.
       HF: (num_heads x 3 x head_dim, hidden_size)
       TRT-LLM: (3 x num_heads x head_dim, hidden_size)
    This is unlike to the other models in HF e.g. GPT where they have the
    same shape with TRT-LLM, i.e., (3 x num_heads x head_dim, hidden_size). We reshape the qkv
        weight: (3 x num_heads x head_dim, hidden).
        bias  : (3 x num_heads x head_dim).
    """

    n_head = model.transformer.num_heads
    hidden_size = model.transformer.embed_dim
    head_dim = hidden_size // n_head

    # (3 x hidden, ...) view as (num_heads, 3, head_dim, ...)
    v = v.reshape(n_head, 3, head_dim, -1)
    # permute to (3, num_heads, head_dim, ...)
    v = v.permute((1, 0, 2, 3))
    # final shape: weight=(3 x hidden, hidden) or bias=(3 x hidden)
    if is_bias:
        return v.reshape(3 * hidden_size)
    return v.reshape(3 * hidden_size, hidden_size)


@torch.no_grad()
def smooth_bloom_model(model, scales, alpha, bloom_qkv_param, bloom_smoother):
    # Smooth the activation and weights with smoother = $\diag{s}$
    for name, module in model.named_modules():
        if not isinstance(module, BloomBlock):
            continue

        # reorder qkv weight/bias and scales
        param = module.self_attention.query_key_value.weight
        param = reorder_torch_qkv_weight_or_bias(param, model, is_bias=False)

        layer_name = name + ".self_attention.query_key_value.weight"
       
