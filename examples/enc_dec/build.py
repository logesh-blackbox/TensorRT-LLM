import argparse
import time
from pathlib import Path

import torch

import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_trt
from tensorrt_llm.builder import Builder
from tensorrt_llm.logger import logger
from tensorrt_llm.network import net_guard

from weight import load_t5_from_pytorch, parse_config  # isort:skip

# Define the name of the model
MODEL_NAME = "enc_dec"


def get_engine_name(model, dtype, tp_size, rank):
    return '{}_{}_tp{}_rank{}.engine'.format(model, dtype, tp_size, rank)


def serialize_engine(engine, path):
    logger.info(f'Serializing engine to {path}...')
    tik = time.time()
    with open(path, 'wb') as f:
        f.write(bytearray(engine))
    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Engine serialized. Total time: {t}')


def parse_arguments(args, component):
    parser = argparse.ArgumentParser()
    # Add arguments for the model directory, data type, logits data type,
    # timing cache, log level, vocabulary size, number of layers,
    # number of positions, embedding size, number of attention heads,
    # activation function, intermediate size, bias, maximum batch size,
    # maximum encoder input length, maximum input length, maximum output
    # length, maximum beam width, BERT attention plugin data type, GPT
    # attention plugin data type, GEMM plugin data type, layernorm plugin
    # data type, enable query-key half accumulation, GPUs per node,
    # builder options, output directory, remove input padding, random
    # seed, and use lookup plugin.
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--dtype',
                        type=str,
                        default='float16',
                        choices=['float16', 'float32', 'bfloat16'])
    parser.add_argument('--logits_dtype',
                        type=str,
                        default='float32',
                        choices=['float16', 'float32'])
    parser.add_argument('--timing_cache',
                        type=str,
                        default='model.cache',
                        help=
        'The path of to read timing cache from, will be ignored if the file does not exist')
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--vocab_size', type=int, default=32128)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--n_positions', type=int, default=1024)
    parser.add_argument('--n_embd', type=int, default=1024)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--hidden_act', type=str, default='gelu')
    parser.add_argument('--inter_size', type=int, default=None)
    parser.add_argument('--no_bias', action="store_false")
    parser.add_argument('--max_batch_size', type=int, default=8)
    parser.add_argument('--max_encoder_input_len', type=int, default=1024)
    parser.add_argument('--max_input_len', type=int, default=200)
    parser.add_argument('--max_output_len', type=int, default=200)
    parser.add_argument('--max_beam_width', type=int, default=1)
    parser.add_argument(
        '--use_bert_attention_plugin',
        nargs='?',
        const=None,
        type=str,
        default=False,
        choices=['float16', 'float32', 'bfloat16'],
        help=
        "Activates BERT attention plugin. You can specify the plugin dtype or leave blank to use the model dtype."
    )
    parser.add_argument(
        '--use_gpt_attention_plugin',
        nargs='?',
        const=None,
        type=str,
        default=False,
        choices=['float16', 'float32', 'bfloat16'],
        help=
        "Activates attention plugin. You can specify the plugin dtype or leave blank to use the model dtype."
    )
    parser.add_argument(
        '--use_gemm_plugin',
        nargs='?',
        const=None,
        type=str,
        default=False,
        choices=['float16', 'float32', 'bfloat16'],
        help=
        "Activates GEMM plugin. You can specify the plugin dtype or leave blank to use the model dtype."
    )
    parser.add_argument(
        '--use_layernorm_plugin',
        nargs='?',
        const=None,
        type=str,
        default=False,
        choices=['float16', 'float32', 'bfloat16'],
        help=
        "Activates layernorm plugin. You can specify the plugin dtype or leave blank to use the model dtype."
    )
    parser.add_argument('--enable_qk_half_accum',
                        default=False,
                        action='store_true')
    parser.add_argument('--gpus_per_node', type=int, default=8)
    parser.add_argument('--builder_opt', type=int, default=None)
    parser.add_argument(
        '--output_dir',
        type=Path,
        default='trt_engines',
        help=
        'The path to save the serialized engine files,
