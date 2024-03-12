# This script is used to generate kernel metadata for TensorRT.

import os
from tensorrt_llm.tools.plugin_gen.core import *

# The root directory of the OpenAI Triton manual plugin examples.
openai_triton_example_root = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "examples",
    "openai_triton", "manual_plugin")

# The function `get_fmha_kernel_meta_data` returns the metadata of the 'fused_attention_kernel'.
def get_fmha_kernel_meta_data():
    return KernelMetaData(
        kernel_name='fused_attention_kernel',  # The name of the kernel.
        ios=[
            # Output arguments.
            OutputArg('Out', Type('tensor[fp16]'), hints=['16', '16']),
            OutputArg('L', Type('tensor[fp32]'), hints=['16', '16']),
            OutputArg('M', Type('tensor[fp16]'), hints=['16', '16']),
            # Input arguments.
            InputArg('Q', Type('tensor[fp16]'), hints=['16', '16']),
            InputArg('K', Type('tensor[fp16]'), hints=['16', '16']),
            InputArg('V', Type('tensor[fp16]'), hints=['16', '16']),
            ParamArg('sm_scale', Type('fp32')),
            DimSizeArg('batch_size'),
            ParamArg('num_heads', Type('i32')),
            DimSizeArg('seq_len', hints=['', '16']),
            # Constexpr arguments.
            Constexpr(128),
            Constexpr(64),
            Constexpr(128),
        ],
        shape_infer_rules=[
            # The following rules help to deduce the shapes of the output tensors.
            "Q[*] -> Out[*]",
            "Q[m,n,k,*] -> L[m,n,k]",
            "Q[m,n,k,*] -> M[m,n,k]",

            # The following rules help to deduce both DimSizeArgs: batch_size and seq_len.
            "Q[m,n,k,*] : m -> batch_size",
            "Q[m,n,k,*] : k -> seq_len",
        ],
        version=0,  # The version of the kernel.
        kernel_file=f'{openai_triton_example_root}/fmha_triton.py',  # The path to the kernel file.
        num_warps=1,  # The number of warps.
        grid
