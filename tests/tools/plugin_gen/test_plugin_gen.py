# Import necessary modules
import os
import pytest
from .kernel_config import get_fmha_kernel_meta_data

# Get the kernel metadata
KERNEL_META_DATA = get_fmha_kernel_meta_data()

# Import the TRT plugin generation functions
from tensorrt_llm.tools.plugin_gen.plugin_gen import TRITON_COMPILE_BIN, gen_trt_plugins

# Define a workspace directory
WORKSPACE = './tmp/'

# Function to check if Triton is installed
def is_triton_installed() -> bool:
    return os.path.exists(TRITON_COMPILE_BIN)

# Skip the test if Triton is not installed
@pytest.mark.skipif(not is_triton_installed(), reason='triton is not installed')
def test_end_to_end():
    # Generate TRT plugins
    gen_trt_plugins(workspace=WORKSPACE, metas=[KERNEL_META_DATA])
