'''
This file is a script tool for generating TensorRT plugin library for Triton.
'''
import argparse
import glob
#import pkg_resources
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import ClassVar, Iterable, List, Tuple, Union

try:
    import triton
except ImportError:
    raise ImportError("Triton is not installed. Please install it first.")

from tensorrt_llm.tools.plugin_gen.core import (KernelMetaData,
                                                PluginCmakeCodegen,
                                                PluginCppCodegen,
                                                PluginPyCodegen,
                                                PluginRegistryCodegen)

PYTHON_BIN = sys.executable

TRITON_ROOT = os.path.dirname(triton.__file__)
TRITON_COMPILE_BIN = os.path.join(TRITON_ROOT, 'tools', 'compile.py')
TRITON_LINK_BIN = os.path.join(TRITON_ROOT, 'tools', 'link.py')


@dataclass
class StageArgs:
    workspace: str  # the root directory for all the stages
    kernel_name: str

    @property
    def sub_workspace(self) -> str:
        return os.path.join(self.workspace, self.kernel_name,
                            f"_{self.stage_name}")


@dataclass
class _TritonAotArgs(StageArgs):
    stage_name: ClassVar[str] = 'triton_aot'

    @dataclass
    class _AotConfig:
        output_name: str
        num_warps: int
        num_stages: int
        signature: str

    kernel_file: str
    configs: List[_AotConfig]
    grid_dims: Tuple[str, str, str]


@dataclass
class _TritonKernelCompileArgs(StageArgs):
    stage_name: ClassVar[str] = 'compile_triton_kernel'


@dataclass
class _TrtPluginGenArgs(StageArgs):
    stage_name: ClassVar[str] = 'generate_trt_plugin'

    config: KernelMetaData


@dataclass
class _TrtPluginCompileArgs:
    workspace: str

    stage_name: ClassVar[str] = 'compile_trt_plugin'

    @property
    def sub_workspace(self) -> str:
        return self.workspace


@dataclass
class _CopyOutputArgs:
    so_path: str
    functional_py_path: str
    output_dir: str

    stage_name: ClassVar[str] = 'copy_output'


@dataclass
class Stage:
    '''
    Stage represents a stage in the plugin generation process. e.g. Triton AOT could be a stage.
    '''

    config: Union[_TritonAotArgs, _TritonKernelCompileArgs, _TrtPluginGenArgs,
                  _TrtPluginCompileArgs, _CopyOutputArgs]

    def run(self):
        stages = {
            _TritonAotArgs.stage_name: self.do_triton_aot,
            _TritonKernelCompileArgs.stage_name: self.do_compile_triton_kernel,
            _TrtPluginGenArgs.stage_name: self.do_generate_trt_plugin,
            _TrtPluginCompileArgs.stage_name: self.do_compile_trt_plugin,
            _CopyOutputArgs.stage_name: self.do_copy_output,
        }

        logging.info(f"Running stage {self.config.stage_name}")

        stages[self.config.stage_name]()

    def do_triton_aot(self):
        compile_dir = self.config.sub_workspace
        _clean_path(compile_dir)

        # compile each config for different hints
        for config in self.config.configs:
            command = [
                PYTHON_BIN, TRITON_COMPILE_BIN, self.config.kernel_file, '-n',
                self.config.kernel_name, '-o', f"{compile_dir}/kernel",
                '--out-name', config.output_name, '-w',
                str(config.num_warps), '-s', f"{config.signature}", '-g',
                ','.join(self.config.grid_dims), '--num-stages',
                str(config.num_stages)
            ]
            _run_command(command)

        # link and get a kernel launcher with all the configs
        h_files = glob.glob(os.path.join(compile_dir, '*.h'))
        command = [
            PYTHON_BIN,
            TRITON_LINK_BIN,
            *h_files,
            '-o',
            os.path.join(compile_dir, 'launcher'),
        ]
        _run_command(command)

    def do_compile_triton_kernel(self):
        '''
        Compile the triton kernel to library.
        '''
        #assert isinstance(self.args, _TritonKernelCompileArgs)

        from triton.common import cuda_include_dir, libcuda_dirs
        kernel_dir = os.path.join(self.config.workspace,
                                  self.config.kernel_name, '_triton_aot')
        compile_dir = self.config.sub_workspace
        _mkdir(compile_dir)
        _clean_path(compile_dir)

        c_files = glob.glob(os.path.join(os.getcwd(), kernel_dir, "*.c"))
        assert c_files
        _run_command([
            "gcc",
            "-c",
            *c_files,
            "-I",
            cuda_include_dir(),
            "-L",
            libcuda_dirs()[0],
           
