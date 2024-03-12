# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import platform
import re
import sys
from typing import List

import setuptools
from setuptools.dist import Distribution


def extract_url(line: str) -> str:
    return next(filter(lambda x: x[0] != "-", line.split()))


def extract_requirements(requirements_filename: str) -> List[str]:
    with open(requirements_filename) as f:
        requirements = f.read().splitlines()

    extra_urls = []
    required_deps = []

    for line in requirements:
        if line[0] == "#":
            continue

        if "-i " in line or "--extra-index-url" in line:
            extra_urls.append(extract_url(line))
        else:
            required_deps.append(line)

    return required_deps, extra_urls


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return False

    def is_pure(self):
        return True


def get_platform_specific_libs() -> List[str]:
    if sys.platform == "win32":
        return ["libs/th_common.dll", "libs/nvinfer_plugin_tensorrt_llm.dll"]
    elif sys.platform.startswith("linux"):
        return ["libs/libth_common.so", "libs/libnvinfer_plugin_tensorrt_llm.so"]
    else:
        raise RuntimeError(f"Unsupported platform: {sys.platform}")


def get_package_data() -> dict:
    return {
        "tensorrt_llm": [
            *get_platform_specific_libs(),
            "tools/plugin_gen/templates/*"
        ]
    }


setup_kwargs = {
    "name": "tensorrt_llm",
    "version": "0.5.0",
    "description": "TensorRT-LLM: A TensorRT Toolbox for Large Language Models",
    "install_requires": [],
    "dependency_links": [],
    "zip_safe": True,
    "packages": setuptools.find_packages(),
    "package_data": get_package_data(),
    "python_requires": ">=3.7, <4",
    "distclass": BinaryDistribution,
}

required_deps, extra_urls = extract_requirements("requirements.txt")
setup_kwargs["install_requires"] = required_deps
setup_kwargs["dependency_links"] = extra_urls

if __name__ == "__main__":
    setuptools.setup(**setup_kwargs)

