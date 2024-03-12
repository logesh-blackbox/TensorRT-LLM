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

from ..functional import cast  # Import the cast function from the functional module


class Cast(Module):
    """
    This class represents a Cast module that casts the input tensor to a specified data type.

    The Cast module is a simple module that takes an input tensor and casts it to a specified data type.
    It is often used for data type conversion before or after certain operations.

    Args:
        output_dtype (str, optional): The desired data type for the output tensor.
                                       Defaults to 'float32'.
                                       Supported data types include 'float32', 'float16', 'bfloat16', 'bool',
                                       'int32', and 'int8'.

    Attributes:
        output_dtype (str): The data type for the output tensor.

    Methods:
        forward(x): Performs the cast operation on the input tensor 'x'.

    """

    def __init__(self, output_dtype: str = 'float32') -> None:
        super().__init__()
        assert output_dtype in ('float32', 'float16', 'bfloat16', 'bool',
                                'int32', 'int8'), TypeError(
                                    "%s is not supported" % output_dtype)
        self.output_dtype = output
