#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#

# Add all cpp files to the source list
file(GLOB SRCS *.cpp)

# Add all cu files to the source list
file(GLOB CU_SRCS *.cu)

# Create a library from the source files
add_library(common_src OBJECT ${SRCS} ${CU_SRCS})

# Enable position independent code
set_property(TARGET common_src PROPERTY POSITION_INDEPENDENT_CODE ON)

# Enable CUDA resolve device symbols
set_property(TARGET common_src PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

