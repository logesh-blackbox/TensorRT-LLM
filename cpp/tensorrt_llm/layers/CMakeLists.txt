// SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
//
// This file is licensed under the Apache License, Version 2.0. You may not use this file
// except in compliance with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.
//
// This CMake script is responsible for gathering all .cpp and .cu files and creating
// a library target named 'layers_src'. The library is built as a position-independent
// code (PIC) and CUDA device symbols are resolved during the linking stage.

file(GLOB SRCS *.cpp)    // Gathers all .cpp files in the current directory
file(GLOB CU_SRCS *.cu)  // Gathers all .cu files in the current directory

add_library(layers_src OBJECT ${SRCS} ${CU_SRCS})  // Creates a library target named 'layers_src'
set_property(TARGET layers_src PROPERTY POSITION_INDEPENDENT_CODE ON)  // Enables position-independent code
set_property(TARGET layers_src PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)  // Enables CUDA device symbol resolution



#
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain
