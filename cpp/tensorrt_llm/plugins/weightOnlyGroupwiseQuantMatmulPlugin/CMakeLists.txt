#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# This script is responsible for gathering all .cpp files in the current directory
# and adding them to the PLUGIN_SOURCES list.
#
file(GLOB SRCS *.cpp)
# Add the gathered .cpp files to the PLUGIN_SOURCES list.
set(PLUGIN_SOURCES ${SRCS})
# Propagate the updated PLUGIN_SOURCES list to the parent scope.
set(PLUGIN_SOURCES
    ${PLUGIN_SOURCES}
    PARENT_SCOPE)



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
# This function sets up the necessary flags and compiler options for building
# the plugin.
function(setup_plugin_build)
    # Set the minimum required CMake version.
    cmake_minimum_required(VERSION 3.14)

    # Set the project name and version.
    project(MyPlugin VERSION 1.0.0)

    # Add the source files to the project.
    add_library(MyPlugin ${PLUGIN_SOURCES})

    # Set the plugin type to SHARED.
    set_target_properties(MyPlugin PROPERTIES
        PLUGIN_TYPE SHARED
    )

    # Set the plugin interface.
    set_target_properties(MyPlugin PROPERTIES
        PLUGIN_INTERFACE "MyPlugin"
    )

    # Set the plugin version.
    set_target_properties(MyPlugin PROPERTIES

