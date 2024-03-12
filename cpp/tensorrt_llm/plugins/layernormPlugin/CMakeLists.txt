#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# This script is responsible for gathering all .cpp files in the current directory
# and adding them to the PLUGIN_SOURCES list.
#
file(GLOB SRCS *.cpp)
# Add the gathered .cpp files to the PLUGIN_SOURCES list.
set(PLUGIN_SOURCES ${PLUGIN_SOURCES} ${SRCS})
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
# This function initializes the plugin by setting up required variables and
# directories.
#
function(initialize_plugin)
    # Set the plugin version
    set(PLUGIN_VERSION "1.0.0" CACHE STRING "Plugin version")
    # Set the plugin name
    set(PLUGIN_NAME "ExamplePlugin" CACHE STRING "Plugin name")
   
