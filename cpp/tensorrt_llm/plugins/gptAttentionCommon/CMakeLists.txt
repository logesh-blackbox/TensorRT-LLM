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
# Set the PLUGIN_SOURCES list in the parent scope to make it accessible
# in the parent directory.
set(PLUGIN_SOURCES
    ${PLUGIN_SOURCES}
    PARENT_SCOPE)

