#!/bin/bash

# This script installs a specific version of CMake on a Linux system.

set -ex  # Exit on error and print commands.

# Get the architecture of the system.
ARCH=$(uname -m)

# Define the desired CMake version.
CMAKE_VERSION="3.24.4"

# Extract the major and minor version from the CMAKE_VERSION variable.
PARSED_CMAKE_VERSION=$(echo $CMAKE_VERSION | sed 's/\.[0-9]*$//')

# Set the CMake file name based on the architecture and version.
CMAKE_FILE_NAME="cmake-${CMAKE_VERSION}-linux-${ARCH}"

# Define the URL for the CMake release archive.
RELEASE_URL_CMAKE=https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/${CMAKE_FILE_NAME}.tar.gz

# Download the CMake release archive to the /tmp directory.
wget --no-verbose ${RELEASE_URL_CMAKE} -P /tmp

# Extract the CMake archive to /usr/local/.
tar -xf /tmp/${CMAKE_FILE_NAME}.tar.gz -C /usr/local/

# Create a symbolic link to the CMake installation directory.
ln -s /usr/local/${CMAKE_FILE_NAME} /usr/local/cmake

# Add the CMake bin directory to the PATH environment variable.
echo 'export PATH=/usr/local/cmake/bin:$PATH' >> "${BASH_ENV}"
