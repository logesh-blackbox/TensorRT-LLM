#!/bin/bash

# This script installs a specific version of CMake on a Linux system.

set -ex  # Exit on error and print commands.

# Get the architecture of the system.
architecture=$(uname -m)

# Define the desired CMake version.
desired_cmake_version="3.24.4"

# Extract the major and minor version from the CMAKE_VERSION variable.
parsed_cmake_version="${desired_cmake_version%.*}"

# Set the CMake file name based on the architecture and version.
cmake_file_name="cmake-${desired_cmake_version}-linux-${architecture}"

# Define the URL for the CMake release archive.
release_url_cmake=https://github.com/Kitware/CMake/releases/download/v${desired_cmake_version}/${cmake_file_name}.tar.gz

# Check if CMake is already installed and if the version matches.
if command -v cmake >/dev/null 2>&1 && cmake --version | grep -q "${parsed_cmake_version}"; then
    echo "CMake ${desired_cmake_version} is already installed."
    exit 0
fi

# Download the CMake release archive to the /tmp directory.
wget --no-verbose "${release_url_cmake}" -P /tmp

# Extract the CMake archive to /usr/local/.
tar -xf "/tmp/${cmake_file_name}.tar.gz" -C /usr/local/

# Create a symbolic link to the CMake installation directory.
ln -s "/usr/local/${cmake_file_name}" /usr/local/cmake

# Add the CMake bin directory to the PATH environment variable.
echo 'export PATH=/usr/local/cmake/bin:$PATH' >> "${BASH_
