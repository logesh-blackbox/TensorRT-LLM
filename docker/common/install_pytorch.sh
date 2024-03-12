#!/bin/bash

# This script installs PyTorch with a specific version and CXX11 ABI setting.

set -ex

# Variables
TORCH_VERSION="2.1.0"
SYSTEM_ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')

# Functions
prepare_environment() {
    # Prepare the environment based on the system type.
    if [[ $SYSTEM_ID == *"ubuntu"* ]]; then
      apt-get update && apt-get -y install ninja-build
      apt-get clean && rm -rf /var/lib/apt/lists/*
    elif [[ $SYSTEM_ID == *"centos"* ]]; then
      yum -y update && yum install -y ninja-build && yum clean all
      if [[ "$1" -eq "1" ]]; then
          # Temporarily disable devtoolset
          mv /tmp/devtoolset_env /tmp/devtoolset_env.bak
          touch /tmp/devtoolset_env
      fi
    else
      echo "This system type cannot be supported..."
      exit 1
    fi
}

restore_environment() {
    # Restore the environment based on the system type.
    if [[ $SYSTEM_ID == *"centos"* ]] && [[ "$1" -eq "1" ]]; then
        # Re-enable devtoolset
        rm -f /tmp/devtoolset_env
        mv /tmp/devtoolset_env.bak /tmp/devtoolset_env
    fi
}

install_from_source() {
    # Install PyTorch from source with a specific CXX11 ABI setting.
    prepare_environment $1
    export _GLIBCXX_USE_CXX11_ABI=$1
    export TORCH_CUDA_ARCH_LIST="8.0;9.0"

    # Uninstall any existing PyTorch installation.
    pip uninstall -y torch

    # Clone, build, and install PyTorch from source.
    cd /tmp
    git clone --depth 1 --branch v$TORCH_VERSION https://github.com/pytorch/pytorch
    cd pytorch
    git submodule sync && git submodule update --init --recursive
    pip install -r requirements.txt
    python setup.py install
    cd /tmp && rm -rf /tmp/pytorch
    restore_environment $1
}

install_from_pypi() {
    # Install PyTorch from PyPI.
    pip install torch==${TORCH_VERSION}
}

# Main
case "$1" in
  "skip")
    ;;
  "pyp
