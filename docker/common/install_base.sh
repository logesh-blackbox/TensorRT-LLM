#!/bin/bash

# This script installs necessary packages and dependencies for running the
# TensorRT inference engine on Ubuntu or CentOS systems.

# It first updates the package lists and installs required packages such as wget,
# gdb, git-lfs, python3-pip, python3-dev, libffi-dev, and OpenMPI.
# It then removes any previous TensorRT installations and installs mpi4py.
# The script also sets up the necessary environment variables for OpenMPI.

# The script checks the base OS and installs the appropriate packages and
# dependencies.
# For Ubuntu, it installs the required packages using apt-get.
# For CentOS, it installs the required packages using yum and sets up the
# necessary environment variables for GCC and OpenMPI.

# The script also installs TensorRT using pip.

# Usage:
#   source install_deps.sh

set -ex

# Initialize Ubuntu
init_ubuntu() {
    apt-get update
    apt-get install -y --no-install-recommends wget gdb git-lfs python3-pip python3-dev python-is-python3 libffi-dev
    if ! command -v mpirun &> /dev/null; then
      DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends openmpi-bin libopenmpi-dev
    fi
    apt-get clean
    rm -rf /var/lib/apt/lists/*
    # Remove previous TRT installation
    if [[ $(apt list --installed | grep libnvinfer) ]]; then
        apt-get remove --purge -y libnvinfer*
    fi
    if [[ $(apt list --installed | grep tensorrt) ]]; then
        apt-get remove --purge -y tensorrt*
    fi
    pip uninstall -y tensorrt
    pip install mpi4py
}

# Install GCC on CentOS
install_gcc_centos() {
    yum -y update
    GCC_VERSION="8.5.0"
    yum install -y gcc gcc-c++ file libtool make wget bzip2 bison yacc flex
    wget https://github.com/gcc-mirror/gcc/archive/refs/tags/releases/gcc-${GCC_VERSION}.tar.gz -O /tmp/gcc-${GCC_VERSION}.tar.gz
    tar -xf /tmp/gcc-${GCC_VERSION}.tar.gz -C /tmp/ && cd /tmp/gcc-releases-gcc-${GCC_VERSION}
    ./contrib/download_prerequisites
    ./configure --disable-multilib --enable-languages=c,c++ --with-pic
    make -j$(nproc) && make install
    echo "export LD_LIBRARY_PATH=/usr/local/lib64:\$LD_LIBRARY_PATH" >> "${BASH_ENV}"
    cd .. && rm -rf /tmp/gcc-*
    yum clean all
}

# Initialize CentOS
init_centos() {
    PY_VERSION=38
    DEVTOOLSET_ENV_FILE="/tmp/devtoolset_env"
    yum -y update
    yum -y install centos-release-scl-rh epel-release
    # https://gitlab.com/nvidia/container-images/cuda
    CUDA_VERSION=$(nvcc --version | sed -n 's/^.*release \([0-9]\+\.[0-9]\+\).*$/\1/p')
    YUM_CUDA=${CUDA_VERSION/./-}
    # Consistent with manylinux2014 centos-7 based version
    yum -y install wget rh-python${PY_VERSION} rh-python${PY_VERSION}-python-devel rh-git227 devtoolset-10 libffi-devel
    yum -y install openmpi3 openmpi3-devel
    echo "source scl_source enable rh-git2
