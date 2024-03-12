#!/bin/bash

# This script automates the process of installing the Polygraphy package version 0.48.1 for TensorRT 9.0.1.

# Step 1: Ensure the script runs with the expected behavior by setting the -ex flags for the set built-in command.
set -ex

# Step 2: Define the URL for the TensorRT 9.0.1 Polygraphy package.
RELEASE_URL_PG="https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/9.0.1/tars/polygraphy-0.48.1-py2.py3-none-any.whl"

# Step 3: Uninstall any currently installed version of the Polygraphy package.
pip uninstall -y polygraphy

# Step 4: Install the specified Polygraphy package version using pip.
pip install "${RELEASE_URL_PG}"

