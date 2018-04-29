#!/bin/bash

# Create conda environment
echo "Creating conda environment"
conda create -n trafficv python=3.5 --yes

# Activate created conda environment
echo "Activating conda environment"
source activate trafficv

# Install necessary python packages
echo "Installing necessary python packages"
pip install -r ${TRAFFICVISION}/scripts/requirements.txt

# Deactivate conda environment
echo "Deactivating conda environment"
source deactivate trafficv