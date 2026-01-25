#!/bin/bash

# Time-FLM Training Script
# This script runs the Time-FLM training with configurations from config.yaml

echo "Starting Time-FLM training..."

# Set environment variables if needed
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the training script
python run_FLM.py

echo "Training completed."