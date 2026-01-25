#!/bin/bash

# Baseline Models Training Script
# This script runs the baseline models training with configurations from config_baseline.yaml

echo "Starting baseline models training..."

# Set environment variables if needed
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the baseline script
python run_baseline.py

echo "Baseline training completed."