# Time-FLM
This repository contains the official implementation of Time-FLM, a Universal Large Model for fed-batch process Time Series Forecasting.

## Installation
Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Configuration
Hyperparameters and settings are stored in YAML configuration files:
- `config.yaml`: For Time-FLM model
- `config_baseline.yaml`: For baseline models

Edit these files to adjust parameters. The random seed is set to 42 for reproducibility.

## Data Preparation
Place your datasets in the `./datasets/` directory. Update the `root_path` in the config files accordingly.

## Training
You can run training using Python scripts directly or via bash scripts for convenience.

### Using Comprehensive Bash Script
For automated training with logging:
```bash
./train_all.sh flm        # Run Time-FLM only
./train_all.sh baseline   # Run baselines only
./train_all.sh all        # Run both
```

### Using Individual Bash Scripts
To train Time-FLM:
```bash
./run_train.sh
```

To run baseline models:
```bash
./run_baseline_train.sh
```

### Using Python Scripts Directly
To train Time-FLM:
```bash
python run_FLM.py
```

To run baseline models:
```bash
python run_baseline.py
```

Logs will be saved in the `logs/` directory.
