# Housing Price Prediction

This is a Python library and workflow for housing price prediction via a Random Forest model.

## Installation

You can install this package natively leveraging the built wheel and conda environment variables.
```bash
# Optional Setup using Conda
conda env create -f env.yml
conda activate mle-env

# Native wheel installation (assuming wheels exist in dist/)
pip install dist/*.whl

# Or install from source:
pip install -e .
```

## Workflows Usage
Logs are saved in the `logs/` dir (use `--log-path`) and datasets in the defined directories.

```bash
# 1. Download and Split
python scripts/ingest_data.py --output_path data/processed

# 2. Train the Model 
python scripts/train.py --dataset data/processed --output_folder artifacts

# 3. Score
python scripts/score.py --model_folder artifacts --dataset data/processed
```

## Running MLFlow Full Workflow automatically
Use the main script to orchestrate and track with MLFlow natively.
```bash
python scripts/main.py
```

## Running Web FastAPI endpoint
```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000
```
