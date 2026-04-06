import argparse
import os
import sys
import pickle
import pandas as pd
from housing.train import train_models
from housing.logger import setup_logger
import logging

def main():
    parser = argparse.ArgumentParser(description="Train house pricing model.")
    parser.add_argument("--dataset", type=str, default="data/processed", help="Path to input datasets folder")
    parser.add_argument("--output_folder", type=str, default="artifacts", help="Path to output models and artifacts")
    parser.add_argument("--log-level", type=str, default="DEBUG", help="Log level")
    parser.add_argument("--log-path", type=str, help="Path to write logs to a file")
    parser.add_argument("--no-console-log", action="store_true", help="Toggle whether or not to write logs to the console")
    
    args = parser.parse_args()
    setup_logger(args.log_level, args.log_path, args.no_console_log)
    logger = logging.getLogger("train")
    
    # Removed MLFlow tracking for production stability
    logger.info("Starting model training process.")
    train_path = os.path.join(args.dataset, "train.csv")
    if not os.path.exists(train_path):
        logger.error(f"Training dataset not found at {train_path}")
        sys.exit(1)
        
    train_set = pd.read_csv(train_path)
    housing = train_set.drop("median_house_value", axis=1)
    housing_labels = train_set["median_house_value"].copy()
    
    models = train_models(housing, housing_labels)
    
    os.makedirs(args.output_folder, exist_ok=True)
    for name, model in models.items():
        model_path = os.path.join(args.output_folder, f"{name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
