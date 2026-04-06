import argparse
import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from housing.logger import setup_logger
import logging

def main():
    parser = argparse.ArgumentParser(description="Score house pricing model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the .pkl model file")
    parser.add_argument("--dataset", type=str, default="data/processed", help="Path to evaluation dataset")
    parser.add_argument("--append", action="store_true", help="Append to metrics file instead of overwriting")
    parser.add_argument("--output", type=str, default="artifacts/metrics.txt", help="Path to output results summary")
    parser.add_argument("--log-level", type=str, default="DEBUG")
    parser.add_argument("--log-path", type=str)
    parser.add_argument("--no-console-log", action="store_true")
    
    args = parser.parse_args()
    setup_logger(args.log_level, args.log_path, args.no_console_log)
    logger = logging.getLogger("score")
    
    logger.info(f"Scoring model: {args.model_path}")
    
    test_path = os.path.join(args.dataset, "test.csv")
    if not all([os.path.exists(test_path), os.path.exists(args.model_path)]):
        logger.error(f"Required files not found: {test_path} or {args.model_path}")
        sys.exit(1)
        
    test_set = pd.read_csv(test_path)
    X_test = test_set.drop("median_house_value", axis=1)
    y_test = test_set["median_house_value"].copy()
    
    with open(args.model_path, "rb") as f:
        model = pickle.load(f)
        
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    model_name = os.path.basename(args.model_path).replace(".pkl", "")
    metrics_str = f"{model_name}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.4f}\n"
    
    mode = "a" if args.append else "w"
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, mode) as f:
        f.write(metrics_str)
        
    logger.info(f"Metrics for {model_name} saved to {args.output}: {metrics_str.strip()}")

if __name__ == "__main__":
    main()
