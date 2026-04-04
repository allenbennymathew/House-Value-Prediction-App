import argparse
import os
import sys
import pickle
import pandas as pd
from housing.score import score_model
from housing.logger import setup_logger
import logging
import mlflow

def main():
    parser = argparse.ArgumentParser(description="Score house pricing model.")
    parser.add_argument("--model_folder", type=str, default="artifacts", help="Path to model and imputer folder")
    parser.add_argument("--dataset", type=str, default="data/processed", help="Path to evaluation dataset")
    parser.add_argument("--output", type=str, default="artifacts", help="Path to output results summary")
    parser.add_argument("--log-level", type=str, default="DEBUG", help="Log level")
    parser.add_argument("--log-path", type=str, help="Path to write logs to a file")
    parser.add_argument("--no-console-log", action="store_true", help="Toggle whether or not to write logs to the console")
    
    args = parser.parse_args()
    setup_logger(args.log_level, args.log_path, args.no_console_log)
    logger = logging.getLogger("score")
    
    with mlflow.start_run(run_name="score_script") as run:
        logger.info("Starting model scoring process.")
        
        test_path = os.path.join(args.dataset, "test.csv")
        model_path = os.path.join(args.model_folder, "model.pkl")
        
        if not all([os.path.exists(test_path), os.path.exists(model_path)]):
            logger.error("Required files not found.")
            sys.exit(1)
            
        test_set = pd.read_csv(test_path)
        X_test = test_set.drop("median_house_value", axis=1)
        y_test = test_set["median_house_value"].copy()
        
        with open(model_path, "rb") as f:
            model = pickle.load(f)
            
        rmse = score_model(model, X_test, y_test)
        
        os.makedirs(args.output, exist_ok=True)
        metric_file = os.path.join(args.output, "metrics.txt")
        with open(metric_file, "w") as f:
            f.write(f"Test RMSE: {rmse}\n")
            
        logger.info(f"Metrics saved to {metric_file}")
        mlflow.log_metric("Test_RMSE", rmse)
        mlflow.log_artifact(metric_file)

if __name__ == "__main__":
    main()
