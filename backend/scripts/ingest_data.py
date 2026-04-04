import argparse
import os
from housing.ingest import fetch_housing_data, load_housing_data, prepare_datasets
from housing.logger import setup_logger
import logging
import mlflow

def main():
    parser = argparse.ArgumentParser(description="Ingest housing data.")
    parser.add_argument("--output_path", type=str, default="data/processed", help="Output path")
    parser.add_argument("--log-level", type=str, default="DEBUG")
    parser.add_argument("--log-path", type=str)
    parser.add_argument("--no-console-log", action="store_true")
    
    args = parser.parse_args()
    setup_logger(args.log_level, args.log_path, args.no_console_log)
    logger = logging.getLogger("ingest_data")
    
    with mlflow.start_run(run_name="ingest_script") as run:
        logger.info("Starting data ingestion process.")
        fetch_housing_data(housing_path="data/raw")
        housing = load_housing_data(housing_path="data/raw")
        train_set, test_set = prepare_datasets(housing)
        
        os.makedirs(args.output_path, exist_ok=True)
        train_path = os.path.join(args.output_path, "train.csv")
        test_path = os.path.join(args.output_path, "test.csv")
        
        train_set.to_csv(train_path, index=False)
        test_set.to_csv(test_path, index=False)
        
        mlflow.log_param("train_samples", len(train_set))
        mlflow.log_param("test_samples", len(test_set))
        logger.info(f"Train dataset saved to {train_path}")
        logger.info(f"Test dataset saved to {test_path}")

if __name__ == "__main__":
    main()
