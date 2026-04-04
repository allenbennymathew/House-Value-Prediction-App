import mlflow
import sys
from ingest_data import main as ingest_main
from train import main as train_main
from score import main as score_main

def run_workflow():
    with mlflow.start_run(run_name="main_workflow") as parent_run:
        # Step 1: Ingest
        with mlflow.start_run(run_name="ingestion", nested=True):
            sys.argv = ["ingest_data.py"]
            ingest_main()
            
        # Step 2: Train
        with mlflow.start_run(run_name="training", nested=True):
            sys.argv = ["train.py"]
            train_main()
            
        # Step 3: Score
        with mlflow.start_run(run_name="scoring", nested=True):
            sys.argv = ["score.py"]
            score_main()
            
if __name__ == "__main__":
    run_workflow()
    print("Workflow complete.")
