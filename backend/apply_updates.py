import os

files = {
    "src/housing/train.py": """import logging
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

logger = logging.getLogger(__name__)

class CustomFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): 
        return self
    def transform(self, X):
        rooms_per_household = X[:, 3] / X[:, 6]
        bedrooms_per_room = X[:, 4] / X[:, 3]
        population_per_household = X[:, 5] / X[:, 6]
        return np.c_[X, rooms_per_household, bedrooms_per_room, population_per_household]

def train_model(housing, housing_labels):
    logger.info("Training Random Forest model using Grid Search with Pipeline...")
    
    housing_num = housing.drop("ocean_proximity", axis=1)
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CustomFeatures()),
    ])

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_attribs),
    ])
    
    full_pipeline_with_predictor = Pipeline([
        ("preparation", full_pipeline),
        ("forest_reg", RandomForestRegressor(random_state=42))
    ])
    
    param_grid = [
        {"forest_reg__n_estimators": [3, 10, 30], "forest_reg__max_features": [2, 4, 6, 8]},
        {"forest_reg__bootstrap": [False], "forest_reg__n_estimators": [3, 10], "forest_reg__max_features": [2, 3, 4]},
    ]
    
    grid_search = GridSearchCV(
        full_pipeline_with_predictor,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    
    grid_search.fit(housing, housing_labels)
    logger.info(f"Best Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_
""",
    "src/housing/score.py": """import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)

def score_model(model, X_test, y_test):
    logger.info("Scoring model...")
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    logger.info(f"Test RMSE: {rmse}")
    return rmse
""",
    "scripts/train.py": """import argparse
import os
import sys
import pickle
import pandas as pd
from housing.train import train_model
from housing.logger import setup_logger
import logging
import mlflow

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
    
    with mlflow.start_run(run_name="train_script") as run:
        mlflow.log_param("dataset", args.dataset)
        
        logger.info("Starting model training process.")
        train_path = os.path.join(args.dataset, "train.csv")
        if not os.path.exists(train_path):
            logger.error(f"Training dataset not found at {train_path}")
            sys.exit(1)
            
        train_set = pd.read_csv(train_path)
        housing = train_set.drop("median_house_value", axis=1)
        housing_labels = train_set["median_house_value"].copy()
        
        model = train_model(housing, housing_labels)
        
        os.makedirs(args.output_folder, exist_ok=True)
        model_path = os.path.join(args.output_folder, "model.pkl")
        
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
            
        logger.info(f"Model saved to {model_path}")
        mlflow.log_artifact(model_path)

if __name__ == "__main__":
    main()
""",
    "scripts/score.py": """import argparse
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
            f.write(f"Test RMSE: {rmse}\\n")
            
        logger.info(f"Metrics saved to {metric_file}")
        mlflow.log_metric("Test_RMSE", rmse)
        mlflow.log_artifact(metric_file)

if __name__ == "__main__":
    main()
""",
    "scripts/ingest_data.py": """import argparse
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
""",
    "scripts/main.py": """import mlflow
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
""",
    "src/app.py": """import os
import pickle
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from datetime import datetime
from pydantic import BaseModel

Base = declarative_base()

class InferenceRecord(Base):
    __tablename__ = "inferences"
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, index=True)
    input_data = Column(String)
    prediction = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

engine = create_engine("sqlite:///./inferences.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

app = FastAPI(title="Housing Price Prediction API")

@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class InputData(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: str

class RecordResponse(BaseModel):
    id: int
    model_name: str
    input_data: str
    prediction: float
    timestamp: datetime
    class Config:
        orm_mode = True

def get_prediction(data_dict, model_name="random_forest"):
    model_path = "artifacts/model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model pickle not found in artifacts/model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    df = pd.DataFrame([data_dict])
    pred = model.predict(df)
    return pred[0]

@app.post("/predict/{model_name}", response_model=RecordResponse)
def predict(model_name: str, input_data: InputData, db: Session = Depends(get_db)):
    data_dict = input_data.dict()
    try:
        prediction = float(get_prediction(data_dict, model_name))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    record = InferenceRecord(
        model_name=model_name,
        input_data=str(data_dict),
        prediction=prediction
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record

@app.get("/inferences/{model_name}")
def get_inferences(model_name: str, db: Session = Depends(get_db)):
    records = db.query(InferenceRecord).filter(InferenceRecord.model_name == model_name).all()
    return records
""",
    "README.md": """# Housing Price Prediction

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
""",
    "tests/unit_tests/test_train.py": """import pandas as pd
import numpy as np
from housing.train import CustomFeatures

def test_custom_features():
    # test our Scikit learn custom transformer
    data = np.array([
        [0, 0, 0, 100, 20, 300, 10, 0],  # total_rooms=100, total_bedrooms=20, population=300, households=10
    ])
    transformer = CustomFeatures()
    transformed = transformer.transform(data)
    
    # rooms_per_hh = 100/10 = 10
    # bedrooms_per_room = 20/100 = 0.2
    # pop_per_hh = 300/10 = 30
    assert transformed[0, -3] == 10.0
    assert transformed[0, -2] == 0.2
    assert transformed[0, -1] == 30.0
""",
    "env.yml": """name: mle-env
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.10
  - numpy
  - pandas
  - scikit-learn
  - pytest
  - flake8
  - sphinx
  - pip
  - mlflow
  - fastapi
  - uvicorn
  - sqlalchemy
  - pip:
    - "-e ."
"""
}

for filepath, content in files.items():
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)

print("Files successfully updated and rewritten!")
