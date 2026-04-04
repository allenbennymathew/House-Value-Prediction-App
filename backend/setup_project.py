import os

files = {
    "setup.py": """from setuptools import setup, find_packages

setup(
    name="housing",
    version="0.2",
    description="Housing price prediction model",
    author="Allen",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "six"
    ],
)
""",
    "src/housing/__init__.py": "",
    "src/housing/logger.py": """import logging
import sys

def setup_logger(log_level="DEBUG", log_path=None, no_console_log=False):
    logger = logging.getLogger() # root logger
    
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    if not no_console_log:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    if log_path:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
""",
    "src/housing/ingest.py": """import os
import tarfile
import logging
import numpy as np
import pandas as pd
from six.moves import urllib
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path="data/raw"):
    logger.info(f"Fetching housing data from {housing_url} to {housing_path}")
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    logger.info("Data extraction complete.")

def load_housing_data(housing_path="data/raw"):
    csv_path = os.path.join(housing_path, "housing.csv")
    logger.info(f"Loading housing data from {csv_path}")
    return pd.read_csv(csv_path)

def prepare_datasets(housing):
    logger.info("Preparing train and test datasets based on income category.")
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)
    logger.info("Train and test splits created successfully.")
    return strat_train_set, strat_test_set
""",
    "src/housing/train.py": """import logging
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

logger = logging.getLogger(__name__)

def preprocess_data(housing, imputer=None):
    logger.info("Preprocessing data...")
    housing_num = housing.drop("ocean_proximity", axis=1)
    
    if imputer is None:
        imputer = SimpleImputer(strategy="median")
        imputer.fit(housing_num)
        
    housing_num_tr = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(
        housing_num_tr,
        columns=housing_num.columns,
        index=housing.index
    )
    
    housing_tr["rooms_per_household"] = housing_tr["total_rooms"] / housing_tr["households"]
    housing_tr["bedrooms_per_room"] = housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
    housing_tr["population_per_household"] = housing_tr["population"] / housing_tr["households"]
    
    housing_cat = housing[["ocean_proximity"]]
    housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))
    
    return housing_prepared, imputer

def train_model(housing_prepared, housing_labels):
    logger.info("Training Random Forest model using Grid Search...")
    forest_reg = RandomForestRegressor(random_state=42)
    param_grid = [
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
    ]
    grid_search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    grid_search.fit(housing_prepared, housing_labels)
    logger.info(f"Best Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_
""",
    "src/housing/score.py": """import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)

def score_model(model, X_test_prepared, y_test):
    logger.info("Scoring model...")
    predictions = model.predict(X_test_prepared)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    logger.info(f"Test RMSE: {rmse}")
    return rmse
""",
    "scripts/ingest_data.py": """import argparse
import os
import sys
from housing.ingest import fetch_housing_data, load_housing_data, prepare_datasets
from housing.logger import setup_logger
import logging

def main():
    parser = argparse.ArgumentParser(description="Ingest housing data.")
    parser.add_argument("--output_path", type=str, default="data/processed", help="Output path for train and test datasets")
    parser.add_argument("--log-level", type=str, default="DEBUG", help="Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    parser.add_argument("--log-path", type=str, help="Path to write logs to a file")
    parser.add_argument("--no-console-log", action="store_true", help="Toggle whether or not to write logs to the console")
    
    args = parser.parse_args()
    
    setup_logger(args.log_level, args.log_path, args.no_console_log)
    logger = logging.getLogger("ingest_data")
    
    logger.info("Starting data ingestion process.")
    fetch_housing_data(housing_path="data/raw")
    housing = load_housing_data(housing_path="data/raw")
    train_set, test_set = prepare_datasets(housing)
    
    os.makedirs(args.output_path, exist_ok=True)
    train_path = os.path.join(args.output_path, "train.csv")
    test_path = os.path.join(args.output_path, "test.csv")
    
    train_set.to_csv(train_path, index=False)
    test_set.to_csv(test_path, index=False)
    
    logger.info(f"Train dataset saved to {train_path}")
    logger.info(f"Test dataset saved to {test_path}")

if __name__ == "__main__":
    main()
""",
    "scripts/train.py": """import argparse
import os
import sys
import pickle
import pandas as pd
from housing.train import preprocess_data, train_model
from housing.logger import setup_logger
import logging

def main():
    parser = argparse.ArgumentParser(description="Train house pricing model.")
    parser.add_argument("--dataset", type=str, default="data/processed", help="Path to input datasets folder")
    parser.add_argument("--output_folder", type=str, default="artifacts", help="Path to output models and artifacts")
    parser.add_argument("--log-level", type=str, default="DEBUG", help="Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    parser.add_argument("--log-path", type=str, help="Path to write logs to a file")
    parser.add_argument("--no-console-log", action="store_true", help="Toggle whether or not to write logs to the console")
    
    args = parser.parse_args()
    setup_logger(args.log_level, args.log_path, args.no_console_log)
    logger = logging.getLogger("train")
    
    logger.info("Starting model training process.")
    
    train_path = os.path.join(args.dataset, "train.csv")
    if not os.path.exists(train_path):
        logger.error(f"Training dataset not found at {train_path}")
        sys.exit(1)
        
    train_set = pd.read_csv(train_path)
    housing = train_set.drop("median_house_value", axis=1)
    housing_labels = train_set["median_house_value"].copy()
    
    housing_prepared, imputer = preprocess_data(housing)
    model = train_model(housing_prepared, housing_labels)
    
    os.makedirs(args.output_folder, exist_ok=True)
    model_path = os.path.join(args.output_folder, "model.pkl")
    imputer_path = os.path.join(args.output_folder, "imputer.pkl")
    
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
        
    with open(imputer_path, "wb") as f:
        pickle.dump(imputer, f)
        
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Imputer saved to {imputer_path}")

if __name__ == "__main__":
    main()
""",
    "scripts/score.py": """import argparse
import os
import sys
import pickle
import pandas as pd
from housing.train import preprocess_data
from housing.score import score_model
from housing.logger import setup_logger
import logging

def main():
    parser = argparse.ArgumentParser(description="Score house pricing model.")
    parser.add_argument("--model_folder", type=str, default="artifacts", help="Path to model and imputer folder")
    parser.add_argument("--dataset", type=str, default="data/processed", help="Path to evaluation dataset")
    parser.add_argument("--output", type=str, default="artifacts", help="Path to output results summary")
    parser.add_argument("--log-level", type=str, default="DEBUG", help="Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    parser.add_argument("--log-path", type=str, help="Path to write logs to a file")
    parser.add_argument("--no-console-log", action="store_true", help="Toggle whether or not to write logs to the console")
    
    args = parser.parse_args()
    setup_logger(args.log_level, args.log_path, args.no_console_log)
    logger = logging.getLogger("score")
    
    logger.info("Starting model scoring process.")
    
    test_path = os.path.join(args.dataset, "test.csv")
    model_path = os.path.join(args.model_folder, "model.pkl")
    imputer_path = os.path.join(args.model_folder, "imputer.pkl")
    
    if not all([os.path.exists(test_path), os.path.exists(model_path), os.path.exists(imputer_path)]):
        logger.error(f"Required files not found. Ensure {test_path}, {model_path}, and {imputer_path} exist.")
        sys.exit(1)
        
    test_set = pd.read_csv(test_path)
    X_test = test_set.drop("median_house_value", axis=1)
    y_test = test_set["median_house_value"].copy()
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    with open(imputer_path, "rb") as f:
        imputer = pickle.load(f)
        
    X_test_prepared, _ = preprocess_data(X_test, imputer=imputer)
    
    rmse = score_model(model, X_test_prepared, y_test)
    
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "metrics.txt"), "w") as f:
        f.write(f"Test RMSE: {rmse}\\n")
        
    logger.info(f"Metrics saved to {os.path.join(args.output, 'metrics.txt')}")

if __name__ == "__main__":
    main()
""",
    "tests/functional_tests/test_installation.py": """def test_import_housing():
    try:
        import housing
        from housing.ingest import fetch_housing_data
        from housing.train import train_model
        from housing.score import score_model
        success = True
    except ImportError:
        success = False
    assert success
""",
    "tests/unit_tests/test_train.py": """import pandas as pd
import numpy as np
from housing.train import preprocess_data

def test_preprocess_data():
    data = {
        "ocean_proximity": ["<1H OCEAN", "INLAND"],
        "total_rooms": [100, 200],
        "households": [10, 20],
        "total_bedrooms": [20, 40],
        "population": [300, 600]
    }
    df = pd.DataFrame(data)
    processed, imputer = preprocess_data(df)
    
    assert "rooms_per_household" in processed.columns
    assert "bedrooms_per_room" in processed.columns
    assert "population_per_household" in processed.columns
    assert imputer is not None
""",
    "tests/unit_tests/test_ingest.py": """from housing.ingest import load_housing_data
import pandas as pd

def test_load_housing_data(tmp_path):
    d = tmp_path / "raw"
    d.mkdir()
    p = d / "housing.csv"
    p.write_text("longitude,latitude,housing_median_age,total_rooms,total_bedrooms,population,households,median_income,median_house_value,ocean_proximity\\n"
                 "-122.23,37.88,41.0,880.0,129.0,322.0,126.0,8.3252,452600.0,NEAR BAY\\n")
    df = load_housing_data(str(d))
    assert len(df) == 1
    assert "median_income" in df.columns
""",
    ".flake8": """[flake8]
max-line-length = 120
exclude = .git,__pycache__,docs/source/conf.py,old,build,dist
"""
}

for filepath, content in files.items():
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)

print("Project setup complete!")
