import os
import tarfile
import numpy as np
import pandas as pd

from scipy.stats import randint
from six.moves import urllib

from sklearn.model_selection import (
    StratifiedShuffleSplit,
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV
)

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ==============================
# DOWNLOAD DATA
# ==============================

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# ==============================
# LOAD DATA
# ==============================

fetch_housing_data()
housing = load_housing_data()

# ==============================
# STRATIFIED SPLIT
# ==============================

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

# ==============================
# PREPARE TRAINING DATA
# ==============================

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

housing_num = housing.drop("ocean_proximity", axis=1)

# Impute missing values
imputer = SimpleImputer(strategy="median")
housing_num_tr = imputer.fit_transform(housing_num)

housing_tr = pd.DataFrame(
    housing_num_tr,
    columns=housing_num.columns,
    index=housing.index
)

# Feature Engineering
housing_tr["rooms_per_household"] = housing_tr["total_rooms"] / housing_tr["households"]
housing_tr["bedrooms_per_room"] = housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
housing_tr["population_per_household"] = housing_tr["population"] / housing_tr["households"]

# One-hot encoding
housing_cat = housing[["ocean_proximity"]]
housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))

# ==============================
# LINEAR REGRESSION
# ==============================

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

lin_predictions = lin_reg.predict(housing_prepared)
lin_rmse = np.sqrt(mean_squared_error(housing_labels, lin_predictions))
lin_mae = mean_absolute_error(housing_labels, lin_predictions)

print("Linear Regression RMSE:", lin_rmse)
print("Linear Regression MAE:", lin_mae)

# ==============================
# DECISION TREE
# ==============================

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)

tree_predictions = tree_reg.predict(housing_prepared)
tree_rmse = np.sqrt(mean_squared_error(housing_labels, tree_predictions))

print("Decision Tree RMSE:", tree_rmse)

# ==============================
# RANDOMIZED SEARCH (Random Forest)
# ==============================

param_distribs = {
    "n_estimators": randint(low=1, high=200),
    "max_features": randint(low=1, high=8),
}

forest_reg = RandomForestRegressor(random_state=42)

rnd_search = RandomizedSearchCV(
    forest_reg,
    param_distributions=param_distribs,
    n_iter=10,
    cv=5,
    scoring="neg_mean_squared_error",
    random_state=42,
)

rnd_search.fit(housing_prepared, housing_labels)

print("\nRandomized Search Results:")
for mean_score, params in zip(
    rnd_search.cv_results_["mean_test_score"],
    rnd_search.cv_results_["params"]
):
    print(np.sqrt(-mean_score), params)

# ==============================
# GRID SEARCH
# ==============================

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

print("\nBest Parameters:", grid_search.best_params_)

# ==============================
# FINAL MODEL EVALUATION
# ==============================

final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_num = X_test.drop("ocean_proximity", axis=1)
X_test_prepared = imputer.transform(X_test_num)

X_test_prepared = pd.DataFrame(
    X_test_prepared,
    columns=X_test_num.columns,
    index=X_test.index
)

# Feature Engineering (test)
X_test_prepared["rooms_per_household"] = (
    X_test_prepared["total_rooms"] / X_test_prepared["households"]
)

X_test_prepared["bedrooms_per_room"] = (
    X_test_prepared["total_bedrooms"] / X_test_prepared["total_rooms"]
)

X_test_prepared["population_per_household"] = (
    X_test_prepared["population"] / X_test_prepared["households"]
)

X_test_cat = X_test[["ocean_proximity"]]
X_test_prepared = X_test_prepared.join(
    pd.get_dummies(X_test_cat, drop_first=True)
)

final_predictions = final_model.predict(X_test_prepared)
final_rmse = np.sqrt(mean_squared_error(y_test, final_predictions))

print("\nFinal Test RMSE:", final_rmse)
