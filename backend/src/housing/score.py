import logging
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
