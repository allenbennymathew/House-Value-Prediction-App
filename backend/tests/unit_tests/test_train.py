import pandas as pd
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

def test_custom_features_inverse():
    data = np.array([
        [0, 0, 0, 100, 20, 300, 10, 0],
    ])
    transformer = CustomFeatures()
    transformed = transformer.transform(data)
    recovered = transformer.inverse_transform(transformed)
    np.testing.assert_array_equal(data, recovered)
