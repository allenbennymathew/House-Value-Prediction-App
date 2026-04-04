from housing.ingest import load_housing_data
import pandas as pd

def test_load_housing_data(tmp_path):
    d = tmp_path / "raw"
    d.mkdir()
    p = d / "housing.csv"
    p.write_text("longitude,latitude,housing_median_age,total_rooms,total_bedrooms,population,households,median_income,median_house_value,ocean_proximity\n"
                 "-122.23,37.88,41.0,880.0,129.0,322.0,126.0,8.3252,452600.0,NEAR BAY\n")
    df = load_housing_data(str(d))
    assert len(df) == 1
    assert "median_income" in df.columns
