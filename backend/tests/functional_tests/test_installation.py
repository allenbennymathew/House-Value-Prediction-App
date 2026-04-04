def test_import_housing():
    try:
        import housing
        from housing.ingest import fetch_housing_data
        from housing.train import train_model
        from housing.score import score_model
        success = True
    except ImportError:
        success = False
    assert success
