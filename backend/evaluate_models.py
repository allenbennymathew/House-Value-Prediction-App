"""
Quick model evaluation script — tests all 3 trained models on the held-out test set.
Run from: backend/
  py evaluate_models.py
"""
import os, sys, pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

TEST_CSV  = "data/processed/test.csv"
ART_DIR   = "artifacts"
MODELS    = ["linear_regression", "decision_tree", "random_forest"]

def load_test():
    df = pd.read_csv(TEST_CSV)
    X  = df.drop("median_house_value", axis=1)
    y  = df["median_house_value"]
    return X, y

def evaluate(model, X, y):
    preds = model.predict(X)
    rmse  = np.sqrt(mean_squared_error(y, preds))
    mae   = mean_absolute_error(y, preds)
    r2    = r2_score(y, preds)
    return rmse, mae, r2

def main():
    print("=" * 65)
    print(f"{'MODEL':<25} {'RMSE':>12} {'MAE':>12} {'R²':>10}")
    print("=" * 65)
    
    if not os.path.exists(TEST_CSV):
        print(f"ERROR: Test data not found at {TEST_CSV}")
        print("Run: py scripts/ingest_data.py  first.")
        sys.exit(1)

    X, y = load_test()
    results = {}

    for name in MODELS:
        path = os.path.join(ART_DIR, f"{name}.pkl")
        if not os.path.exists(path):
            print(f"  ⚠ {name:<23} NOT FOUND — run py build_all.py")
            continue
        with open(path, "rb") as f:
            model = pickle.load(f)
        rmse, mae, r2 = evaluate(model, X, y)
        results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2}
        print(f"  {name:<23} ${rmse:>10,.0f}  ${mae:>10,.0f}  {r2:>9.4f}")

    print("=" * 65)
    if results:
        best = min(results, key=lambda k: results[k]["RMSE"])
        print(f"\n  ✅ Best model by RMSE: {best.upper()} (${results[best]['RMSE']:,.0f})")
        print()
        # Persist all metrics
        with open(os.path.join(ART_DIR, "metrics.txt"), "w") as f:
            for name, m in results.items():
                f.write(f"{name}: RMSE={m['RMSE']:.2f}, MAE={m['MAE']:.2f}, R2={m['R2']:.4f}\n")
        print("  Metrics written to artifacts/metrics.txt")

if __name__ == "__main__":
    main()
