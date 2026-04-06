import sys
import os
import subprocess

def run_cmd(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

try:
    # 0. Clean and Initialize Directories
    import shutil
    for dir_name in ["logs", "docs/source", "artifacts"]:
        os.makedirs(dir_name, exist_ok=True)
    
    if os.path.exists("artifacts"):
        print("Clearing old artifacts...")
        shutil.rmtree("artifacts")
    os.makedirs("artifacts", exist_ok=True)
    
    # 1. Install local package properly
    print("Installing backend local package...")
    run_cmd(f'"{sys.executable}" -m pip install -e .')
    
    # 2. Run Ingestion
    print("Running Data Ingestion...")
    run_cmd(f'"{sys.executable}" scripts/ingest_data.py --log-path logs/ingest.log')
    
    # 3. Run Training
    print("Running Model Training...")
    run_cmd(f'"{sys.executable}" scripts/train.py --log-path logs/train.log')
    
    # 4. Run Scoring for ALL models
    print("Running Model Scoring for all engines...")
    # First model overwrites, others append
    run_cmd(f'"{sys.executable}" scripts/score.py --model_path artifacts/linear_regression.pkl --log-path logs/score.log')
    run_cmd(f'"{sys.executable}" scripts/score.py --model_path artifacts/decision_tree.pkl --append --log-path logs/score.log')
    run_cmd(f'"{sys.executable}" scripts/score.py --model_path artifacts/random_forest.pkl --append --log-path logs/score.log')
    
    # 5. Build sphinx config (optional)
    try:
        import sphinx
        print("Building Sphinx documentation...")
        run_cmd(f'"{sys.executable}" -m sphinx docs/source docs/build/html')
    except ImportError:
        print("Sphinx not installed, skipping documentation build.")
    
    # 6. Verify Artifacts
    print("Verifying generated artifacts...")
    required_files = ["artifacts/linear_regression.pkl", "artifacts/decision_tree.pkl", "artifacts/random_forest.pkl", "artifacts/metrics.txt"]
    for f in required_files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"CRITICAL ERROR: {f} was not generated during build!")
        print(f"Verified: {f}")

    print("SUCCESS: All tasks successfully completed!")
except Exception as e:
    print(f"ERROR OCCURRED DURING BUILD: {e}")
    sys.exit(1)
