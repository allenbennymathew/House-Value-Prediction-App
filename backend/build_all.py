import sys
import os
import subprocess

def run_cmd(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

try:
    os.makedirs("logs", exist_ok=True)
    os.makedirs("docs/source", exist_ok=True)
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
    
    # 4. Run Scoring
    print("Running Model Scoring...")
    run_cmd(f'"{sys.executable}" scripts/score.py --log-path logs/score.log')
    
    # 5. Build sphinx config
    conf_py = """
import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

project = 'Housing Prediction'
copyright = '2026'
author = 'Allen'
release = '0.1'

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon']
templates_path = ['_templates']
exclude_patterns = []

html_theme = 'alabaster'
"""
    index_rst = """
Housing Prediction Documentation
================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules
"""
    modules_rst = """
Modules
=======

.. automodule:: housing.ingest
   :members:
   
.. automodule:: housing.train
   :members:
   
.. automodule:: housing.score
   :members:
"""
    with open("docs/source/conf.py", "w") as f: f.write(conf_py)
    with open("docs/source/index.rst", "w") as f: f.write(index_rst)
    with open("docs/source/modules.rst", "w") as f: f.write(modules_rst)
    
    print("Building Sphinx documentation...")
    run_cmd(f'"{sys.executable}" -m sphinx docs/source docs/build/html')
    
    print("SUCCESS: All tasks successfully completed!")
except Exception as e:
    print(f"ERROR OCCURRED: {e}")
