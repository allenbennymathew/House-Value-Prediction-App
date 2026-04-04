# ⌂ Sovereign: EXHAUSTIVE File System Registry

This is the absolute map of every file and folder currently in the project. Use this for deep technical discussions or for when someone asks, "What exactly is in this codebase?"

---

## 📁 Root Directory
*   **`backend/`**: The entire server-side application (Python/FastAPI).
*   **`frontend/`**: The entire client-side web application (React/Vite).
*   **`README_SETUP.md`**: Official guide to installing and running the system.
*   **`INTERVIEW_NARRATIVE.md`**: Strategic guide for project presentation.
*   **`PROJECT_STRUCTURE.md`**: (This file) A complete index of the repository.
*   **`kill_uvicorn.py`**: A utility script to quickly stop any hidden backend processes.

---

## 📁 `backend/` (The Engine)
### 📂 Root Level
*   **`build_all.py`**: The master automation script for a full system rebuild.
*   **`setup.py`**: Standard Python package config; installs the `housing` module.
*   **`setup_project.py`**: A setup script used for initial project initialization.
*   **`requirements.txt`**: List of all third-party Python libraries.
*   **`env.yml`**: Configuration for creating a Conda environment.
*   **`config.yml`**: Project-wide configuration settings (hyperparameters, paths).
*   **`inferences.db`**: The SQLite database where all predictions are archived.
*   **`tasks.py`**: Command-line task runner (Invoke/Fabric style).
*   **`evaluate_models.py`**: A developer-only script to run benchmarking tests.
*   **`nonstandardcode.py`**: A clean, standalone playground for experimental scripts.
*   **`apply_updates.py`**: A maintenance script for patching the system.
*   **`.flake8`**: Formatting rules for the Python linter.
*   **`.gitlab-ci.yml`**: Automation for continuous integration and testing.

### 📂 `backend/src/` (Source Code)
*   **`app.py`**: The **Master API**. It is the gateway between the models and the Web UI.
*   **`housing/`**: The core package for the ML pipeline.
    *   `__init__.py`: Makes the folder recognized as a Python package.
    *   `ingest.py`: Core logic for data extraction and cleaning.
    *   `train.py`: Core logic for model training architectures.
    *   `score.py`: Pure logic for evaluation and metrics calculation.
    *   `logger.py`: System-wide logging configuration for internal tracking.

### 📂 `backend/scripts/` (Operational Scripts)
*   **`main.py`**: Entry point for running the entire backend locally.
*   **`ingest_data.py`**: Script to fetch fresh house price data.
*   **`train.py`**: Command-line interface to trigger a new training cycle.
*   **`score.py`**: CLI tool to generate performance reports without the UI.

### 📂 `backend/artifacts/` (Model Vault)
*   **`random_forest.pkl`**: The saved trained Random Forest model.
*   **`decision_tree.pkl`**: The saved trained Decision Tree model.
*   **`linear_regression.pkl`**: The saved trained Linear Regression model.
*   **`metrics.txt`**: A live text file containing performance data for the UI to read.
*   **`model.pkl`**: A duplicate/backup of the primary active model.

### 📂 `backend/data/` (Storage)
*   **`housing.csv`**: The local copy of the raw California dataset.
*   **`processed/`**: Folder containing clean data ready for ingestion.

### 📂 `backend/docs/` (Documentation)
*   **`source/` & `build/`**: Files for the automatic user manual (Sphinx documentation).

---

## 📁 `frontend/` (The Terminal UI)
### 📂 `frontend/src/` (Web Source)
*   **`main.jsx`**: The React entry point that boots up the entire interface.
*   **`App.jsx`**: The core application engine (Pages, History, Modals, API calls).
*   **`index.css`**: The custom "Sovereign" Design System & Dark Mode theme.
*   **`assets/`**: Static image files and logo assets.

### 📂 Root Level
*   **`package.json`**: The core configuration for all JavaScript dependencies.
*   **`vite.config.js`**: Optimization rules for the building engine and hot-reloading.
*   **`index.html`**: The foundational HTML page that hosts the React app.
*   **`eslint.config.js`**: Rules to ensure the code stays bug-free and clean.
*   **`node_modules/`**: Thousands of utility files that make React and Vite run.
