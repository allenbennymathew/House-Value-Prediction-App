import os

files = {
    "deploy/docker/modelserver.Dockerfile": """# Multi-stage build for optimal performance and security

# Stage 1: Build stage
FROM python:3.10-slim AS builder

WORKDIR /build

# Copy distribution dependencies
COPY dist/*.whl /build/
# Ensure pip is up to date and install the package
RUN pip install --no-cache-dir dist/*.whl
RUN pip install --no-cache-dir uvicorn fastapi mlflow sqlalchemy pydantic

# Prepare final stage
FROM python:3.10-slim

WORKDIR /app

# Create a non-root user
RUN addgroup --system appgroup && adduser --system --group appuser

# Copy installed python packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy app code and artifacts
COPY src/ /app/src/
COPY artifacts/ /app/artifacts/

# Change ownership
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

EXPOSE 8000

# Execute server
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
""",
    "tasks.py": """from invoke import task

@task
def complexity(c):
    \"\"\"Run radon to measure code complexity.\"\"\"
    print("Running Radon Cyclomatic Complexity evaluation...")
    c.run("radon cc src/ scripts/ -a -nc")
    
    print("\\nRunning Radon Maintainability Index evaluation...")
    c.run("radon mi src/ scripts/")
""",
    "config.yml": """training:
  param_grid:
    - forest_reg__n_estimators: [3, 10, 30]
      forest_reg__max_features: [2, 4, 6, 8]
    - forest_reg__bootstrap: [False]
      forest_reg__n_estimators: [3, 10]
      forest_reg__max_features: [2, 3, 4]
"""
}

for filepath, content in files.items():
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)

# Update env.yml
with open("env.yml", "r") as f:
    env_content = f.read()

if "invoke" not in env_content:
    env_content = env_content.replace(
        "  - pip:\\n",
        "  - invoke\\n  - radon\\n  - pyyaml\\n  - pip:\\n"
    )
    with open("env.yml", "w") as f:
        f.write(env_content)

# Update tests/unit_tests/test_train.py
with open("tests/unit_tests/test_train.py", "a") as f:
    f.write('''
def test_custom_features_inverse():
    data = np.array([
        [0, 0, 0, 100, 20, 300, 10, 0],
    ])
    transformer = CustomFeatures()
    transformed = transformer.transform(data)
    recovered = transformer.inverse_transform(transformed)
    np.testing.assert_array_equal(data, recovered)
''')

# Update src/housing/train.py
with open("src/housing/train.py", "r") as f:
    train_code = f.read()

train_code = "import yaml\\n" + train_code
train_code = train_code.replace(
    "return np.c_[X, rooms_per_household, bedrooms_per_room, population_per_household]",
    "return np.c_[X, rooms_per_household, bedrooms_per_room, population_per_household]\\n\\n    def inverse_transform(self, X):\\n        return X[:, :-3]"
)

param_grid_old = """    param_grid = [
        {"forest_reg__n_estimators": [3, 10, 30], "forest_reg__max_features": [2, 4, 6, 8]},
        {"forest_reg__bootstrap": [False], "forest_reg__n_estimators": [3, 10], "forest_reg__max_features": [2, 3, 4]},
    ]"""

param_grid_new = """    param_grid = [
        {"forest_reg__n_estimators": [3, 10, 30], "forest_reg__max_features": [2, 4, 6, 8]},
        {"forest_reg__bootstrap": [False], "forest_reg__n_estimators": [3, 10], "forest_reg__max_features": [2, 3, 4]},
    ]
    if os.path.exists("config.yml"):
        with open("config.yml", "r") as f:
            config = yaml.safe_load(f)
            if "training" in config and "param_grid" in config["training"]:
                param_grid = config["training"]["param_grid"]"""

train_code = train_code.replace(param_grid_old, param_grid_new)

with open("src/housing/train.py", "w") as f:
    f.write(train_code)

# Append to .gitlab-ci.yml
with open(".gitlab-ci.yml", "a") as f:
    f.write('''
build_docker_job:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker build -t my_docker_hub_user/fastapi-model:latest -f deploy/docker/modelserver.Dockerfile .

run_docker_job:
  stage: test
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker run -d -p 8000:8000 --name model_container my_docker_hub_user/fastapi-model:latest
    - sleep 10
    - curl -X POST -H "Content-Type: application/json" -d '{"longitude":-122.23,"latitude":37.88,"housing_median_age":41.0,"total_rooms":880.0,"total_bedrooms":129.0,"population":322.0,"households":126.0,"median_income":8.3252,"ocean_proximity":"NEAR BAY"}' http://model_container:8000/predict/random_forest > curl_output.html
  artifacts:
    paths:
      - curl_output.html
''')

print("Capstone updates applied.")
