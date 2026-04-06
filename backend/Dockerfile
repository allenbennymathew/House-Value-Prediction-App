# Use Python 3.10 slim as base
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Install system dependencies (build-essential for pandas/numpy/scikit-learn)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies (upgrade pip first for best wheel support)
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Install the local package (housing)
RUN pip install --no-deps -e .

# Run the build/training pipeline
# This ensures that all artifacts (model.pkl, etc.) are generated inside the image
RUN python build_all.py

# Expose port (FastAPI default is 8000, Render overrides but let's be explicit)
EXPOSE 8000

# Start command
# We use gunicorn with uvicorn workers for production stability
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "src.app:app", "--bind", "0.0.0.0:10000"]
