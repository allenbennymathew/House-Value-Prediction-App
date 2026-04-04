# Multi-stage build for optimal performance and security

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
