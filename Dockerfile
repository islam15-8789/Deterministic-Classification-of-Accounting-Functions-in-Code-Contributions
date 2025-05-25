# -------------------------------
# ✅ Stage 1: Build with pip
# -------------------------------
FROM python:3.12-slim AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /usr/src/app

# Copy only requirement files and install dependencies
COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copy app source code (only code — not .venv or dotfiles)
COPY . .

RUN chmod +x run_data_pipeline.sh

# -------------------------------
# ✅ Stage 2: Minimal runtime
# -------------------------------
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /usr/src/app

# Copy installed packages
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# Copy application code (only code, not Python binary!)
COPY --from=builder /usr/src/app /usr/src/app

# Set entrypoint
ENTRYPOINT ["/bin/bash"]
