# NeuroPlex Docker Image for Hugging Face Spaces
# Base image with Python 3.10
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies (updated for Debian Trixie)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with increased timeout
RUN pip install --no-cache-dir --timeout=1000 -r requirements.txt

# Copy the entire application
COPY . .

# Create necessary directories
RUN mkdir -p backend/reports backend/models logs

# Expose port 7860 (Hugging Face Spaces default)
ENV PORT=7860
EXPOSE 7860

# Set Flask environment variables
ENV FLASK_APP=backend/app.py \
    FLASK_ENV=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Run the Flask application
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=7860"]
