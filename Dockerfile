# NeuroPlex AI - FedFusionNet++ Docker Image
# Production-ready containerized deployment for Hugging Face Spaces
# Base image with Python 3.10
FROM python:3.10-slim

# Metadata
LABEL maintainer="Afrid Pasha <afridpasha@example.com>"
LABEL description="NeuroPlex AI - Two-Stage Oral Cancer Detection System"
LABEL version="1.0.0"

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies (updated for Debian Trixie)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    curl \
    wget \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    libgthread-2.0-0 \
    libfontconfig1 \
    libice6 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with increased timeout
RUN pip install --no-cache-dir --timeout=1000 --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --timeout=1000 -r requirements.txt

# Copy the entire application
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p \
    backend/reports \
    backend/models \
    backend/uploads \
    backend/temp \
    logs \
    static/uploads \
    static/temp \
    && chmod -R 755 backend logs static

# Expose port 7860 (Hugging Face Spaces default)
ENV PORT=7860
EXPOSE 7860

# Set Flask environment variables
ENV FLASK_APP=backend/app.py \
    FLASK_ENV=production \
    FLASK_DEBUG=0

# Application-specific environment variables
ENV SECRET_KEY=your_production_secret_key_change_this \
    MAX_CONTENT_LENGTH=16777216 \
    UPLOAD_FOLDER=backend/uploads \
    ALLOWED_EXTENSIONS=png,jpg,jpeg,tiff,svs,ndpi

# VLM Service Configuration (Optional - set via Hugging Face Secrets)
ENV VLM_PROVIDER=gemini \
    GEMINI_API_KEY="" \
    GROQ_API_KEY=""

# MongoDB Configuration (Optional - set via Hugging Face Secrets)
ENV MONGODB_URI="" \
    DATABASE_NAME=fedfusionnet

# Cloudflare R2 Configuration (Optional - set via Hugging Face Secrets)
ENV R2_ACCOUNT_ID="" \
    R2_ACCESS_KEY_ID="" \
    R2_SECRET_ACCESS_KEY="" \
    R2_BUCKET_NAME=neuroplex-reports

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/api/health || exit 1

# Run the Flask application with Gunicorn for production
# Fallback to Flask development server if Gunicorn not available
CMD if command -v gunicorn > /dev/null 2>&1; then \
        gunicorn --bind 0.0.0.0:7860 \
                 --workers 2 \
                 --threads 4 \
                 --timeout 300 \
                 --access-logfile - \
                 --error-logfile - \
                 --log-level info \
                 backend.app:app; \
    else \
        python -m flask run --host=0.0.0.0 --port=7860; \
    fi
