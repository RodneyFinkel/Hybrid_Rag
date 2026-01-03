# Use official Python runtime as base image
FROM python:3.12-slim

# Set working directory in container
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies required by packages (include git for GitPython)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libopenblas-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    libssl-dev \
    libffi-dev \
    portaudio19-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy the entire application
COPY . .

# Create directories for uploads and sessions
RUN mkdir -p uploads flask_session chroma_storage

# Expose port for Flask app
EXPOSE 5000

# Avoid GitPython raising on missing/invalid git at runtime
ENV GIT_PYTHON_REFRESH=quiet
ENV GIT_PYTHON_GIT_EXECUTABLE=/usr/bin/git

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000', timeout=5)"

# Run the Flask app with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-", "alpha_app2:app"]
