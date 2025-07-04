# ASL-to-Text AI - Optimized for Google Cloud Run
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PORT=8080
ENV PYTHONPATH=/app/src

# Set work directory
WORKDIR /app

# Install system dependencies (minimal for Cloud Run)
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libopencv-dev \
    libjpeg-dev \
    libpng-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/models /app/data/datasets /app/data/vocabulary

# Expose port (Cloud Run uses PORT env variable)
EXPOSE 8080

# Run as non-root user
RUN useradd -m -u 1000 asl-user && chown -R asl-user:asl-user /app
USER asl-user

# Default command
CMD ["python", "web_app/app.py"]
