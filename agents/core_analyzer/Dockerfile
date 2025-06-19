# CoreAnalyzer Dockerfile for Project Apex
FROM python:3.11-slim

# Allow statements and log output to show immediately
ENV PYTHONUNBUFFERED true

WORKDIR /app

# Install Python dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Default port for Cloud Run
ENV PORT 8080

# Use Gunicorn with single worker (multithreaded) - Cloud Run scales horizontally
CMD ["gunicorn", "--bind", ":$PORT", "--workers", "1", "--threads", "8", "--timeout", "0", "main:app"]
