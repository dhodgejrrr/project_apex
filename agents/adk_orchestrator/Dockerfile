FROM python:3.11-slim

ENV PYTHONUNBUFFERED=true
WORKDIR /app

# Copy and install dependencies. This will now work without git.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source code
COPY main.py .

ENV PORT 8080
CMD ["gunicorn", "--bind", ":$PORT", "--workers", "1", "--threads", "8", "--timeout", "0", "main:app"]