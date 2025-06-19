# InsightHunter Dockerfile for Project Apex
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=true
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT 8080
CMD ["gunicorn", "--bind", ":$PORT", "--workers", "1", "--threads", "8", "--timeout", "0", "main:app"]
