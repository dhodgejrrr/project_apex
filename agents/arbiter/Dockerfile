# Universal Dockerfile for all Project Apex Agents

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=true
WORKDIR /app

# Copy the entire 'agents' directory into the build context
COPY ./agents /app/agents

# Copy the specific agent's requirements file
COPY ./agents/arbiter/requirements.txt .

# Install dependencies. The -e flag installs the 'agents' directory
# as an editable package, making 'agents.common' importable.
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e ./agents

# Set the final working directory to the specific agent's folder
WORKDIR /app/agents/arbiter

ENV PORT 8080
CMD ["gunicorn", "--bind", ":$PORT", "--workers", "1", "--threads", "8", "--timeout", "0", "main:app"]