# This is a universal, multi-stage Dockerfile for all Project Apex agents.
# It is designed to be built from the project root.

# --- Builder Stage ---
# Use a full python image that has all necessary build tools.
FROM python:3.11-slim as builder
WORKDIR /app
COPY . .

# First, install git, which is needed by the ADK requirements.
RUN apt-get update && apt-get install -y git

# Install all Python packages for all agents into a shared layer.
# This is efficient because Cloud Build will cache this layer.
# We first combine all requirements into one file.
RUN cat agents/*/requirements.txt > all_requirements.txt && \
    pip install --no-cache-dir --prefix="/install" -r all_requirements.txt && \
    pip install --no-cache-dir --prefix="/install" -e .

# --- Final Stage ---
# Start from a clean, lightweight image.
FROM python:3.11-slim
ARG AGENT_NAME

ENV PYTHONUNBUFFERED=true
ENV PYTHONPATH=/app
WORKDIR /app/agents/${AGENT_NAME}

# Copy the pre-built dependencies from the 'builder' stage.
COPY --from=builder /install /usr/local

# Copy the entire 'agents' source code.
COPY --from=builder /app/agents /app/agents

# Install WeasyPrint system dependencies ONLY for the scribe agent
RUN if [ "$AGENT_NAME" = "scribe" ]; then \
        apt-get update && apt-get install -y --no-install-recommends \
        libpango-1.0-0 \
        libharfbuzz0b \
        libpangoft2-1.0-0 \
        libpangocairo-1.0-0 \
        libgdk-pixbuf-2.0-0 \
        libcairo2 && \
        rm -rf /var/lib/apt/lists/*; \
    fi

ENV PORT 8080
CMD gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app