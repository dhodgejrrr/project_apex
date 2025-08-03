---
description: Repository Information Overview
alwaysApply: true
---

# Project Apex Information

## Summary
Project Apex is an autonomous, multi-agent system built on Google's Agent Development Kit (ADK) that leverages Vertex AI to transform raw motorsport data into actionable insights. The system consists of multiple specialized AI agents that analyze, investigate, synthesize, and generate content from motorsport telemetry data.

## Structure
- **agents/**: Contains all agent implementations, each in its own directory
- **scripts/**: Deployment and infrastructure setup scripts
- **project_documents/**: Documentation and design files
- **tests/**: End-to-end test implementations
- **local_raw/**: Sample raw data for local testing
- **out_local/**: Output directory for local runs

## Language & Runtime
**Language**: Python
**Version**: Python 3.11 (production), 3.13 (development)
**Build System**: Docker-based containerization
**Package Manager**: pip

## Dependencies
**Main Dependencies**:
- Flask (API framework)
- Google Cloud Storage/BigQuery (data storage)
- Pandas/NumPy (data analysis)
- Agent Development Kit (ADK) (agent orchestration)
- Vertex AI Gemini API (AI reasoning)
- Streamlit (UI portal)

**Development Dependencies**:
- pytest (testing)
- Cloud emulators (local development)

## Build & Installation
```bash
# Local development setup
pip install -r requirements.txt

# Local pipeline run
python full_pipeline_local.py --out_dir ./local_run_output
```

## Docker
**Dockerfile**: Universal multi-stage Dockerfile for all agents
**Image**: Each agent has its own container image
**Configuration**: 
- Base Python 3.11 slim image
- Shared dependency layer for efficiency
- Agent-specific environment variables
- Special dependencies for specific agents (e.g., WeasyPrint for scribe)

## Deployment
**Cloud Platform**: Google Cloud Platform
**Services**: Cloud Run (serverless containers)
**Deployment Method**:

```bash
# Build and deploy all agents
./scripts/deploy_agents.sh

# Deploy specific agents
./scripts/deploy_agents.sh core_analyzer historian
```

**Agent Deployment Configuration**:
- Memory: 2Gi
- CPU: 2 cores
- Timeout: 900 seconds
- Service Account: apex-agent-runner@[PROJECT_ID].iam.gserviceaccount.com
- No public access (authenticated requests only)

## Agent Architecture
**ADK Orchestrator**: Central orchestration service managing the workflow
**Core Agents**:
- **core_analyzer**: Processes raw race data and performs initial analysis
- **insight_hunter**: Identifies patterns and anomalies in the analyzed data
- **historian**: Provides historical context from previous races
- **arbiter**: Synthesizes reports from other agents into a coherent narrative
- **visualizer**: Creates visual charts and graphs on demand
- **scribe**: Generates professional engineering reports
- **publicist**: Creates social media content
- **ui_portal**: User-facing interface for uploading data and viewing results

**Agent Communication**: 
- HTTP APIs between agents
- Cloud Pub/Sub for event-driven communication
- Cloud Storage for sharing artifacts

## Local Development
```bash
# Start local development environment with emulators
docker-compose up

# Run end-to-end tests
python run_local_e2e.py
```

## Testing
**Framework**: pytest
**Test Location**: tests/ directory
**Run Command**:
```bash
python -m pytest tests/
```