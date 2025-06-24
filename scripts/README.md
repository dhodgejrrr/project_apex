# Project Apex Deployment Scripts

This directory contains scripts for deploying Project Apex agents to Google Cloud Run.

## Prerequisites

1. Install the [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
2. Authenticate with Google Cloud:
   ```bash
   gcloud auth login
   gcloud auth configure-docker
   ```
3. Set your GCP project:
   ```bash
   gcloud config set project YOUR_PROJECT_ID
   ```

## Environment Variables

Set these environment variables before running the deployment script:

```bash
export GCP_PROJECT="your-project-id"
export GCP_REGION="us-central1"  # or your preferred region
export AR_REPO="apex-agent-repo"  # Artifact Registry repository name
export SA_EMAIL="your-service-account@your-project.iam.gserviceaccount.com"
```

## Usage

### Deploy All Agents

```bash
./scripts/deploy_agents.sh
```

### Deploy Specific Agents

```bash
# Deploy one agent
./scripts/deploy_agents.sh core_analyzer

# Deploy multiple agents
./scripts/deploy_agents.sh core_analyzer ui_portal visualizer
```

### Available Agents

- adk_orchestrator
- arbiter
- core_analyzer
- historian
- insight_hunter
- publicist
- scribe
- ui_portal
- visualizer

## Customizing Deployment

You can customize the deployment by modifying these variables in the script:

- `MEMORY`: Memory allocation (default: 2Gi)
- `CPU`: CPU allocation (default: 2)
- `TIMEOUT`: Request timeout in seconds (default: 900)

## Notes

- The script builds and deploys each agent in sequence.
- Each agent is deployed as a separate Cloud Run service with the naming convention `{agent_name}-service`.
- The service account specified in `SA_EMAIL` must have the necessary permissions to deploy to Cloud Run and push to Artifact Registry.
