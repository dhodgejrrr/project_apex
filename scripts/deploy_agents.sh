#!/bin/bash

# Exit on error
set -e

# Default values
GCP_PROJECT="${GCP_PROJECT:-$(gcloud config get-value project)}"
GCP_REGION="${GCP_REGION:-us-central1}"  # Change to your preferred region
AR_REPO="${AR_REPO:-apex-agent-repo}"      # Artifact Registry repository name
SA_EMAIL="apex-agent-runner@${GCP_PROJECT}.iam.gserviceaccount.com"  # Service account email
MEMORY="2Gi"
CPU="2"
TIMEOUT="900"

# List of agents to deploy
ALL_AGENTS=(
    "arbiter"
    "core_analyzer"
    "historian"
    "insight_hunter"
    "publicist"
    "scribe"
    "ui_portal"
    "visualizer"
)

# Function to convert agent name to valid service name
# Replaces underscores with hyphens and ensures it's a valid Cloud Run service name
get_service_name() {
    local name=$1
    # Convert to lowercase and replace underscores with hyphens
    echo "${name//_/-}-service" | tr '[:upper:]' '[:lower:]'
}

# Function to deploy a single agent
deploy_agent() {
    local agent_name=$1
    local image_tag="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${AR_REPO}/${agent_name}:latest"
    local service_name=$(get_service_name "${agent_name}")
    
    echo "🚀 Deploying ${agent_name}..."
    
    # Deploy to Cloud Run
    echo "☁️  Deploying ${service_name} to Cloud Run..."
    gcloud run deploy "${service_name}" \
        --image "${image_tag}" \
        --service-account="${SA_EMAIL}" \
        --no-allow-unauthenticated \
        --region="${GCP_REGION}" \
        --platform=managed \
        --memory="${MEMORY}" \
        --cpu="${CPU}" \
        --timeout="${TIMEOUT}" \
        --set-env-vars="AGENT_NAME=${agent_name}" \
        --quiet
    
    echo "✅ Successfully deployed ${service_name}"
    echo ""
}

# Main script
main() {
    # Check if GCP project is set
    if [ -z "${GCP_PROJECT}" ]; then
        echo "❌ Error: GCP project not set. Please set GCP_PROJECT environment variable or configure gcloud."
        exit 1
    fi
    
    # Determine which agents to deploy
    local agents_to_deploy=("$@")
    
    # If no agents specified, deploy all
    if [ ${#agents_to_deploy[@]} -eq 0 ]; then
        echo "ℹ️  No agents specified. Deploying all agents."
        agents_to_deploy=("${ALL_AGENTS[@]}")
    fi
    
    # Deploy each agent
    for agent in "${agents_to_deploy[@]}"; do
        if [[ " ${ALL_AGENTS[*]} " =~ " ${agent} " ]]; then
            deploy_agent "${agent}"
        else
            echo "⚠️  Warning: Unknown agent '${agent}'. Skipping..."
        fi
    done
    
    echo "✨ All deployments completed successfully!"
}

# Run the main function with all arguments
main "$@"
