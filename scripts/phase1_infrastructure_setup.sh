#!/bin/bash

# Phase 1: Infrastructure Foundation Setup for Project Apex
# This script implements Phase 1 of the production deployment plan

set -e  # Exit on any error

# Configuration
PROJECT_ID="pelagic-pod-463419-m4"
REGION="us-central1"

echo "🚀 Phase 1: Infrastructure Foundation Setup"
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo ""

# Set the project
echo "Setting GCP project..."
gcloud config set project $PROJECT_ID

echo ""
echo "📢 Checkpoint 1.1: Creating Missing Pub/Sub Infrastructure"
echo "================================================="

# Create the critical orchestration topic
echo "Creating orchestration-requests topic..."
if gcloud pubsub topics create orchestration-requests --project=$PROJECT_ID 2>/dev/null; then
    echo "✅ Created orchestration-requests topic"
else
    echo "⚠️  Topic orchestration-requests may already exist"
fi

# Create downstream topics for agent communication
echo "Creating visualization-requests topic..."
if gcloud pubsub topics create visualization-requests --project=$PROJECT_ID 2>/dev/null; then
    echo "✅ Created visualization-requests topic"
else
    echo "⚠️  Topic visualization-requests may already exist"
fi

echo "Creating briefing-requests topic..."
if gcloud pubsub topics create briefing-requests --project=$PROJECT_ID 2>/dev/null; then
    echo "✅ Created briefing-requests topic"
else
    echo "⚠️  Topic briefing-requests may already exist"
fi

echo "Creating social-requests topic..."
if gcloud pubsub topics create social-requests --project=$PROJECT_ID 2>/dev/null; then
    echo "✅ Created social-requests topic"
else
    echo "⚠️  Topic social-requests may already exist"
fi

echo "Creating final-report-requests topic..."
if gcloud pubsub topics create final-report-requests --project=$PROJECT_ID 2>/dev/null; then
    echo "✅ Created final-report-requests topic"
else
    echo "⚠️  Topic final-report-requests may already exist"
fi

echo ""
echo "Verifying all topics exist..."
echo "Current Pub/Sub topics:"
gcloud pubsub topics list --project=$PROJECT_ID --format="table(name)"

echo ""
echo "💾 Checkpoint 1.2: Verifying Storage Infrastructure"
echo "================================================="

# Verify GCS buckets exist and have correct permissions
echo "Checking raw data bucket..."
if gsutil ls gs://imsa-raw-data-project-apex-v1/ >/dev/null 2>&1; then
    echo "✅ Raw data bucket exists and is accessible"
else
    echo "❌ Raw data bucket not accessible or doesn't exist"
    echo "Creating bucket..."
    gsutil mb -p $PROJECT_ID -l $REGION gs://imsa-raw-data-project-apex-v1/
fi

echo "Checking analyzed data bucket..."
if gsutil ls gs://imsa-analyzed-data-project-apex-v1/ >/dev/null 2>&1; then
    echo "✅ Analyzed data bucket exists and is accessible"
else
    echo "❌ Analyzed data bucket not accessible or doesn't exist"
    echo "Creating bucket..."
    gsutil mb -p $PROJECT_ID -l $REGION gs://imsa-analyzed-data-project-apex-v1/
fi

# Test upload permissions
echo "Testing upload permissions to raw bucket..."
if echo "test" | gsutil cp - gs://imsa-raw-data-project-apex-v1/test.txt 2>/dev/null; then
    echo "✅ Upload permissions verified"
    gsutil rm gs://imsa-raw-data-project-apex-v1/test.txt 2>/dev/null
else
    echo "❌ Upload permissions test failed"
fi

echo ""
echo "🗄️  Checkpoint 1.3: Creating BigQuery Infrastructure"
echo "================================================="

# Create BigQuery dataset for Historian agent
echo "Creating imsa_history dataset..."
if bq mk --dataset --location=US $PROJECT_ID:imsa_history 2>/dev/null; then
    echo "✅ Created imsa_history dataset"
else
    echo "⚠️  Dataset imsa_history may already exist"
fi

# Create race analysis table with proper schema
echo "Creating race_analyses table..."
if bq mk --table $PROJECT_ID:imsa_history.race_analyses \
    event_id:STRING,track:STRING,year:INTEGER,session_type:STRING,analysis_json:JSON,created_at:TIMESTAMP 2>/dev/null; then
    echo "✅ Created race_analyses table"
else
    echo "⚠️  Table race_analyses may already exist"
fi

echo "Verifying BigQuery setup..."
bq ls $PROJECT_ID:imsa_history

echo ""
echo "🎉 Phase 1 Infrastructure Setup Complete!"
echo "========================================="
echo ""
echo "✅ Pub/Sub Topics Created:"
echo "   - orchestration-requests (critical for UI Portal → ADK Orchestrator)"
echo "   - visualization-requests"
echo "   - briefing-requests"
echo "   - social-requests"
echo "   - final-report-requests"
echo ""
echo "✅ Storage Infrastructure Verified:"
echo "   - gs://imsa-raw-data-project-apex-v1/"
echo "   - gs://imsa-analyzed-data-project-apex-v1/"
echo ""
echo "✅ BigQuery Infrastructure Created:"
echo "   - Dataset: $PROJECT_ID:imsa_history"
echo "   - Table: race_analyses"
echo ""
echo "🔄 Next Steps: Proceed to Phase 2 (Service Account & IAM Configuration)"
