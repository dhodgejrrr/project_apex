# Local Deployment and Testing Guide for Project Apex

This guide explains how to deploy Project Apex agents locally using Docker and run the end-to-end tests.

## Prerequisites

1. **Docker and Docker Compose**
   - Docker Desktop must be installed and running
   - Docker Compose v2 is required

2. **Google Cloud SDK**
   - Application Default Credentials must be set up
   - Run `gcloud auth application-default login` if not already done

3. **Environment Variables**
   - Set `GOOGLE_CLOUD_PROJECT` environment variable
   - Create a `.env` file in the project root or export the variable:
     ```bash
     export GOOGLE_CLOUD_PROJECT=your-project-id
     ```

4. **Python Dependencies**
   - Install required Python packages:
     ```bash
     pip install -r requirements.txt
     ```

## Deployment and Testing Process

### Method 1: Using the Automated Script (Recommended)

The `run_local_e2e.py` script handles the entire process automatically:

1. **Run the script**:
   ```bash
   python run_local_e2e.py
   ```

2. **What the script does**:
   - Checks prerequisites (Docker, GCP credentials)
   - Tears down any existing Docker services
   - Builds and starts all Docker containers defined in docker-compose.yml
   - Sets up GCS and Pub/Sub emulators
   - Waits for all services to be healthy
   - Uploads test data to the GCS emulator
   - Triggers the ADK Orchestrator to start the pipeline
   - Downloads and verifies all artifacts
   - Tears down Docker services when complete

3. **Output location**:
   - All test outputs are saved to the `local_e2e_outputs` directory
   - Each run creates a timestamped folder (e.g., `local-run-1751006004`)

### Method 2: Manual Deployment and Testing

If you need more control over the process, you can deploy and test manually:

1. **Start Docker services**:
   ```bash
   docker compose up --build -d
   ```

2. **Verify services are running**:
   ```bash
   docker ps
   ```

3. **Check service health**:
   ```bash
   curl http://localhost:8087/health  # ADK Orchestrator
   curl http://localhost:8080/health  # Core Analyzer
   # Check other services as needed
   ```

4. **Upload test data manually**:
   - Use the GCS emulator at http://localhost:4443
   - Upload files from `agents/test_data` directory

5. **Trigger the pipeline**:
   ```bash
   # Create a JSON payload with the test data paths
   curl -X POST http://localhost:8087/ \
     -H "Content-Type: application/json" \
     -d '{
       "message": {
         "data": "BASE64_ENCODED_DATA",
         "messageId": "local-test-message-1"
       },
       "subscription": "projects/local-dev/subscriptions/local-trigger"
     }'
   ```

6. **Monitor the process**:
   ```bash
   # Check logs for specific services
   docker logs project_apex-adk-orchestrator-1
   docker logs project_apex-core-analyzer-1
   ```

7. **Shut down services when done**:
   ```bash
   docker compose down -v --remove-orphans
   ```

## Troubleshooting

### Common Issues

1. **Docker services not starting**
   - Check Docker logs: `docker logs project_apex-[service-name]-1`
   - Ensure Docker has enough resources allocated (memory, CPU)
   - Verify no port conflicts with existing services

2. **Authentication errors**
   - Ensure Application Default Credentials are set up correctly
   - Check the mounted credentials volume in docker-compose.yml

3. **Service communication issues**
   - Verify all services are on the same Docker network
   - Check environment variables for service URLs

4. **Emulator issues**
   - GCS emulator: Ensure it's accessible at http://localhost:4443
   - Pub/Sub emulator: Verify it's running on port 8085

### Debugging Tips

1. **Check container logs**:
   ```bash
   docker logs project_apex-adk-orchestrator-1
   ```

2. **Inspect Docker network**:
   ```bash
   docker network inspect project_apex_apex-net
   ```

3. **Access service directly**:
   ```bash
   # Example: Test the Core Analyzer directly
   curl -X POST http://localhost:8080/analyze \
     -H "Content-Type: application/json" \
     -d '{"csv_gcs_path": "gs://imsa-analyzed-data-project-apex-v1/test/glen_race.csv"}'
   ```

4. **Restart specific services**:
   ```bash
   docker compose restart core-analyzer
   ```

## Agent-Specific Information

### ADK Orchestrator
- **Port**: 8087
- **Health Check**: `/health`
- **Dependencies**: All other agents

### Core Analyzer
- **Port**: 8080
- **Health Check**: `/health`
- **API Endpoint**: `/analyze`

### Insight Hunter
- **Port**: 8081
- **Health Check**: `/`

### Historian
- **Port**: 8082
- **Health Check**: `/`

### Visualizer
- **Port**: 8083
- **Health Check**: `/`

### Scribe
- **Port**: 8084
- **Health Check**: `/`

### Publicist
- **Port**: 8086
- **Health Check**: `/`

### Arbiter
- **Port**: 8088
- **Health Check**: `/`

### Tool Registry
- **Port**: 8090
- **Health Check**: `/`

### Performance Monitor
- **Port**: 8091
- **Health Check**: `/`

### Task Router
- **Port**: 8092
- **Health Check**: `/`