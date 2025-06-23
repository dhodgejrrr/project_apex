# Project Apex: Autonomous Motorsport Intelligence Platform

**Project Apex is an autonomous, multi-agent system built on Google's Agent Development Kit (ADK) that leverages Vertex AI to transform raw motorsport data into a real-time competitive advantage and a brand-building asset.**

This system moves beyond simple data processing, creating a collaborative team of AI agents that can analyze, investigate, synthesize, and generate content, demonstrating a sophisticated, modern agentic architecture.

[![Video Pitch](https://img.shields.io/badge/Video-Watch%20Our%20Pitch-red?style=for-the-badge&logo=youtube)](https://www.youtube.com/watch?v=your-video-id)
[![UI Portal](https://img.shields.io/badge/Live%20Demo-UI%20Portal-blue?style=for-the-badge&logo=google-cloud)](https://your-ui-portal-url.a.run.app)

---

## Inspiration

In the high-stakes world of endurance racing, teams are inundated with thousands of data points every minute. We saw a critical gap between this raw telemetry and the actionable intelligence needed by two key personas: the **Race Strategist** making split-second decisions on the pit wall, and the **Marketing Manager** needing to tell a compelling story to fans.

Our inspiration was to build a system that bridges this gap—an autonomous team of AI agents that could not only analyze the data but also understand its context, synthesize a narrative, and craft outputs tailored to both strategy and fan engagement.

## What it does

Our system unlocks the winning story hidden in motorsport telemetry. It works in three autonomous stages:

1.  **Autonomous Investigation:** Analyst agents ingest raw race data. If they spot an anomaly, they have the autonomy to **go back and request a "drill-down" report** from our `CoreAnalyzer` toolbox to investigate the root cause.
2.  **Narrative Synthesis:** Our `Arbiter` agent acts like a Chief Strategist. It receives reports from the `InsightHunter` and `Historian` and synthesizes them, resolving conflicts to create a single, coherent master narrative for the race.
3.  **Tailored Content Generation:** The master narrative is passed to our output agents. The `Scribe` generates a professional engineering report for the pit wall, while our `Publicist` drafts engaging social media posts, even **calling on a `Visualizer` tool on-demand** to create charts for the most impactful insights.

### Product Flow

This diagram illustrates the journey from a user's perspective, focusing on the value delivered to each persona.

![Product Flow Diagram](mermaid_product.svg)

## How We Built It

Project Apex is built entirely on Google Cloud, architected around the principles of the Agent Development Kit (ADK) to create a truly orchestrated and autonomous system.

*   **Orchestration:** The heart of the system is our **ADK Orchestrator**, a Cloud Run service that manages the entire workflow using an `execution.Graph` to define the sequence and enable parallel execution.
*   **Agent Architecture:** Each agent is a specialized, containerized **Cloud Run** service. This serverless design ensures scalability and cost-efficiency.
*   **Autonomous Tool Use:** We implemented an "agent-as-a-toolbox" pattern. The `CoreAnalyzer` and `Visualizer` are services with specific API endpoints that other agents can call on-demand.
*   **Technology Stack:**
    *   **Agent Development Kit (ADK):** For defining and executing the multi-agent workflow.
    *   **Vertex AI Gemini API:** Powers the reasoning, synthesis, and content generation for our most advanced agents.
    *   **Cloud Run & Cloud Build:** To host and build our containerized, serverless agents.
    *   **Streamlit:** Powers our user-facing `UI Portal`.
    *   **BigQuery:** Serves as the long-term memory for our `Historian` agent.
    *   **Cloud Storage & Pub/Sub:** For artifact storage and event-driven triggers.

### Technical Architecture

This diagram shows how our ADK Orchestrator manages the flow of information and tool calls between agents.

![Technical Flow Diagram](mermaid_flow.svg)

## Testing the Application

### Method 1: Live Cloud Application (Recommended)

1.  Navigate to our deployed **[UI Portal](https://your-ui-portal-url.a.run.app)**.
2.  Upload the sample `glen_race.csv` and `glen_pit.json` files from the `agents/test_data` directory.
3.  Click **"Start Analysis"**.
4.  You will be redirected to a status page. The full pipeline takes 3-5 minutes to complete.
5.  Refresh the page to see the final artifacts appear, including the PDF report, generated visuals, and social media content.

### Method 2: Local End-to-End Pipeline

For development and testing, the entire pipeline can be run locally without requiring any cloud services.

1.  **Prerequisites:** Python 3.11+ and all dependencies installed.
    ```bash
    # It's recommended to use a virtual environment
    pip install -r requirements.txt 
    ```
2.  **Run the local pipeline script:** This uses a test harness that mocks all GCP and Vertex AI interactions.
    ```bash
    python full_pipeline_local.py --out_dir ./local_run_output
    ```
3.  **Verify Outputs:** Check the `local_run_output/` directory for the generated PDF, PNGs, and JSON files.

## What's Next for Project Apex

We are just scratching the surface of what an autonomous agent team can do. Our next steps include:
1.  **Real-Time Ingestion:** Transition from file uploads to a live WebSocket stream from the timing & scoring feed for in-race alerts.
2.  **Human-in-the-Loop:** Add a UI approval step for social media posts before they are published via the X/Twitter API.
3.  **Predictive Strategy Tools:** Train an AutoML model on our BigQuery data to provide predictive insights, such as the probability of a podium finish for different pit strategies.
4.  **Unified Data Model:** Create an `Adapter` agent to transform data from other series (F1, WEC) into a universal format, making our analysis platform truly versatile and scalable.