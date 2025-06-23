### **Comprehensive Engineering Plan: Project Apex Autonomy Upgrade**

**High-Level Goal:** Transform the existing Pub/Sub-chained pipeline into a sophisticated, ADK-orchestrated multi-agent system with a user-facing portal. Key agents will be upgraded with autonomous decision-making logic and on-demand tool-use capabilities.

---

### **Phase 0: Project Restructuring and Cleanup**

**Objective:** To prepare the codebase for the new architecture by reorganizing files and removing obsolete components.

*   **Task 0.1: [Human Engineer] Reorganize the Project Directory.**
    *   **Delete `agents/data_ingestor/`:** This agent is now obsolete. The UI Portal will be the new entry point.
    *   **Delete `adk_orchestrator.py` from the root directory.** It will be replaced by a new, more complex orchestrator service.
    *   **Create new directories:**
        *   `agents/adk_orchestrator/`
        *   `agents/ui_portal/`
        *   `agents/arbiter/`
    *   Your final `agents` directory should look like this:
        ```
        agents/
        ├── adk_orchestrator/
        ├── arbiter/
        ├── core_analyzer/
        ├── historian/
        ├── insight_hunter/
        ├── publicist/
        ├── scribe/
        ├── ui_portal/
        └── visualizer/
        ```

---

### **Phase 1: The New User-Facing & Orchestration Layers**

**Objective:** Build the new front door (`UI Portal`) and the new "conductor" (`ADK Orchestrator`).

*   **Task 1.1: [Coding AI Agent] Generate the UI Portal Agent.**
    *   **AI Prompt:**
        ```
        You are an expert Google Cloud developer creating a user interface for "Project Apex" using Streamlit.

        **Goal:**
        Create a single-file Streamlit application (`main.py`) that serves as the human interface for the entire analysis pipeline. It will have two main functions: uploading new race data and displaying the results of a finished run.

        **Detailed Requirements:**
        1.  **Libraries:** Use `streamlit`, `google-cloud-storage`, and `google-cloud-pubsub`.
        2.  **Configuration:** The script must get `PROJECT_ID`, `RAW_DATA_BUCKET`, `ANALYZED_DATA_BUCKET`, and `ORCHESTRATION_TOPIC_ID` from environment variables.
        3.  **Upload Logic:**
            a. The main page should present a form with file uploaders for a race CSV and a pit data JSON.
            b. On submission, it must:
                i. Generate a unique `run_id` by combining the CSV's basename with a few random characters (e.g., `2025_mido_race-a4f1c8`).
                ii. Upload both files to the `RAW_DATA_BUCKET` under a directory named after the `run_id` (e.g., `gs://[BUCKET]/{run_id}/...`).
                iii. Publish a single message to the `ORCHESTRATION_TOPIC_ID`. The message payload MUST be a JSON object containing the `run_id` and the full GCS paths to the two uploaded files.
                iv. Use `st.session_state` and `st.query_params` to redirect the user to the results page for the newly created `run_id`.
        4.  **Results Page Logic:**
            a. The page should be triggered if `run_id` is in the URL's query parameters.
            b. It should display the `run_id` in the title.
            c. It must periodically check the `ANALYZED_DATA_BUCKET` for the final output artifacts within the `{run_id}` folder.
            d. If artifacts are found, it should display them:
                - Render any `.png` files found in the `visuals/` subfolder.
                - Provide a download link for the PDF report found in the `reports/` subfolder and embed it in an iframe.
                - Display the text of any generated tweets from the `social/` subfolder in text areas.
            e. If no artifacts are found, it should display a "Processing..." message and a refresh button.
        5.  **GCP Clients:** Use `@st.cache_resource` to initialize and cache the `storage.Client` and `pubsub_v1.PublisherClient` to prevent re-authentication on every user interaction.
        6.  Also generate a `Dockerfile` to containerize the Streamlit app and a `requirements.txt` file.
        ```
*   **Task 1.2: [Coding AI Agent] Generate the ADK Orchestrator Agent.**
    *   **AI Prompt:**
        ```
        You are an expert AI engineer using the Google Agent Development Kit (ADK). Your task is to write the central orchestrator (`main.py`) for the "Project Apex" multi-agent system.

        **Goal:**
        This agent will be a long-running Cloud Run service triggered by Pub/Sub. It will manage the entire analysis workflow by calling other agents via authenticated HTTP requests, managing a state dictionary, and executing steps in parallel according to a defined dependency graph.

        **Detailed Requirements:**
        1.  **Framework:** Use Flask to receive the initial Pub/Sub trigger and the ADK for workflow execution.
        2.  **Trigger:** The Flask app should have a `/` route that accepts POST requests from a Pub/Sub push subscription. It should decode the message to get the initial state: `{"run_id": "...", "csv_gcs_path": "...", "pit_gcs_path": "..."}`.
        3.  **State Management:** The `state` dictionary is the single source of truth and will be passed between steps.
        4.  **Secure Agent Invocation:** Create a helper function `invoke_cloud_run(service_url, payload)` that:
            a. Fetches an OIDC identity token for the target `service_url` using `google.auth.fetch_id_token`.
            b. Makes a `requests.post()` call with the JSON `payload` and the `Authorization: Bearer {token}` header.
            c. Includes a long timeout (e.g., 900 seconds) to accommodate long-running analysis tasks.
            d. Returns the JSON response from the invoked service.
        5.  **ADK Step Definitions:** Define the following functions using `@adk.step`:
            *   `analyze_data(state)`: Calls the `CoreAnalyzer` service, passing the `run_id` and file paths. Updates `state['analysis_path']` with the result.
            *   `find_tactical_insights(state)`: Calls the `InsightHunter`. Updates `state['insights_path']`.
            *   `find_historical_insights(state)`: Calls the `Historian`. Updates `state['historical_path']`.
            *   `run_arbiter(state)`: Calls the new `Arbiter` agent. Updates `state['briefing_path']`.
            *   `generate_outputs(state)`: This will be a placeholder step that simply returns the state, as the final agents will be called by the Arbiter.
        6.  **ADK Graph Definition:** In the `main()` function, after receiving the Pub/Sub message, construct the `execution.Graph`:
            *   `step1_analyze = execution.Executable(analyze_data)`
            *   `step2a_tactical = execution.Executable(find_tactical_insights, depends_on=[step1_analyze])`
            *   `step2b_historical = execution.Executable(find_historical_insights, depends_on=[step1_analyze])`
            *   `step3_arbiter = execution.Executable(run_arbiter, depends_on=[step2a_tactical, step2b_historical])`
            *   `step4_outputs = execution.Executable(generate_outputs, depends_on=[step3_arbiter])`
        7.  **Execution:** Call `execution.execute(graph, initial_state)` to run the workflow.
        8.  **Configuration:** All service URLs (`CORE_ANALYZER_URL`, etc.) must be read from environment variables.
        9.  Provide a `Dockerfile` and `requirements.txt` (`Flask`, `gunicorn`, `requests`, `google-auth`, `adk`).
        ```

---

### **Phase 2: Refactor Core Agents for HTTP Invocation**

**Objective:** Convert agents from Pub/Sub subscribers to simple HTTP services that can be called directly by the ADK Orchestrator.

*   **Task 2.1: [Coding AI Agent] Refactor `CoreAnalyzer`.**
    *   **AI Prompt:**
        ```
        You are an expert Google Cloud developer. Your task is to refactor the existing `CoreAnalyzer` agent's `main.py`.

        **Current Behavior:** The agent is triggered by a Pub/Sub push subscription, decodes a Base64 message, and publishes its result to another Pub/Sub topic.
        **New Behavior:** The agent must be a simple HTTP service that receives a direct JSON payload, performs its task, and returns a JSON response. It should no longer interact with Pub/Sub at all.

        **Detailed Refactoring Steps:**
        1.  Remove all `google-cloud-pubsub` imports and client initialization code.
        2.  Modify the main `/` Flask route.
        3.  The request payload will now be a simple JSON object sent by the ADK orchestrator: `{"run_id": "...", "csv_path": "...", "pit_json_path": "..."}`.
        4.  The logic to download files from GCS and run `IMSADataAnalyzer` remains the same.
        5.  **CRITICAL:** The output filename must now be based on the `run_id` from the payload, NOT the basename of the original file. The output path in GCS should be `gs://[BUCKET]/{run_id}/[run_id]_results_enhanced.json`. This ensures all artifacts for a single run are grouped together.
        6.  After uploading the result to GCS, the function must `return jsonify({"analysis_path": "gs://..."})` with the path to the created file.
        7.  Update the `requirements.txt` to remove `google-cloud-pubsub`.
        ```
*   **Task 2.2: [Human Engineer] Apply Refactoring Pattern.**
    *   Use the AI-generated `CoreAnalyzer` as a template to manually refactor the following agents with the same pattern (receive direct JSON, return JSON, no Pub/Sub):
        *   `Historian`
        *   `InsightHunter`
        *   `Scribe`
        *   `Publicist`
        *   `Visualizer`

---

### **Phase 3: Implement Agent Toolboxes**

**Objective:** Deconstruct monolithic agents into on-demand "toolbox" services, which is the foundation for autonomous tool use.

*   **Task 3.1: [Coding AI Agent] Convert `CoreAnalyzer` to a Toolbox.**
    *   **AI Prompt:**
        ```
        You are an expert API developer. Your task is to refactor the `CoreAnalyzer`'s `main.py` into a "Toolbox" service.

        **Goal:**
        Keep the primary `/` endpoint for initial analysis, but expose the granular, deep-dive functions from `imsa_analyzer.py` as separate, callable API endpoints.

        **Detailed Requirements:**
        1.  Implement an in-memory Python dictionary to serve as a cache: `ANALYZER_CACHE: Dict[str, IMSADataAnalyzer] = {}`.
        2.  Create a helper function `get_analyzer(run_id)` that first checks the cache for an analyzer instance. If not found, it should reconstruct the instance by downloading the source files from GCS based on the `run_id` (e.g., `gs://.../{run_id}/{run_id}.csv`) and then add it to the cache.
        3.  The existing `/` endpoint should now use this `get_analyzer` function to create and cache the instance during the initial run.
        4.  Create the following new "tool" endpoints:
            *   **`POST /tools/driver_deltas`**:
                - Expects a JSON payload: `{"run_id": "...", "car_number": "optional_string"}`.
                - Uses `get_analyzer` to get the relevant instance.
                - Calls `analyzer.get_driver_deltas_by_car()`.
                - If `car_number` is provided, filters the results for that car.
                - Returns the resulting data as a JSON response.
            *   **`POST /tools/trend_analysis`**: (This is a new feature for the Historian to use)
                - Expects: `{"track": "...", "session": "...", "manufacturer": "...", "num_years": 5}`.
                - This tool will query BigQuery to get the fastest lap time for the specified manufacturer at that track for the last `num_years`.
                - It should return a JSON object with the trend data: `{"trend_data": [{"year": YYYY, "lap_time": "M:SS.mmm"}, ...]}`.
        ```
*   **Task 3.2: [Coding AI Agent] Convert `Visualizer` to a Toolbox.**
    *   **AI Prompt:**
        ```
        You are an expert API developer. Refactor the `Visualizer` agent's `main.py` into a "Toolbox" service.

        **Goal:**
        Instead of a single endpoint that generates all charts, create a separate API endpoint for each plotting function.

        **Detailed Requirements:**
        1.  Remove the main `/` route and the `generate_all_visuals` function.
        2.  Create the following new "tool" endpoints:
            *   **`POST /plot/pit_times`**:
                - Expects: `{"analysis_path": "gs://..."}`.
                - Downloads the analysis JSON, calls `plot_pit_stationary_times`, uploads the resulting PNG to GCS under the correct `run_id` subfolder, and returns `jsonify({"image_gcs_path": "gs://..."})`.
            *   **`POST /plot/consistency`**: Does the same for the `plot_driver_consistency` function.
            *   **`POST /plot/stint_falloff`**:
                - Expects: `{"analysis_path": "gs://...", "car_number": "..."}`.
                - Calls the `plot_stint_pace_falloff` function with the specified car number.
                - Uploads the result and returns the GCS path.
        ```

---

### **Phase 4: Implement Autonomous Logic in Agents**

**Objective:** Upgrade the "thinking" agents to use their new tools and decision-making capabilities.

*   **Task 4.1: [Coding AI Agent] Implement Autonomous `InsightHunter`.**
    *   **AI Prompt:**
        ```
        You are an AI engineer implementing autonomous behavior in the `InsightHunter` agent.

        **Goal:**
        Refactor the agent to use a two-step "Plan and Execute" logic. It will first identify anomalies from high-level data and then autonomously call tools on the `CoreAnalyzer` service to get detailed root-cause data.

        **Detailed Requirements:**
        1.  Create a new main function `generate_insights_autonomously(analysis_data)`.
        2.  **Step 1 (Plan):** Craft a detailed prompt for Gemini. This prompt should provide a summary of the analysis data and ask the LLM to return a JSON "investigation plan" listing findings and the specific `CoreAnalyzer` tool needed to investigate each one (e.g., `{"finding": "...", "required_tool": {"name": "driver_deltas", "parameters": ...}}`).
        3.  **Step 2 (Execute):**
            a. Parse the JSON plan from the LLM.
            b. For each task in the plan, make an authenticated HTTP call to the specified tool endpoint on the `CoreAnalyzer` service (e.g., `POST /tools/driver_deltas`).
            c. **Step 3 (Synthesize):** For each task, create a *second* prompt for Gemini. This prompt will include the initial finding and the detailed data returned from the tool. Ask the LLM to write a final, rich insight that combines both pieces of information.
        4.  Aggregate these final, synthesized insights into a list.
        5.  The main Flask handler should now call this autonomous function and return the GCS path to the final insights file.
        ```

*   **Task 4.2: [Coding AI Agent] Implement the `Arbiter` and Autonomous `Publicist`.**
    *   **AI Prompt:**
        ```
        You are an AI engineer building two new agents: `Arbiter` and an autonomous `Publicist`.

        **Part 1: The `Arbiter` Agent**
        1.  Create a new agent in `agents/arbiter/`.
        2.  Its `main.py` should be an HTTP service that receives the GCS paths to the tactical `insights.json` and the `historical_insights.json`.
        3.  It will download both files.
        4.  Craft a master prompt for Gemini. The prompt will instruct the LLM to act as a Chief Strategist, synthesize both reports, resolve conflicts, and produce a "Final Briefing" JSON object containing an `executive_summary` and a list of `marketing_angles`.
        5.  The agent saves this briefing to GCS and returns its path.

        **Part 2: The Autonomous `Publicist` Agent**
        1.  Refactor the existing `Publicist` agent.
        2.  Its main function should now receive the `Final Briefing` JSON from the Arbiter.
        3.  Implement a **self-correction loop**:
            a. Craft a prompt that uses the `marketing_angles` to generate a package of draft social media posts. The prompt should ask the LLM to also decide if a visual is needed for each post and, if so, which tool to call (`generate_pit_time_chart`, etc.). The response should be a structured JSON plan.
            b. Create a `invoke_visualizer_tool` helper function that makes an authenticated call to the `Visualizer` toolbox service.
            c. Process the LLM's plan: for each planned post, call the specified visualizer tool (if any) and get back the image URL.
            d. Assemble the final package: `[{"text": "...", "image_url": "..."}, ...]`.
            e. **Critique Step:** Craft a second "Reviewer" prompt. Feed the generated package to Gemini again, asking it to critique the work from the perspective of a demanding marketing manager.
            f. If the critique's `approved` field is `false`, loop back to step (a), adding the critique as feedback to the prompt.
        4. The agent saves the final, approved social media package to GCS.
        ```

---

### **Phase 5: Deployment and Final Integration**

**Objective:** Deploy all the new and refactored services and configure them to work together.

*   **Task 5.1: [Human Engineer] Deploy All Services.**
    *   For each agent (`ui_portal`, `adk_orchestrator`, `core_analyzer`, `insight_hunter`, `historian`, `visualizer`, `arbiter`, `scribe`, `publicist`):
        1.  Build the Docker image using `gcloud builds submit`.
        2.  Deploy to Cloud Run using `gcloud run deploy`.
        3.  **CRITICAL:** For the `adk_orchestrator` deployment, pass the service URLs of all other agents as environment variables (e.g., `--update-env-vars="CORE_ANALYZER_URL=https://..."`).
        4.  Ensure all services are deployed with `--no-allow-unauthenticated` and use the same service account.
        5.  Configure the IAM permissions to allow the service account to invoke other Cloud Run services (`roles/run.invoker`).

*   **Task 5.2: [Human Engineer] Configure Pub/Sub Trigger.**
    *   Create the `orchestration-requests` Pub/Sub topic.
    *   Create a push subscription targeting the `adk-orchestrator` service.

### **Phase 6: Final Validation**

*   **Task 6.1: [Human Engineer] End-to-End Test.**
    1.  Navigate to the `ui_portal`'s URL.
    2.  Upload the sample `race.csv` and `pit.json` files.
    3.  Monitor the Cloud Logging for the `adk_orchestrator` service. You should see it progressing through the steps of the graph.
    4.  Check the logs for the individual agents to see the tool calls and autonomous loops executing.
    5.  Once the workflow completes, refresh the UI Portal results page and verify that all artifacts (PDF, images, tweets) are displayed correctly.

This detailed plan provides the necessary structure and context for an AI engineer to systematically implement the entire advanced, autonomous architecture.