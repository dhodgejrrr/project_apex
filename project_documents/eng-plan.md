This plan is structured in phases. Each task is explicitly assigned to either the **[Human Engineer]** (for infrastructure, deployment, and oversight) or the **[Coding AI Agent]** (for generating application code). The prompts for the AI are designed to be self-contained and ready for use.

---

## **Project Apex: Detailed Engineering & Implementation Plan**

### **Phase 0: Google Cloud Project Foundation**

This phase establishes the cloud infrastructure. All tasks are for the **[Human Engineer]**.

*   **Task 0.1: Project Setup**
    1.  Create a new Google Cloud Project or select an existing one. Note the **Project ID**.
    2.  Install and initialize the `gcloud` CLI, authenticating with your user account.
    3.  Set your project configuration: `gcloud config set project [YOUR_PROJECT_ID]`
    4.  Set a default region: `gcloud config set compute/region us-central1`

*   **Task 0.2: Enable APIs**
    *   Run the following command to enable all necessary service APIs:
        ```bash
        gcloud services enable \
          run.googleapis.com \
          cloudbuild.googleapis.com \
          iam.googleapis.com \
          storage.googleapis.com \
          pubsub.googleapis.com \
          artifactregistry.googleapis.com \
          cloudfunctions.googleapis.com \
          bigquery.googleapis.com \
          aiplatform.googleapis.com
        ```

*   **Task 0.3: Create Service Account**
    *   Create a dedicated service account for the agents to use.
        ```bash
        gcloud iam service-accounts create apex-agent-runner \
          --display-name="Service Account for Project Apex Agents"
        ```
    *   Assign necessary IAM roles to the service account. This gives it permissions to interact with other Google Cloud services.
        ```bash
        PROJECT_ID=$(gcloud config get-value project)
        SA_EMAIL="apex-agent-runner@${PROJECT_ID}.iam.gserviceaccount.com"

        # Role for running Cloud Run & Cloud Functions
        gcloud projects add-iam-policy-binding $PROJECT_ID \
          --member="serviceAccount:${SA_EMAIL}" \
          --role="roles/run.invoker"
        gcloud projects add-iam-policy-binding $PROJECT_ID \
          --member="serviceAccount:${SA_EMAIL}" \
          --role="roles/cloudfunctions.invoker"

        # Role for accessing Cloud Storage
        gcloud projects add-iam-policy-binding $PROJECT_ID \
          --member="serviceAccount:${SA_EMAIL}" \
          --role="roles/storage.objectAdmin"

        # Role for publishing to Pub/Sub
        gcloud projects add-iam-policy-binding $PROJECT_ID \
          --member="serviceAccount:${SA_EMAIL}" \
          --role="roles/pubsub.publisher"

        # Role for reading/writing to BigQuery
        gcloud projects add-iam-policy-binding $PROJECT_ID \
          --member="serviceAccount:${SA_EMAIL}" \
          --role="roles/bigquery.dataEditor"
        gcloud projects add-iam-policy-binding $PROJECT_ID \
          --member="serviceAccount:${SA_EMAIL}" \
          --role="roles/bigquery.jobUser"

        # Role for calling Vertex AI
        gcloud projects add-iam-policy-binding $PROJECT_ID \
          --member="serviceAccount:${SA_EMAIL}" \
          --role="roles/aiplatform.user"
        ```

*   **Task 0.4: Create Infrastructure Resources**
    *   **Cloud Storage Buckets:**
        ```bash
        # Use a unique suffix for your bucket names
        UNIQUE_SUFFIX="[YOUR_TEAM_NAME_OR_RANDOM_CHARS]"
        
        gsutil mb -p [YOUR_PROJECT_ID] -l US-CENTRAL1 gs://imsa-raw-data-${UNIQUE_SUFFIX}
        gsutil mb -p [YOUR_PROJECT_ID] -l US-CENTRAL1 gs://imsa-analyzed-data-${UNIQUE_SUFFIX}
        ```
    *   **Pub/Sub Topics:**
        ```bash
        gcloud pubsub topics create analysis-requests
        gcloud pubsub topics create insight-requests
        gcloud pubsub topics create visualization-requests
        ```
    *   **Artifact Registry:**
        ```bash
        gcloud artifacts repositories create apex-agent-repo \
          --repository-format=docker \
          --location=us-central1 \
          --description="Docker repository for Project Apex agents"
        ```
    *   **BigQuery Dataset & Table:**
        ```bash
        bq --location=US-CENTRAL1 mk --dataset [YOUR_PROJECT_ID]:imsa_history

        bq mk --table [YOUR_PROJECT_ID]:imsa_history.race_analyses \
           event_id:STRING,track:STRING,year:INTEGER,session_type:STRING,analysis_json:JSON
        ```
*   **Task 0.5: Local Project Structure Setup**
    *   Create the following directory structure on your local machine:
        ```
        project-apex/
        ├── adk-orchestrator.py
        ├── agents/
        │   ├── core_analyzer/
        │   │   ├── main.py
        │   │   ├── Dockerfile
        │   │   ├── requirements.txt
        │   │   └── imsa_analyzer.py  <-- Copy your existing file here
        │   ├── data_ingestor/
        │   │   ├── main.py
        │   │   └── requirements.txt
        │   ├── insight_hunter/
        │   │   ├── main.py
        │   │   ├── Dockerfile
        │   │   └── requirements.txt
        │   ├── historian/
        │   │   ├── main.py
        │   │   ├── Dockerfile
        │   │   └── requirements.txt
        │   ├── visualizer/
        │   │   ├── main.py
        │   │   ├── Dockerfile
        │   │   └── requirements.txt
        │   ├── scribe/
        │   │   ├── main.py
        │   │   ├── Dockerfile
        │   │   ├── requirements.txt
        │   │   └── report_template.html
        │   └── publicist/
        │       ├── main.py
        │       ├── Dockerfile
        │       └── requirements.txt
        └── README.md
        ```

---

### **Phase 1: The `DataIngestor` Agent (Cloud Function)**

*   **Task 1.1: Generate Agent Code**
    *   **[Coding AI Agent]** Use the following prompt:

    **AI Prompt:**
    ```
    You are an expert Google Cloud developer. Your task is to write the complete Python code for a Google Cloud Function named `DataIngestor` for a project called "Project Apex".

    **Function Goal:**
    This function is triggered when a file is uploaded to a Cloud Storage bucket. It checks if both a CSV and a corresponding `.json` pit file exist for a given event. If both are present, it publishes a message to a Pub/Sub topic to start an analysis job.

    **Detailed Requirements:**
    1.  The function should be written in Python 3.11 for the `main.py` file.
    2.  It must be triggered by a Cloud Storage `google.storage.object.finalize` event.
    3.  The function should parse the uploaded filename. Assume filenames like `2025_mido_race.csv` or `2025_mido_race_pits.json`. The base name is `2025_mido_race`.
    4.  Upon triggering, it should determine the base name of the event from the filename (e.g., `2025_mido_race`).
    5.  It must then check for the existence of the sibling file in the same bucket.
        - If a `.csv` is uploaded, it checks if `[basename]_pits.json` exists.
        - If a `_pits.json` is uploaded, it checks if `[basename].csv` exists.
    6.  If and only if both files exist, it should publish a message to the `analysis-requests` Pub/Sub topic.
    7.  The message payload must be a JSON string containing the full GCS paths to both files: `{"csv_path": "gs://[BUCKET]/[BASENAME].csv", "pit_json_path": "gs://[BUCKET]/[BASENAME]_pits.json"}`.
    8.  The function must handle potential errors gracefully (e.g., file not found) and log informative messages.
    9.  The code should be well-commented, explaining the logic.
    10. Also, create a `requirements.txt` file listing all necessary dependencies, which are `google-cloud-storage` and `google-cloud-pubsub`.

    Provide the complete, ready-to-deploy code for `main.py` and `requirements.txt`.
    ```

*   **Task 1.2: Deploy the Cloud Function**
    *   **[Human Engineer]**
    1.  Place the generated `main.py` and `requirements.txt` into the `project-apex/agents/data_ingestor/` directory.
    2.  From within that directory, run the deployment command. Replace placeholders.
        ```bash
        # In directory: project-apex/agents/data_ingestor/
        
        gcloud functions deploy data-ingestor-function \
          --gen2 \
          --runtime=python311 \
          --region=us-central1 \
          --source=. \
          --entry-point=trigger_analysis \
          --trigger-bucket=imsa-raw-data-[YOUR_UNIQUE_SUFFIX] \
          --service-account=apex-agent-runner@$(gcloud config get-value project).iam.gserviceaccount.com
        ```

---

### **Phase 2: The `CoreAnalyzer` Agent (Cloud Run)**

*   **Task 2.1: Generate Agent Code**
    *   **[Coding AI Agent]** Use the following prompt:

    **AI Prompt:**
    ```
    You are an expert Google Cloud developer building a containerized agent for "Project Apex". Your task is to create the web server and containerization files for the `CoreAnalyzer` agent.

    **Agent Goal:**
    This agent is a Cloud Run service that listens for messages from a Pub/Sub push subscription. When it receives a message containing GCS paths for race data, it downloads the data, runs a complex analysis using a provided `imsa_analyzer.py` module, and uploads the resulting JSON report back to Cloud Storage before publishing a notification message.

    **Project Structure within the container:**
    ```
    /app/
    ├── main.py           # The Flask server you will write
    ├── imsa_analyzer.py  # The existing analysis class (I will provide its code)
    └── requirements.txt  # You will create this
    ```
    
    **`imsa_analyzer.py` Code (for context):**
    ```python
    # --- PASTE THE ENTIRE CONTENT OF imsa_analyzer.py HERE ---
    ```
    
    **Detailed Requirements for `main.py`:**
    1.  Use Python 3.11 and Flask.
    2.  Create a Flask application.
    3.  Define a single POST route at `/`. This endpoint will be the target for the Pub/Sub push subscription.
    4.  The request body will be a standard Pub/Sub push message format: `{"message": {"data": "BASE64_ENCODED_STRING", ...}}`.
    5.  Your code must:
        a. Extract the base64-encoded `data` from the JSON payload.
        b. Decode the data to get a JSON string (e.g., `{"csv_path": "...", "pit_json_path": "..."}`).
        c. Parse this JSON to get the GCS paths.
        d. Use the `google-cloud-storage` library to download the two files from their GCS paths into a temporary local directory (e.g., `/tmp/`).
        e. Instantiate the analyzer: `analyzer = IMSADataAnalyzer(local_csv_path, local_pit_json_path)`.
        f. Run the analysis: `results = analyzer.run_all_analyses()`.
        g. Determine the output filename. E.g., if the input was `2025_mido_race.csv`, the output should be `2025_mido_race_results_enhanced.json`.
        h. Save the `results` dictionary as a JSON file locally.
        i. Upload this new JSON report to the `imsa-analyzed-data-[SUFFIX]` bucket.
        j. Publish a notification message to the `insight-requests` Pub/Sub topic. The message payload should be a JSON string with the GCS path of the new report: `{"analysis_path": "gs://imsa-analyzed-data-[SUFFIX]/[OUTPUT_FILENAME]"}`.
        k. Implement robust `try...except...finally` blocks to handle errors and ensure temporary files are cleaned up.
        l. Return an HTTP `204 No Content` on success, or an appropriate error code (e.g., `500`) on failure.
    
    **Detailed Requirements for `Dockerfile`:**
    1.  Start from a slim Python 3.11 base image (e.g., `python:3.11-slim`).
    2.  Set a working directory `/app`.
    3.  Copy `requirements.txt` and install dependencies.
    4.  Copy the rest of the application code (`main.py`, `imsa_analyzer.py`).
    5.  Set the `CMD` to run the Flask application using a production-ready server like `gunicorn`. E.g., `gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app`. Use the `$PORT` environment variable.

    **Detailed Requirements for `requirements.txt`:**
    *   Include all necessary libraries: `Flask`, `gunicorn`, `pandas`, `numpy`, `google-cloud-storage`, `google-cloud-pubsub`. Specify versions if known, otherwise list the library names.

    Provide the complete, ready-to-use code for `main.py`, `Dockerfile`, and `requirements.txt`.
    ```
*   **Task 2.2: Build and Deploy the Container**
    *   **[Human Engineer]**
    1.  Place the generated `main.py`, `Dockerfile`, and `requirements.txt` into the `project-apex/agents/core_analyzer/` directory.
    2.  From within that directory, build the container image and push it to Artifact Registry.
        ```bash
        # In directory: project-apex/agents/core_analyzer/
        
        PROJECT_ID=$(gcloud config get-value project)
        REPO_NAME="apex-agent-repo"
        REGION="us-central1"

        gcloud builds submit --tag ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/core-analyzer:latest
        ```
    3.  Deploy the container to Cloud Run.
        ```bash
        # In directory: project-apex/agents/core_analyzer/
        
        gcloud run deploy core-analyzer-service \
          --image=${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/core-analyzer:latest \
          --platform=managed \
          --region=${REGION} \
          --service-account=apex-agent-runner@${PROJECT_ID}.iam.gserviceaccount.com \
          --no-allow-unauthenticated \
          --update-env-vars="ANALYZED_DATA_BUCKET=imsa-analyzed-data-[YOUR_UNIQUE_SUFFIX]"
        ```
    4.  Create a Pub/Sub push subscription that targets the new Cloud Run service.
        ```bash
        # This gives Pub/Sub permission to invoke your Cloud Run service
        gcloud run services add-iam-policy-binding core-analyzer-service \
          --member="serviceAccount:service-${PROJECT_ID}@gcp-sa-pubsub.iam.gserviceaccount.com" \
          --role="roles/run.invoker" \
          --region=${REGION} \
          --platform=managed
          
        SERVICE_URL=$(gcloud run services describe core-analyzer-service --platform=managed --region=${REGION} --format='value(status.url)')

        gcloud pubsub subscriptions create core-analyzer-sub \
          --topic=analysis-requests \
          --push-endpoint=${SERVICE_URL}/ \
          --push-auth-service-account=apex-agent-runner@${PROJECT_ID}.iam.gserviceaccount.com
        ```
        
---

### **Phase 3: The `InsightHunter` Agent (Cloud Run)**

This is the most complex new agent. The process mirrors Phase 2: AI generates code, Human deploys.

*   **Task 3.1: Generate Agent Code**
    *   **[Coding AI Agent]** Use the following detailed prompt:

    **AI Prompt:**
    ```
    You are an expert data scientist and Python developer creating the `InsightHunter` agent for "Project Apex".

    **Agent Goal:**
    This agent is a Cloud Run service that receives a GCS path to an `...enhanced.json` analysis file. It must systematically parse this large JSON, apply a series of predefined analytical heuristics to find tactical insights, structure these insights into a new JSON object, and upload it back to GCS.

    **Detailed Requirements:**
    1.  Create `main.py`, `Dockerfile`, and `requirements.txt` for a Flask-based Cloud Run service, just like the `CoreAnalyzer` agent.
    2.  The `main.py` will have a `/` POST route that receives a Pub/Sub push message containing `{"analysis_path": "gs://..."}`.
    3.  The core logic will download the specified `...enhanced.json` file and process it.
    4.  Create a main function, `find_insights(data)`, that takes the loaded JSON data as a Python dictionary. This function should orchestrate calls to smaller, specialized functions, one for each insight category.
    5.  Implement the following insight-finding functions. Each function should return a list of dictionary objects, where each dictionary represents one insight. If no insight is found, return an empty list.

    **Insight Function Specifications:**

    *   `find_pit_stop_insights(strategy_data)`:
        *   **Input:** The list from the `race_strategy_by_car` key.
        *   **Logic 1 (Pit Delta Outlier):** For each car, if `total_pit_stops > 1`, calculate the average `total_pit_lane_time` (by parsing the formatted string). Iterate through `pit_stop_details`. If a stop's time is > 1.5x the average, create an insight object: `{"category": "Pit Stop Intelligence", "type": "Pit Delta Outlier", "car_number": "...", "details": "Stop #X was Ys slower than average."}`.
        *   **Logic 2 (Driver Change Cost):** For each car with `total_driver_changes > 0`, calculate the average stationary time for stops with `driver_change: true` vs. `driver_change: false`. Create insight: `{"category": "Pit Stop Intelligence", "type": "Driver Change Cost", "car_number": "...", "details": "Driver changes cost an average of Zs more in stationary time."}`.
        *   **Logic 3 (Stationary Time Champion):** Parse `avg_pit_stationary_time` from all cars in `enhanced_strategy_analysis`. Find the car with the minimum time. Create insight: `{"category": "Pit Stop Intelligence", "type": "Stationary Time Champion", "car_number": "...", "details": "Fastest crew on pit road with an average stationary time of Xs."}`.

    *   `find_performance_insights(data)`:
        *   **Input:** The full JSON `data` dictionary.
        *   **Logic 1 (Consistency King):** From `enhanced_strategy_analysis`, find the car with the lowest non-null `race_pace_consistency_stdev`. Create insight: `{"category": "Performance", "type": "Consistency King", "car_number": "...", "driver_name": "...", "details": "Most consistent driver with a standard deviation of only X.XXXs on clean, fuel-corrected laps."}`.
        *   **Logic 2 (Leaving Time on the Table):** From `fastest_by_car_number`, calculate the delta between `fastest_lap` and `optimal_lap_time`. If delta > 0.3s, create insight: `{"category": "Performance", "type": "Optimal Lap Delta", "car_number": "...", "details": "Car #X has a Ys gap between their fastest lap and their optimal lap, indicating inconsistent sector performance."}`.
    
    *   `find_degradation_insights(degradation_data)`:
        *   **Input:** The list from the `enhanced_strategy_analysis` key.
        *   **Logic 1 (Tire-Killer Alert):** Group by `manufacturer`. For each manufacturer, average the `deg_coeff_a` from the `tire_degradation_model`. Find the manufacturer with the highest (most positive) average coefficient. Create insight: `{"category": "Tire Management", "type": "Tire-Killer Alert", "manufacturer": "...", "details": "[Manufacturer] shows the highest tire degradation rate based on the polynomial model."}`.

    *   `find_driver_delta_insights(delta_data)`:
        *   **Input:** The list from the `driver_deltas_by_car` key.
        *   **Logic (Largest Teammate Pace Gap):** Find the car with the largest `average_lap_time_delta_for_car`. Create insight: `{"category": "Driver Performance", "type": "Largest Teammate Pace Gap", "car_number": "...", "details": "Car #X has the largest pace delta (Ys) between its drivers."}`.

    6.  The main handler should aggregate all returned insights into a single list.
    7.  This list of insights should be saved as `[BASENAME]_insights.json` and uploaded to the `imsa-analyzed-data-[SUFFIX]` bucket.
    8.  Finally, publish a message to `visualization-requests` with the GCS paths for both the `...enhanced.json` and the new `...insights.json`.
    9.  Remember to include helper functions to parse time strings like "1:23.456" into seconds for calculations.
    10. The `requirements.txt` should include `Flask`, `gunicorn`, `google-cloud-storage`, and `google-cloud-pubsub`.
    ```
*   **Task 3.2: Deploy the Agent**
    *   **[Human Engineer]** Follow the same build and deploy steps as in Task 2.2, but for the `insight-hunter` agent, deploying to a service named `insight-hunter-service` and creating a subscription `insight-hunter-sub` on the `insight-requests` topic.

---

### **Phases 4, 5, 6: Historian, Visualizer, Scribe, Publicist**

These agents follow the exact same "AI Prompt -> Human Deploy" pattern as Phase 3. The prompts would be similarly detailed, specifying the exact logic for querying BigQuery (`Historian`), generating Plotly/Matplotlib charts (`Visualizer`), populating a Jinja2 template (`Scribe`), and calling the Gemini API (`Publicist`).

Absolutely. Let's build out the detailed engineering plans and AI prompts for the remaining, more advanced agents. The key to success here is providing the AI with extremely rich, structured context, including data schemas, desired outputs, and step-by-step logic.

### **Guidance on Crafting Effective AI Prompts for these Agents**

1.  **Be the "Expert User":** You know exactly what a race engineer wants to see. Describe the final output (the chart, the PDF, the tweet) in meticulous detail. The AI is a tool to get you there; you are the architect.
2.  **Provide Schema and Data Snippets:** Never assume the AI knows the structure of your JSON files. Paste snippets of the `...enhanced.json` and `...insights.json` directly into the prompt. This anchors the AI's logic to your actual data.
3.  **Use Step-by-Step Logic:** Break down complex tasks into a numbered list of instructions. Instead of "Make a chart," say "1. Filter the data for...", "2. Create a bar chart with X on the x-axis and Y on the y-axis...", "3. Set the title to...", "4. Save the file as a PNG to a temporary directory."
4.  **Specify Libraries and Functions:** Tell the AI *which* libraries to use (e.g., `google-cloud-bigquery`, `plotly.graph_objects`, `jinja2`, `weasyprint`, `google-cloud-aiplatform`). This prevents it from choosing an obscure or incompatible library.
5.  **Iterate and Refine:** Your first generated code might be 90% correct. Instead of starting over, copy the code, paste it back into a new prompt, and give specific refinement instructions: "This is good, but change the chart color to blue," or "Add an error handling block for when the BigQuery query returns no results."

---

### **Phase 4: The `Historian` Agent (Cloud Run)**

*   **Primary Goal:** To provide year-over-year (YoY) or event-over-event context by querying a historical database.
*   **[Human Engineer] Task 4.1: Populate BigQuery (One-time Task)**
    *   Before the agent can work, it needs data. Manually upload a few of your past `...enhanced.json` files to the `imsa_history.race_analyses` table.
        ```bash
        # For each historical JSON file you have:
        bq load --source_format=NEWLINE_DELIMITED_JSON \
          imsa_history.race_analyses \
          /path/to/your/historical_file.jsonl \
          event_id:STRING,track:STRING,year:INTEGER,session_type:STRING,analysis_json:JSON
        ```
    *   **Note:** You may need to slightly reformat your JSON files into newline-delimited JSON (one JSON object per line) or write a small script to do so. For this project, you can create a `.jsonl` file where each line is a JSON object like: `{"event_id": "2024_mido_race", "track": "mid-ohio", "year": 2024, "session_type": "race", "analysis_json": { ... your entire enhanced json ... }}`.

*   **[Coding AI Agent] Task 4.2: Generate Agent Code**

    **AI Prompt:**
    ```
    You are an expert Google Cloud data engineer creating the `Historian` agent for "Project Apex".

    **Agent Goal:**
    This Cloud Run service receives a GCS path to a *current* analysis file. It must query a BigQuery table containing *historical* analyses for the same track and session type, perform a year-over-year comparison on key metrics, and output the findings as a new insights JSON.

    **BigQuery Schema (Table: `imsa_history.race_analyses`):**
    `event_id:STRING`, `track:STRING`, `year:INTEGER`, `session_type:STRING`, `analysis_json:JSON`

    **Current Analysis JSON Snippet (`...enhanced.json`):**
    ```json
    {
        "fastest_by_manufacturer": [
            {
                "manufacturer": "Porsche",
                "fastest_lap": { "time": "1:26.740", ... },
                ...
            }
        ],
        "enhanced_strategy_analysis": [
            {
                "car_number": "57",
                "manufacturer": "Mercedes-AMG",
                "tire_degradation_model": {
                    "deg_coeff_a": 0.001, ...
                }
            }
        ]
    }
    ```
    
    **Detailed Requirements:**
    1.  Create `main.py`, `Dockerfile`, and `requirements.txt` for a Flask-based Cloud Run service. The push subscription will provide `{"analysis_path": "gs://..."}`.
    2.  The `main.py` handler must:
        a. Download the *current* analysis JSON from the GCS path.
        b. Extract the event details from the filename (e.g., `2025_mido_race` -> track: 'mido', year: 2025, session: 'race').
        c. Connect to BigQuery using the `google-cloud-bigquery` library.
        d. Construct and execute a SQL query to fetch the `analysis_json` for the *previous year* at the *same track* and for the *same session type*.
           - `SELECT analysis_json FROM \`[PROJECT_ID].imsa_history.race_analyses\` WHERE track = @track AND year = @year AND session_type = @session`
           - Use query parameters for security. `@year` should be the current year - 1.
        e. If the query returns no results, log a warning and exit gracefully.
        f. If results are found, load the historical `analysis_json` into a Python dictionary.
    3.  Create a function `compare_analyses(current_data, historical_data)` that performs the following comparisons and returns a list of insight dictionaries:
        *   **YoY Pole/Fastest Lap Delta:** Compare `fastest_lap` time for each manufacturer present in both reports. Create an insight: `{"category": "Historical Comparison", "type": "YoY Manufacturer Pace", "manufacturer": "...", "details": "[Manufacturer] is Xs faster/slower than last year."}`.
        *   **YoY Tire Degradation:** For key manufacturers (e.g., "Porsche", "BMW", "Mercedes-AMG"), compare the `deg_coeff_a` from `enhanced_strategy_analysis`. Create an insight: `{"category": "Historical Comparison", "type": "YoY Tire Degradation", "manufacturer": "...", "details": "[Manufacturer] tire degradation profile has improved/worsened by Z% year-over-year."}`.
    4.  Aggregate the insights.
    5.  Save the insights to `[BASENAME]_historical_insights.json` and upload to the analyzed data bucket.
    6.  This agent does not need to publish a Pub/Sub message. It is a terminal analysis step.
    7.  `requirements.txt` must include: `Flask`, `gunicorn`, `google-cloud-storage`, `google-cloud-bigquery`.
    ```
*   **[Human Engineer] Task 4.3: Deploy the Agent**
    *   Follow the same build/deploy/subscribe pattern as before, creating a service `historian-service` and a subscription `historian-sub` on the `insight-requests` topic. Note: Both `InsightHunter` and `Historian` can subscribe to the same topic to run in parallel.

---

### **Phase 5: The `Visualizer` Agent (Cloud Run)**

*   **Primary Goal:** To translate structured JSON insights into compelling, shareable graphs.
*   **[Coding AI Agent] Task 5.1: Generate Agent Code**

    **AI Prompt:**
    ```
    You are an expert data visualization developer creating the `Visualizer` agent for "Project Apex".

    **Agent Goal:**
    This Cloud Run service listens for requests and generates a set of predefined charts based on the provided analysis data (`...enhanced.json`) and insights data (`...insights.json`).

    **Input Data Snippets:**
    *   `...enhanced.json`: Contains `race_strategy_by_car` -> `stints` -> `avg_green_lap_time`, `best_5_lap_avg`, etc. and `enhanced_strategy_analysis` -> `avg_pit_stationary_time`.
    *   `...insights.json`: `[{"category": "Pit Stop Intelligence", "type": "Stationary Time Champion", ...}]`

    **Detailed Requirements:**
    1.  Create `main.py`, `Dockerfile`, and `requirements.txt` for a Flask service. The push subscription will provide `{"analysis_path": "...", "insights_path": "..."}`.
    2.  The handler must download both JSON files from GCS.
    3.  Create a main function `generate_all_visuals(analysis_data, insights_data)` that calls individual plotting functions. Each plotting function will save a `.png` file to `/tmp/`.
    4.  After all plots are generated, the handler must upload all files from `/tmp/` to a subfolder in the analyzed data bucket, e.g., `gs://imsa-analyzed-data-[SUFFIX]/[BASENAME]/visuals/`.
    5.  Implement the following plotting functions using **Matplotlib** and **Seaborn** for aesthetics.

    **Plotting Function Specifications:**
    
    *   `plot_pit_stationary_times(analysis_data, output_path)`:
        a. Extract `car_number` and `avg_pit_stationary_time` (parsed to seconds) for all cars from `enhanced_strategy_analysis`.
        b. Create a horizontal bar chart showing the average stationary time for each car.
        c. Sort the bars from fastest (shortest time) to slowest.
        d. Add data labels to the end of each bar showing the time in seconds.
        e. Set title to "Average Pit Stop Stationary Time by Car". Label axes appropriately.
        f. Save as a high-DPI PNG to `output_path`.

    *   `plot_stint_pace_falloff(analysis_data, car_number, output_path)`:
        a. **This function should be called for our team's car number.**
        b. From `race_strategy_by_car`, find the data for the specified `car_number`.
        c. For each stint with more than 10 laps, plot the fuel-corrected lap time (`LAP_TIME_FUEL_CORRECTED_SEC` from the raw data, which needs to be added to the report) for each `lap_in_stint`.
        d. Use a different color for each stint.
        e. Overlay the tire degradation polynomial curve from the `tire_degradation_model` for that car.
        f. Set title to "Car #[car_number] - Fuel-Corrected Pace & Tire Model by Stint".
        g. Save to `output_path`.

    *   `plot_driver_consistency(analysis_data, output_path)`:
        a. Extract `car_number` and `race_pace_consistency_stdev` for all cars.
        b. Create a bar chart showing the standard deviation for each car. Sort from most consistent (lowest stdev) to least.
        c. Set title to "Race Pace Consistency (StDev of Clean Laps)".
        d. Save to `output_path`.

    6. `requirements.txt` must include: `Flask`, `gunicorn`, `google-cloud-storage`, `matplotlib`, `seaborn`, `pandas`, `numpy`.
    ```
*   **[Human Engineer] Task 5.2: Deploy the Agent**
    *   Build, deploy, and subscribe `visualizer-service` to the `visualization-requests` topic.

---

### **Phase 6: The `Scribe` and `Publicist` Agents (Cloud Run)**

These are the final output agents. They can be triggered by the same `visualization-requests` topic as they represent parallel output streams.

#### **`Scribe` Agent**

*   **Primary Goal:** To create a professional, automated PDF engineering report.
*   **[Coding AI Agent] Task 6.1: Generate `Scribe` Agent Code**

    **AI Prompt:**
    ```
    You are an expert report generation developer creating the `Scribe` agent for "Project Apex".

    **Agent Goal:**
    Create a Cloud Run service that generates a PDF report from the analysis and insights JSON files using Jinja2 and WeasyPrint.

    **Detailed Requirements:**
    1.  Create `main.py`, `Dockerfile`, `requirements.txt`, and a `report_template.html` for a Flask service.
    2.  The handler will receive `{"analysis_path": "...", "insights_path": "..."}`. It will download both JSONs.
    3.  Create a `report_template.html` file. This is a Jinja2 template. It should have a professional layout with a title, sections, and placeholders for data. Use CSS for styling (can be in a `<style>` tag).
        *   **Template Structure:**
            - Title: "Project Apex Race Report - [Event Name]"
            - Section: "Executive Summary" -> Loop through all insights where `category == 'Historical Comparison'`.
            - Section: "Performance Insights" -> Loop through insights where `category == 'Performance'`.
            - Section: "Pit & Strategy Insights" -> Loop through insights where `category == 'Pit Stop Intelligence'`.
            - Section: "Tire Management Insights" -> Loop through insights where `category == 'Tire Management'`.
            - You can use Jinja2 control structures: `{% for insight in insights %}` and `{{ insight.details }}`.
    4.  The `main.py` handler must:
        a. Load the Jinja2 template from the file.
        b. Render the template, passing the `insights` list to it.
        c. Use `weasyprint` to convert the rendered HTML string into a PDF byte stream.
        d. Save the PDF as `[BASENAME]_Race_Report.pdf` and upload it to the analyzed data bucket.
    5.  `requirements.txt`: `Flask`, `gunicorn`, `google-cloud-storage`, `Jinja2`, `WeasyPrint`.
    ```

#### **`Publicist` Agent**

*   **Primary Goal:** To use a Generative AI model to draft social media content.
*   **[Coding AI Agent] Task 6.2: Generate `Publicist` Agent Code**

    **AI Prompt:**
    ```
    You are an AI engineer creating the `Publicist` agent for "Project Apex", leveraging Google's Vertex AI Gemini API.

    **Agent Goal:**
    This Cloud Run service takes key insights, selects a few that are highly positive and shareable, and uses the Gemini Pro model to generate engaging social media posts.

    **Detailed Requirements:**
    1.  Create `main.py`, `Dockerfile`, and `requirements.txt` for a Flask service. The handler receives `{"insights_path": "..."}`.
    2.  The handler must:
        a. Download the `...insights.json` file.
        b. Filter the insights to find "shareable" ones. Good candidates are types like "Stationary Time Champion", "Consistency King", or any insight where our team's car number is featured positively.
        c. For each selected shareable insight, initialize the Vertex AI client: `vertexai.init(project="[YOUR_PROJECT_ID]", location="us-central1")` and the model `gemini_pro_model = GenerativeModel("gemini-pro")`.
        d. Craft a specific, detailed prompt for Gemini. The prompt should provide context and ask for a specific output format.
            **Example Gemini Prompt:**
            ```
            "You are the social media manager for the [YOUR TEAM NAME] professional racing team. Your tone is exciting, confident, and tech-focused. Based on the following data insight, write three distinct tweet variations. Each tweet must be under 280 characters, include the hashtags #IMSA and #MichelinPilotChallenge, and tag @IMSA.

            Data Insight:
            Category: Pit Stop Intelligence
            Type: Stationary Time Champion
            Car Number: [OUR CAR #]
            Details: Fastest crew on pit road with an average stationary time of 24.1s.

            Generate the response as a JSON array of strings."
            ```
        e. Send the prompt to the Gemini API: `response = gemini_pro_model.generate_content(prompt)`.
        f. Parse the response to get the list of generated tweets.
        g. Aggregate all generated posts into a single JSON object.
        h. Save the object as `[BASENAME]_social_posts.json` and upload to GCS.
    3.  `requirements.txt`: `Flask`, `gunicorn`, `google-cloud-storage`, `google-cloud-aiplatform`.
    ```

*   **[Human Engineer] Task 6.3: Deploy `Scribe` and `Publicist`**
    *   Build, deploy, and subscribe `scribe-service` and `publicist-service` to the `visualization-requests` topic. They will run in parallel with the `Visualizer`.


### **Phase 7: Orchestration and Final Testing**

*   **Task 7.1: Write the ADK Orchestrator**
    *   **[Human Engineer]** The ADK script is the "user interface" for this system. It won't call the agents directly but will start the process. A simple version could upload the files to GCS to kick off the chain reaction.
        ```python
        # adk-orchestrator.py
        from google.cloud import storage
        import argparse
        
        def start_analysis_workflow(csv_file, pit_json_file, bucket_name):
            """Uploads local data files to GCS to trigger the workflow."""
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
        
            csv_blob = bucket.blob(csv_file.split('/')[-1])
            pit_blob = bucket.blob(pit_json_file.split('/')[-1])
            
            print(f"Uploading {csv_file} to {bucket_name}...")
            csv_blob.upload_from_filename(csv_file)
            print("...Done.")
            
            print(f"Uploading {pit_json_file} to {bucket_name}...")
            pit_blob.upload_from_filename(pit_json_file)
            print("...Done.")
            
            print("\nWorkflow triggered. Monitor logs in Google Cloud Console.")

        if __name__ == "__main__":
            parser = argparse.ArgumentParser(description='Trigger Project Apex Analysis Workflow.')
            parser.add_argument('csv', help='Path to the local CSV data file.')
            parser.add_argument('pits', help='Path to the local pit JSON data file.')
            parser.add_argument('--bucket', required=True, help='Name of the GCS raw data bucket.')
            
            args = parser.parse_args()
            
            start_analysis_workflow(args.csv, args.pits, args.bucket)
        ```

*   **Task 7.2: End-to-End Test**
    *   **[Human Engineer]**
    1.  Get your sample `2025_impc_mido.csv` and a `2025_impc_mido_pits.json` file.
    2.  Run the orchestrator from your local machine:
        ```bash
        python adk-orchestrator.py 2025_impc_mido.csv 2025_impc_mido_pits.json --bucket imsa-raw-data-[YOUR_UNIQUE_SUFFIX]
        ```
    3.  Go to the Google Cloud Console.
    4.  Check the logs for the `data-ingestor-function`. It should show it found both files and published a message.
    5.  Check the logs for the `core-analyzer-service`. It should show it received the message and started analysis.
    6.  Check the `imsa-analyzed-data` bucket. You should see `2025_impc_mido_results_enhanced.json` appear.
    7.  Check the logs for the `insight-hunter-service`. It should show it received the new path.
    8.  Check the bucket again. You should see `2025_impc_mido_results_insights.json` appear.
    9.  Continue monitoring the chain for all deployed agents. This verifies the entire asynchronous, event-driven architecture is working correctly.