Excellent. Let's move from the high-level strategy to a detailed engineering and requirements blueprint. This will serve as your technical guide for the hackathon.

We will break down each agent's specific role, the data it consumes, the logic it applies, and the artifacts it produces.

---

### **Engineering Blueprint: Project Apex**

#### **Core Principle: Data-Driven Orchestration**
The entire system revolves around the `_enhanced.json` report your `CoreAnalyzer` generates. Every subsequent agent is designed to parse, interpret, and add value to this central data artifact.

---

### **Agent 1: `DataIngestor`**

*   **Primary Goal:** To reliably and automatically detect new race data and initiate the analysis pipeline.
*   **Engineering Details:**
    *   **Inputs:**
        *   Raw IMSA timing & scoring CSV files (`*.csv`).
        *   Raw IMSA pit data JSON files (`*.json`).
    *   **Core Logic:**
        *   This agent will be implemented as a **Google Cloud Function**.
        *   It will be configured with a **Cloud Storage Trigger**. It will monitor a specific bucket, e.g., `gs://imsa-raw-data-uploads/2025/mid-ohio/`.
        *   When a new file is uploaded, the function triggers. It will parse the filename to understand the event (e.g., `2025_mido_race.csv`, `2025_mido_pits.json`).
        *   It will wait for both the CSV and the corresponding pit JSON to be present before proceeding to avoid partial analysis. A simple check for the sibling file can work.
    *   **Outputs:**
        *   A message published to a **Google Pub/Sub topic** called `analysis-requests`.
        *   The message payload will be a JSON object containing the Cloud Storage paths to the validated CSV and pit JSON files: `{ "csv_path": "gs://...", "pit_json_path": "gs://..." }`.
*   **Google Cloud Integration:**
    *   **Cloud Storage:** For storing raw data uploads.
    *   **Cloud Functions:** For the trigger-based, serverless logic.
    *   **Pub/Sub:** To decouple the ingestion from the analysis, making the system resilient and scalable.

---

### **Agent 2: `CoreAnalyzer`**

*   **Primary Goal:** To execute your existing `imsa_analyzer.py` script in a scalable, containerized environment.
*   **Engineering Details:**
    *   **Inputs:**
        *   Subscribes to the `analysis-requests` Pub/Sub topic.
        *   Receives the JSON message with file paths.
    *   **Core Logic:**
        *   This agent will be a **Docker container** running a lightweight Python web server (like Flask or FastAPI).
        *   It will be deployed as a **Google Cloud Run** service. This allows it to scale down to zero when not in use (cost-effective) and scale up if multiple sessions are analyzed at once.
        *   The main script (`app.py`) will have an endpoint that listens for the Pub/Sub push subscription.
        *   Upon receiving a message, it downloads the data files from Cloud Storage, instantiates `IMSADataAnalyzer(csv_path, pit_json_path)`, runs `run_all_analyses()`, and gets the final JSON result.
    *   **Outputs:**
        *   The comprehensive analysis result, e.g., `2025_mido_race_results_enhanced.json`.
        *   This output file is uploaded to a different Cloud Storage bucket, e.g., `gs://imsa-analyzed-data/2025/mid-ohio/`.
        *   It then publishes a message to a new Pub/Sub topic, `insight-requests`, with the path to the newly created analysis file.
*   **Google Cloud Integration:**
    *   **Cloud Run:** To host the containerized Python analysis code.
    *   **Cloud Storage:** To read raw data and write analyzed data.
    *   **Pub/Sub:** To receive requests and trigger the next stage of the pipeline.
    *   **Artifact Registry:** To store the Docker container image.

---

### **Agent 3: `InsightHunter` - The Brains of the Operation**

*   **Primary Goal:** To systematically scan the analysis JSON, identify tactically significant patterns, and flag critical insights for the race team.
*   **Engineering Details:**
    *   **Inputs:** Subscribes to the `insight-requests` Pub/Sub topic and receives the path to the `...enhanced.json` file.
    *   **Core Logic:** Deployed as a Cloud Run service, just like the `CoreAnalyzer`. Its Python code will perform a series of checks based on the categories below. It will iterate through the data and build a list of "Insight" objects.
    *   **Outputs:** A structured JSON object called `insights.json` containing a list of findings. This file is saved to the analyzed data bucket and its path is passed to downstream agents.

#### **Firm Insights & Categories to Capture (Leveraging your existing report data):**

Here's a detailed breakdown of what the `InsightHunter` will look for.

**Category 1: Pit Stop & Strategy Intelligence**
*   **Insight: "Pit Delta Outlier"**
    *   **Logic:** For each car, find the `average_pit_time` from `race_strategy_by_car`. Then iterate through its `pit_stop_details`. If any stop's `total_pit_lane_time` is > 1.5x the average (excluding obvious long stops for damage), flag it.
    *   **Value:** Identifies fumbled pit stops or potential mechanical issues for competitors.
*   **Insight: "Driver Change Cost"**
    *   **Logic:** For each car, calculate the average `stationary_time` for stops where `driver_change: true` vs. `driver_change: false`.
    *   **Value:** Quantifies how much time a team loses to a driver change. Essential for strategy planning.
*   **Insight: "Stationary Time Champion"**
    *   **Logic:** From `enhanced_strategy_analysis`, find the car with the lowest `avg_pit_stationary_time`. Rank the top 3.
    *   **Value:** Identifies the most efficient crews on pit road. A great marketing point if it's us.

**Category 2: Performance, Pace & Consistency**
*   **Insight: "Consistency King"**
    *   **Logic:** From `enhanced_strategy_analysis`, find the car with the lowest `race_pace_consistency_stdev`. Rank top 3.
    *   **Value:** Highlights drivers who can reel off consistent lap times, which is often more valuable than a single fast lap.
*   **Insight: "Leaving Time on the Table"**
    *   **Logic:** From `fastest_by_car_number`, compare `fastest_lap` (time) to `optimal_lap_time`. Calculate the delta. Flag cars with a delta > 0.3 seconds.
    *   **Value:** Shows which drivers are having trouble stringing their best sectors together in a single lap.
*   **Insight: "Stint Pace Hero"**
    *   **Logic:** From `race_strategy_by_car` -> `stints`, find the best `best_5_lap_avg` across all cars and all stints.
    *   **Value:** Identifies the car/driver combination with the best raw pace during a stint's peak performance window.

**Category 3: Tire Degradation & Management**
*   **Insight: "Tire-Killer Alert"**
    *   **Logic:** From `enhanced_strategy_analysis` -> `tire_degradation_model`, find the manufacturer with the highest positive `deg_coeff_a` (the quadratic term). This indicates the fastest drop-off.
    *   **Value:** Critical BoP insight. If a certain car model is eating its tires, we know we can pressure them late in a stint.
*   **Insight: "Late-Stint Prowess"**
    *   **Logic:** For each stint longer than 20 laps, compare the average time of the first 5 laps to the average time of the last 5 laps (fuel-corrected). Find the driver with the smallest drop-off.
    *   **Value:** Identifies drivers who are excellent at managing their tires over a full run.
*   **Insight: "Traffic Master"**
    *   **Logic:** From `race_strategy_by_car` -> `stints`, find the car with the highest percentage of `traffic_in_class_laps` and `traffic_out_of_class_laps`. Calculate their average lap time on these "traffic" laps and compare it to their average "normal" laps.
    *   **Value:** Shows which drivers are most affected by traffic, or conversely, who is best at navigating it with minimal time loss.

**Category 4: Driver-Specific Deltas**
*   **Insight: "Largest Teammate Pace Gap"**
    *   **Logic:** From `driver_deltas_by_car`, find the car with the largest `lap_time_delta` between its fastest and slowest driver.
    *   **Value:** Pinpoints teams with a significant performance drop-off depending on the driver. We can attack when their slower driver is in the car.

---

### **Agent 4: `Historian`**

*   **Primary Goal:** To contextualize current performance against historical data.
*   **Engineering Details:**
    *   **Inputs:** A request containing the current `...enhanced.json` path and the comparison target (e.g., "same race, last year").
    *   **Core Logic:**
        *   Deployed as a Cloud Run service.
        *   **BigQuery Schema:** You will need a BigQuery table to store historical analysis JSONs. The schema can be simple: `event_id (STRING)`, `track (STRING)`, `year (INTEGER)`, `session_type (STRING)`, `analysis_json (JSON)`.
        *   The agent queries BigQuery for the relevant historical data.
        *   It then performs a diffing operation on key metrics:
            *   Overall fastest lap (class-by-class).
            *   Average green flag pace for top 5 cars.
            *   Tire degradation coefficients for key manufacturers.
    *   **Outputs:** A `historical_insights.json` file with findings like "Field is 0.8s faster on average than 2024" or "Porsche tire degradation has improved by 15% year-over-year."
*   **Google Cloud Integration:**
    *   **BigQuery:** The heart of this agent. It's the long-term memory for all your analysis.
    *   **Cloud Run:** To run the query and comparison logic.

---

### **Agent 5 & 6 & 7: `Visualizer`, `Scribe`, `Publicist`**

These are the "Output" agents. They are triggered after the `InsightHunter` and `Historian` complete their work.

*   **Engineering Details:**
    *   **Inputs:** The `insights.json` and/or `historical_insights.json` files.
    *   **Core Logic:**
        *   **`Visualizer`**: Uses **Plotly** to generate interactive HTML/JS charts (great for web-based dashboards) or Matplotlib for static PNGs. It will have a library of plotting functions, one for each insight category (e.g., `plot_pit_time_barchart`, `plot_degradation_curves`).
        *   **`Scribe`**: Uses a PDF generation library (**WeasyPrint** or **ReportLab**) with a Jinja2 template. It populates a pre-formatted engineering report with the text insights and embeds the generated charts.
        *   **`Publicist`**: This is a prime candidate for using the **Vertex AI Gemini API**. You can feed it an insight (e.g., "Insight: Stationary Time Champion, Car: #OUR_CAR, Time: 25.3s") and give it a prompt: *"You are a social media manager for a professional race team. Write three exciting and engaging tweet variations for the following insight. Use relevant hashtags and tag @IMSA."*
    *   **Outputs:**
        *   `Visualizer`: `.png` or `.html` files in `gs://imsa-analyzed-data/2025/mid-ohio/visuals/`.
        *   `Scribe`: `Engineering_Report_Race.pdf` in the same bucket.
        *   `Publicist`: A JSON file with suggested social media copy, `social_media_posts.json`.
*   **Google Cloud Integration:**
    *   **Cloud Run:** To host all three agents.
    *   **Cloud Storage:** To store the final output artifacts.
    *   **Vertex AI (Gemini):** A powerful addition for the `Publicist` to showcase advanced AI capabilities.

This detailed plan gives you a clear path forward. You have the core engine; now you're building an intelligent, automated factory around it using Google's ADK and cloud services. This is a very strong foundation for a winning hackathon project.