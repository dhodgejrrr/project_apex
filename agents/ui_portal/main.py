import os
import pathlib
import secrets
import shutil
import json
import base64

import streamlit as st
from google.cloud import storage, pubsub_v1

# --- Configuration ---
# These come from environment variables set in the Cloud Run service
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT")
if not PROJECT_ID:
    raise ValueError("GOOGLE_CLOUD_PROJECT or GCP_PROJECT environment variable must be set")

RAW_BUCKET_NAME = os.getenv("RAW_DATA_BUCKET")
if not RAW_DATA_BUCKET:
    raise ValueError("RAW_DATA_BUCKET environment variable must be set")
    
ANALYZED_BUCKET_NAME = os.getenv("ANALYZED_DATA_BUCKET")
if not ANALYZED_BUCKET_NAME:
    raise ValueError("ANALYZED_DATA_BUCKET environment variable must be set")
    
ORCHESTRATION_TOPIC_ID = os.getenv("ORCHESTRATION_TOPIC_ID")
if not ORCHESTRATION_TOPIC_ID:
    raise ValueError("ORCHESTRATION_TOPIC_ID environment variable must be set")

# Use st.cache_resource for clients to avoid re-initializing on every interaction.
@st.cache_resource
def get_gcs_client() -> storage.Client:
    return storage.Client()

@st.cache_resource
def get_pubsub_client() -> pubsub_v1.PublisherClient:
    return pubsub_v1.PublisherClient()

def trigger_orchestration(run_id: str, csv_gcs_path: str, pit_gcs_path: str):
    """Publishes a message to the orchestration topic to start the workflow."""
    try:
        publisher = get_pubsub_client()
        topic_path = publisher.topic_path(PROJECT_ID, ORCHESTRATION_TOPIC_ID)
        
        # Log the project ID and topic path for debugging
        st.sidebar.info(f"Publishing to project: {PROJECT_ID}")
        st.sidebar.info(f"Topic path: {topic_path}")
        
        payload = {
            "run_id": run_id,
            "csv_gcs_path": csv_gcs_path,
            "pit_gcs_path": pit_gcs_path,
        }
        
        # Publish the message
        future = publisher.publish(
            topic_path,
            json.dumps(payload).encode("utf-8")
        )
        
        # Wait for the publish to complete
        message_id = future.result(timeout=60)
        st.success(f"Orchestration triggered for Run ID: {run_id}")
        st.sidebar.success(f"Message published with ID: {message_id}")
        
    except Exception as e:
        st.error(f"Failed to trigger orchestration: {str(e)}")
        st.sidebar.error(f"Error details: {str(e)}")
        raise

def upload_page():
    st.title("🚀 Project Apex: Trigger New Analysis")
    st.write("Upload a race CSV and its corresponding pit data JSON to begin the multi-agent analysis.")

    # Create a unique form key using a timestamp to prevent duplicates
    form_key = f"upload_form_{st.session_state.get('form_key', 0)}"
    with st.form(key=form_key):
        csv_file = st.file_uploader("Upload Race CSV", type=["csv"])
        pit_file = st.file_uploader("Upload Pit Data JSON", type=["json"])
        submitted = st.form_submit_button("Start Analysis")

        if submitted and csv_file and pit_file:
            with st.spinner("Uploading files and triggering workflow..."):
                run_id_base = pathlib.Path(csv_file.name).stem.replace("_race", "")
                run_id = f"{run_id_base}-{secrets.token_hex(3)}"
                
                storage_client = get_gcs_client()
                bucket = storage_client.bucket(RAW_BUCKET_NAME)

                csv_blob_name = f"{run_id}/{csv_file.name}"
                pit_blob_name = f"{run_id}/{pit_file.name}"

                bucket.blob(csv_blob_name).upload_from_file(csv_file)
                bucket.blob(pit_blob_name).upload_from_file(pit_file)

                csv_gcs_path = f"gs://{RAW_BUCKET_NAME}/{csv_blob_name}"
                pit_gcs_path = f"gs://{RAW_BUCKET_NAME}/{pit_blob_name}"

                trigger_orchestration(run_id, csv_gcs_path, pit_gcs_path)
                st.session_state['run_id'] = run_id
                st.experimental_rerun()

def results_page():
    # Page config is now set in app.py
    st.set_page_config(layout="wide")
    run_id = st.query_params["run_id"]
    st.title(f"📊 Results for Run: `{run_id}`")
    
    if st.button("Check for Updates"):
        st.experimental_rerun()

    storage_client = get_gcs_client()
    bucket = storage_client.bucket(ANALYZED_BUCKET_NAME)
    
    # Check for artifacts
    report_blob = bucket.get_blob(f"{run_id}/reports/race_report.pdf")
    tweets_blob = bucket.get_blob(f"{run_id}/social/social_media_posts.json")
    visuals_blobs = list(storage_client.list_blobs(ANALYZED_BUCKET_NAME, prefix=f"{run_id}/visuals/"))

    if not any([report_blob, tweets_blob, visuals_blobs]):
        st.info("Analysis is in progress. Please check back in a few minutes.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.header("📝 Scribe Agent Report")
        if report_blob:
            st.link_button("Download Full PDF Report", report_blob.public_url)
            # Embedding PDF
            pdf_base64 = base64.b64encode(report_blob.download_as_bytes()).decode('utf-8')
            st.markdown(f'<iframe src="data:application/pdf;base64,{pdf_base64}" width="100%" height="800px"></iframe>', unsafe_allow_html=True)
        else:
            st.warning("PDF report is not yet available.")

    with col2:
        st.header("📱 Publicist Agent Tweets")
        if tweets_blob:
            tweets = json.loads(tweets_blob.download_as_string())
            for i, tweet in enumerate(tweets.get("posts", [])):
                st.text_area(f"Tweet Suggestion {i+1}", tweet, height=100)
        else:
            st.warning("Social media posts are not yet available.")
            
        st.header("📈 Visualizer Agent Plots")
        if visuals_blobs:
            for blob in visuals_blobs:
                if blob.name.endswith(".png"):
                    st.image(blob.public_url, caption=pathlib.Path(blob.name).name, use_column_width=True)
        else:
            st.warning("Visuals are not yet available.")

# This code is now handled in app.py
# The main router has been moved to app.py to avoid duplicate execution