import os
import streamlit as st
from main import upload_page, results_page

def app():
    # This is the entry point for gunicorn
    port = int(os.environ.get("PORT", 8080))
    
    # Set the page config
    st.set_page_config(
        page_title="Project Apex Portal",
        page_icon="🚀",
        layout="centered"
    )
    
    # Simple router based on URL parameters
    if 'run_id' in st.query_params:
        results_page()
    else:
        upload_page()

# For local development
if __name__ == "__main__":
    app()
