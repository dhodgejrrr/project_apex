import os
import streamlit as st
from main import upload_page, results_page

# Set page config at the module level
st.set_page_config(
    page_title="Project Apex Portal",
    page_icon="🚀",
    layout="centered"
)

def main():
    # Simple router based on URL parameters
    if 'run_id' in st.query_params:
        results_page()
    else:
        upload_page()

if __name__ == "__main__":
    main()
