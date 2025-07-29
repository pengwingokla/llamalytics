import streamlit as st
import pandas as pd
import tempfile
import os
import sys
import json
from io import StringIO
import traceback
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from agents.workflow import ask_question
    from tools.csv_tools import CSVAnalyzer
    WORKFLOW_AVAILABLE = True
except ImportError as e:
    WORKFLOW_AVAILABLE = False
    IMPORT_ERROR = str(e)

st.set_page_config(
    page_title="Agentic AI Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_embeded_dataset():
    ds_paths = [
        "data/NJ_graduation_data.csv"
    ]
    for path in ds_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                return df, path, True
            except Exception as e:
                continue
    return None, None, False

def initialize_session_state():
    if 'dataset_loaded' not in st.session_state:
        df, csv_path, success = load_embeded_dataset()
        
        if success:
            st.session_state.df = df
            st.session_state.csv_path = csv_path
            st.session_state.dataset_loaded = True
            st.session_state.dataset_name = os.path.basename(csv_path)
        else:
            st.session_state.dataset_loaded = False

# Streamed response emulator
def response_generator(user_question):
    try:
        yield "Analyzing your question... \n"
        csv_path = st.session_state.csv_path
        response = ask_question(user_question, csv_path)

        yield response

    except Exception as e:
        yield f"Error: {str(e)}" 

    # for word in response.split():
    #     yield word + " "
    #     time.sleep(0.05)

def main():
    initialize_session_state()

    # Header
    st.markdown("AI Data Analytics")

    # ===== CHECK BACKEND AND DATA LOADER =====
    # Check if backend is available
    if not WORKFLOW_AVAILABLE:
        st.error(f"Workflow not available: {IMPORT_ERROR}")
    
    # Check if dataset loaded
    if not st.session_state.get('dataset_loaded', False):
        st.error("No embded dataset found. CSV file to the data/ directory required.")

    # ===== CHAT ======
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a question about the graduation data..."):
        # Add user message to chat history and display in chat container
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container and store response to history
        with st.chat_message("assistant"):
            response = st.write_stream(response_generator(prompt))
        st.session_state.messages.append({"role": "assistant", "content": response})
if __name__ == "__main__":
    main()