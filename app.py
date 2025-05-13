import os
import io
import streamlit as st
import pandas as pd
import json
import plotly.io as pio
from openai import OpenAI

# Load from .env if exists
from dotenv import load_dotenv
load_dotenv()

# UI Configuration
st.set_page_config(page_title="Open Pandas AI Dashboard", layout="wide")
st.title("📊 Open Pandas AI Dashboard")

# Sidebar API Input
st.sidebar.header("🔐 API Key Management")
key_file = ".openai_key.txt"

# Load API key from file or prompt user
if "api_key" not in st.session_state:
    if os.path.exists(key_file):
        with open(key_file, "r") as f:
            st.session_state.api_key = f.read().strip()
    else:
        st.session_state.api_key = ""

if not st.session_state.api_key:
    new_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
    if new_key:
        with open(key_file, "w") as f:
            f.write(new_key.strip())
        st.session_state.api_key = new_key.strip()
        st.rerun()
else:
    st.sidebar.success("API Key is loaded.")
    if st.sidebar.button("🔄 Update API Key"):
        st.session_state.api_key = ""
        if os.path.exists(key_file):
            os.remove(key_file)
        st.rerun()

if not st.session_state.api_key:
    st.warning("Please enter your OpenAI API key to continue.")
    st.stop()

# Initialize OpenAI client
try:
    client = OpenAI(api_key=st.session_state.api_key)
    models = client.models.list()
    st.sidebar.success("API Key is valid!")
except Exception as e:
    st.sidebar.error(f"Invalid API Key: {e}")
    st.stop()

# File upload
st.markdown("Upload a CSV or Excel file to begin analyzing.")
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("File uploaded successfully.")
        st.subheader("🔍 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

        # --- AGENT SYSTEM SETUP ---
        from ai_modules.pandas_analyst import PandasAnalyst
        from ai_modules.wrangling_agent import WranglingAgent
        from ai_modules.visualization_agent import VisualizationAgent

        wrangler = WranglingAgent(model=client)
        visualizer = VisualizationAgent(model=client)
        analyst = PandasAnalyst(model=client, wrangler=wrangler, visualizer=visualizer)

        # Store for downstream calls
        st.session_state.df = df
        st.session_state.analyst = analyst

    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()
else:
    st.info("Please upload a data file to continue.")
    st.stop()

# Dataset tools
df = st.session_state.df
analyst = st.session_state.analyst

st.subheader("📊 Dataset Tools")
if st.button("🧠 Summarize This Dataset"):
    with st.spinner("Summarizing dataset..."):
        try:
            analyst.invoke_agent("Summarize this dataset", df)
            st.subheader("🧠 Dataset Summary")
            st.write(analyst.get_workflow_summary(markdown=True))
        except Exception as e:
            st.error(f"Failed to summarize: {e}")

# User Question
st.subheader("💬 Ask a Question About the Data")
query = st.text_input("Example: Show the top 5 categories by sales")

if query:
    with st.spinner("Running analysis..."):
        try:
            analyst.invoke_agent(query, df)

            st.markdown("**🔎 Answer:**")
            if analyst.get_plotly_graph():
                st.plotly_chart(analyst.get_plotly_graph(), use_container_width=True)
            elif analyst.get_data_wrangled() is not None:
                st.dataframe(analyst.get_data_wrangled(), use_container_width=True)
            else:
                st.warning("No chart or table detected.")

            st.subheader("🛠️ Generated Code")
            if analyst.get_data_wrangler_function():
                st.code(analyst.get_data_wrangler_function(), language="python")
            if analyst.get_data_visualization_function():
                st.code(analyst.get_data_visualization_function(), language="python")

        except Exception as e:
            st.error(f"Failed to analyze query: {e}")
