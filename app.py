import os
import json
import io
import streamlit as st
import pandas as pd
import plotly.io as pio
from openai import OpenAI
from dotenv import load_dotenv

# Load from .env
load_dotenv()

# UI Config
st.set_page_config(page_title="Open Pandas AI Dashboard", layout="wide")
st.title("📊 Open Pandas AI Dashboard")

# API Key
st.sidebar.header("🔐 API Key Management")
key_file = ".openai_key.txt"

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

# Init OpenAI
try:
    client = OpenAI(api_key=st.session_state.api_key)
except Exception as e:
    st.sidebar.error(f"Invalid API Key: {e}")
    st.stop()

# Upload
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

        # Agents
        from ai_modules.pandas_analyst import PandasAnalyst
        from ai_modules.wrangling_agent import DataWranglingAgent
        from ai_modules.visualization_agent import VisualizationAgent

        wrangler = DataWranglingAgent(client=client)
        visualizer = VisualizationAgent(model=client)
        analyst = PandasAnalyst(model=client, wrangler=wrangler, visualizer=visualizer)

        st.session_state.df = df
        st.session_state.analyst = analyst

    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()
else:
    st.info("Please upload a data file to continue.")
    st.stop()

# Define the dataset summary function
def generate_dataset_summary(df: pd.DataFrame, client: OpenAI) -> str:
    col_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        na = df[col].isna().sum()
        unique = df[col].nunique()
        sample = df[col].dropna().astype(str).unique().tolist()[:5]
        col_info.append(f"- **{col}** ({dtype}), {unique} unique, {na} missing. Examples: {sample}")

    schema_description = "\n".join(col_info)
    prompt = f"""
You are a data analyst.

Generate a detailed, helpful, human-readable summary of the dataset based on the following schema details:

{schema_description}

Include:
- Number of rows and columns
- Notes on missing data
- Types of columns (categorical, numeric, dates)
- Observations or patterns (e.g. skewed data, many duplicates, etc)

Write in markdown format.
""".strip()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a data profiling assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content.strip()

# Data Access
df = st.session_state.df
analyst = st.session_state.analyst

# === Dataset Tools ===
st.subheader("📊 Dataset Tools")
if st.button("🧠 Summarize This Dataset"):
    with st.spinner("Summarizing..."):
        try:
            summary_text = generate_dataset_summary(df, client)
            st.subheader("🧠 Dataset Summary")
            st.markdown(summary_text)
        except Exception as e:
            st.error(f"Error during summarization: {e}")

# === Ask a Question ===
st.subheader("💬 Ask a Question About the Data")
query = st.text_input("Example: Show the top 5 categories by sales")

if query:
    with st.spinner("Analyzing..."):
        try:
            analyst.invoke_agent(query, df)

            st.subheader("🔎 Answer")
            st.write(analyst.get_workflow_summary(markdown=True))

            if analyst.get_plotly_graph():
                st.plotly_chart(analyst.get_plotly_graph(), use_container_width=True)
            elif analyst.get_data_wrangled() is not None:
                st.dataframe(analyst.get_data_wrangled(), use_container_width=True)
            else:
                st.warning("No chart or table detected.")

        except Exception as e:
            st.error(f"Failed to analyze query: {e}")

