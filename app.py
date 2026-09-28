"""
app.py  —  AI Data Analyst Agent  |  Main Application
=======================================================

Run with:   streamlit run app.py

How the app works:
  1. User uploads a CSV  (or we use the built-in sample)
  2. We auto-clean it    (remove duplicates, fix types, fill nulls)
  3. User sees stats     (KPI cards, schema, cleaning report)
  4. User asks a question (plain English)
  5. AI generates code   (via local Ollama — fully offline)
  6. We validate & run it (AST security check → execute locally)
  7. Result is shown     (table + auto chart)
  8. Download PDF        (question + result + code)

Files in this project:
  app.py           → This file  (Streamlit UI)
  agent.py         → AI brain   (schema, LLM call, code validation, code execution)
  data_utils.py    → Data help  (CSV loading, cleaning, statistics, PDF export)
  visualization.py → UI pieces  (KPI cards, auto charts)
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

# Import our helper modules — one per responsibility
from agent import build_schema, schema_to_json, ask_llm, validate_code, run_code
from data_utils import load_csv, clean_data, get_stats, export_pdf
from visualization import show_kpis, show_cleaning_kpis, show_result_kpi, show_chart


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_CSV     = Path(__file__).parent / "data" / "sales.csv"
MAX_ROWS_SHOWN = 200  # Maximum rows we display in any table


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Render one Q&A turn in the chat
# ─────────────────────────────────────────────────────────────────────────────

def render_chat_entry(entry: dict) -> None:
    """
    Display one question + answer pair in the chat UI.

    Each entry is a dict with:
      entry["question"] → the user's question text
      entry["code"]     → the pandas code the AI generated
      entry["result"]   → the result (DataFrame, Series, number, or string)
      entry["error"]    → error message if something went wrong (or None)
    """
    with st.chat_message("user"):
        st.markdown(entry["question"])

    with st.chat_message("assistant"):
        # If there was an error, show it in red and stop
        if entry.get("error"):
            st.error(entry["error"])
            return

        result = entry["result"]

        # Show a big KPI card for single-value answers (e.g. "Total Revenue: 1.2M")
        show_result_kpi(result)

        # Show the result as a table for DataFrames and Series
        if isinstance(result, pd.DataFrame):
            st.dataframe(result.head(MAX_ROWS_SHOWN), use_container_width=True)
            if len(result) > MAX_ROWS_SHOWN:
                st.caption(f"Showing {MAX_ROWS_SHOWN} of {len(result):,} rows.")
        elif isinstance(result, pd.Series):
            st.dataframe(result.head(MAX_ROWS_SHOWN).to_frame(), use_container_width=True)
        elif isinstance(result, (list, tuple, dict)):
            st.write(result)

        # Auto-pick and render the most relevant chart
        if show_chart(result):
            st.caption("📈 Auto-generated chart")

        # Show the generated code (collapsed — keeps the UI clean)
        with st.expander("💻 See the generated Python code"):
            st.code(entry["code"], language="python")


# ─────────────────────────────────────────────────────────────────────────────
# Page Setup
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Data Analyst Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 AI Data Analyst Agent")
st.caption(
    "Ask questions about your data in plain English. "
    "The AI writes pandas code that runs **locally** — your raw data never leaves your machine."
)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("📂 Dataset")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    st.caption("Limits: 50 MB · 250,000 rows · 200 columns")
    st.markdown("---")
    st.markdown("### ⚙️ Agent Info")
    st.markdown("🤖 **Model:** Llama 3.1 8B (via Ollama)")
    st.markdown("🔒 **Privacy:** Fully offline")
    st.markdown("🛡️ **Security:** AST code validation")


# ─────────────────────────────────────────────────────────────────────────────
# Load the CSV
# ─────────────────────────────────────────────────────────────────────────────

try:
    if uploaded_file is not None:
        raw_df    = load_csv(uploaded_file)
        file_hash = hashlib.sha256(uploaded_file.getbuffer()).hexdigest()
        st.sidebar.success(f"✅ Loaded — {len(raw_df):,} rows")
    else:
        raw_df    = load_csv(SAMPLE_CSV)
        file_hash = hashlib.sha256(SAMPLE_CSV.read_bytes()).hexdigest()
        st.sidebar.info("Using built-in sample: `sales.csv`")

except Exception as exc:
    st.error(f"❌ Could not load the file: {exc}")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Auto-clean whenever a new file is uploaded
# (Streamlit re-runs the whole script on every interaction, so we cache in
#  st.session_state to avoid re-cleaning on every click)
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.get("file_hash") != file_hash:
    # A new file was uploaded — clean it and reset everything
    cleaned_df, cleaning_report = clean_data(raw_df)

    st.session_state.file_hash       = file_hash
    st.session_state.raw_df          = raw_df
    st.session_state.cleaned_df      = cleaned_df
    st.session_state.cleaning_report = cleaning_report
    st.session_state.schema          = build_schema(cleaned_df)
    st.session_state.conversation    = []   # clear chat history for new file
    st.session_state.last_result     = None

# Read everything back from session state
raw_df          = st.session_state.raw_df
cleaned_df      = st.session_state.cleaned_df
cleaning_report = st.session_state.cleaning_report
schema          = st.session_state.schema


# ─────────────────────────────────────────────────────────────────────────────
# KPI Cards — quick numbers at the top
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("📌 Dataset at a Glance")
show_kpis(cleaned_df, raw_df)

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# Data Explorer — Four tabs to explore the data
# ─────────────────────────────────────────────────────────────────────────────

tab_orig, tab_clean, tab_report, tab_schema, tab_stats = st.tabs([
    "📁 Original Data",
    "✨ Cleaned Data",
    "🧹 Cleaning Report",
    "📋 Schema  ← what AI sees",
    "📈 Statistics",
])

with tab_orig:
    # Show the raw unmodified data as it came from the CSV
    st.caption(f"{len(raw_df):,} rows × {len(raw_df.columns)} columns — raw, before cleaning")
    st.dataframe(raw_df.head(15), use_container_width=True)

with tab_clean:
    # This is what the AI actually uses for analysis
    st.caption(f"{len(cleaned_df):,} rows × {len(cleaned_df.columns)} columns — **used for AI analysis**")
    st.dataframe(cleaned_df.head(15), use_container_width=True)

with tab_report:
    st.markdown("### What the auto-cleaner changed")
    show_cleaning_kpis(cleaning_report)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Rows removed:**")
        st.write(f"- Empty rows removed: **{cleaning_report['empty_rows_removed']}**")
        st.write(f"- Duplicate rows removed: **{cleaning_report['duplicates_removed']}**")
    with col2:
        st.markdown("**Type fixes & null filling:**")
        for change in cleaning_report.get("columns_converted", []):
            st.write(f"- {change}")
        for col, n in cleaning_report.get("nulls_filled", {}).items():
            st.write(f"- `{col}`: **{n}** nulls filled")

with tab_schema:
    # Show exactly what goes to the AI — no raw rows, just descriptions
    st.caption(
        "⚠️ This is the ONLY information sent to the AI model. "
        "No actual CSV rows are included."
    )
    st.dataframe(pd.DataFrame(schema), use_container_width=True, hide_index=True)

with tab_stats:
    stats = get_stats(cleaned_df)
    if stats.empty:
        st.info("No numeric columns to summarise.")
    else:
        st.dataframe(stats, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Chat Interface
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("💬 Ask Your Data")
st.caption("Ask anything in plain English — the AI writes the code, we run it here.")

# Show all previous conversation turns
for entry in st.session_state.conversation:
    render_chat_entry(entry)

# The chat input box at the bottom
question = st.chat_input("e.g. Which product has the highest total revenue?")

if question:
    new_entry = {"question": question, "code": None, "result": None, "error": None}

    with st.spinner("🤖 AI is generating code…"):
        try:
            # Convert schema to JSON text for the prompt
            schema_text = schema_to_json(schema)

            # Ask Ollama to write pandas code for the question
            generated_code = ask_llm(schema_text, question)

            # Security check before running
            safe_code = validate_code(generated_code)

            # Execute the code on a copy of the cleaned data
            result = run_code(safe_code, cleaned_df)

            new_entry.update(code=safe_code, result=result)

        except Exception as exc:
            new_entry["error"] = str(exc)

    # Save to conversation history
    st.session_state.conversation.append(new_entry)

    # Remember the last successful result for PDF export
    if new_entry["error"] is None:
        st.session_state.last_result = new_entry

    # Show the result immediately
    render_chat_entry(new_entry)


# ─────────────────────────────────────────────────────────────────────────────
# PDF Download
# ─────────────────────────────────────────────────────────────────────────────

last = st.session_state.get("last_result")
if last:
    st.markdown("---")
    try:
        pdf_bytes = export_pdf(last["question"], last["code"], last["result"])
        st.download_button(
            label="📄 Download result as PDF",
            data=pdf_bytes,
            file_name="analysis-report.pdf",
            mime="application/pdf",
        )
    except RuntimeError as exc:
        st.warning(str(exc))
