"""
home.py
Dashboard landing page. Shows a clickable box for each available evaluation
tool; clicking one navigates to that tool's page (or shows a status message
for tools still in development).
"""

import streamlit as st

st.title("Dashboard")
st.caption("Select an evaluation tool below to get started.")
st.write("")

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.subheader("RAG Evaluation using RAGAS")
        st.write(
            "Evaluate the quality of a RAG system's answers using RAGAS — Response Relevancy, "
            "Faithfulness, Context Precision, and Context Recall."
        )
        if st.button("Open →", key="open_rag_evaluation", type="primary", width="stretch"):
            st.switch_page("app_pages/rag_evaluation.py")

with col2:
    with st.container(border=True):
        st.subheader("Query Generator")
        st.write(
            "Automatically generate test queries and ground-truth answers for your RAG "
            "system, so you don't have to write test cases by hand."
        )
        if st.button("Open →", key="open_query_generator", width="stretch"):
            st.info("🚧 Coming soon! This tool is on our roadmap — check back shortly.")
