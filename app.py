"""
app.py
Entry point / router. Defines the dashboard (home) page and the RAG
evaluation page, and wires up navigation between them.
"""

import streamlit as st

st.set_page_config(page_title="RAGAS Evaluation", layout="wide")

pg = st.navigation(
    [
        st.Page("app_pages/home.py", title="Dashboard", url_path="dashboard", default=True),
        st.Page(
            "app_pages/rag_evaluation.py",
            title="RAG Evaluation using RAGAS",
            url_path="rag-evaluation",
            visibility="hidden",
        ),
    ]
)
pg.run()
