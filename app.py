import json
import os
from datetime import datetime

import streamlit as st
import requests
import xml.etree.ElementTree as ET
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
import google.generativeai as genai


# ===================== STREAMLIT CONFIG =====================
st.set_page_config(
    page_title="AI Research Copilot",
    layout="wide"
)


# ===================== GEMINI SAFE CONFIG =====================
# IMPORTANT: This reads from Streamlit Secrets
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

model = genai.GenerativeModel("gemini-1.5-flash")


# ===================== UI =====================
st.title("AI Research Copilot (RAG Powered)")
st.write("A hybrid retrieval + LLM research assistant")


# ===================== ARXIV FETCH =====================
def fetch_arxiv_papers(topic, max_results=5):
    url = f"http://export.arxiv.org/api/query?search_query=all:{topic}&start=0&max_results={max_results}"
    response = requests.get(url)
    root = ET.fromstring(response.content)

    papers = []

    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        title = entry.find("{http://www.w3.org/2005/Atom}title").text
        summary = entry.find("{http://www.w3.org/2005/Atom}summary").text

        papers.append({
            "title": title.strip(),
            "abstract": summary.strip()
        })

    return papers


# ===================== HYBRID RAG =====================
def hybrid_retrieve(papers, query, top_k=3):

    docs = [p["title"] + " " + p["abstract"] for p in papers]

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(docs + [query])

    query_vec = tfidf_matrix[-1]
    doc_vecs = tfidf_matrix[:-1]

    scores = (doc_vecs @ query_vec.T).toarray().flatten()

    top_indices = np.argsort(scores)[::-1][:top_k]

    return [papers[i] for i in top_indices]


# ===================== INPUT =====================
topic = st.text_input("Enter Research Topic")


# ===================== MAIN FLOW =====================
if st.button("Generate Research Report"):

    if not topic:
        st.warning("Please enter a topic")
        st.stop()

    st.info("Fetching papers...")

    papers = fetch_arxiv_papers(topic, 5)

    st.info("Running RAG retrieval...")

    top_papers = hybrid_retrieve(papers, topic, top_k=3)

    context = ""

    for p in top_papers:
        context += f"""
Title: {p['title']}
Abstract: {p['abstract']}
-------------------
"""

    prompt = f"""
You are an expert research assistant.

Use ONLY the given context.

CONTEXT:
{context}

TASK:
1. Literature Review
2. Key Insights
3. Research Gaps
4. Future Scope
5. 3 Research Questions

Keep output structured and academic.
Do NOT hallucinate.
"""

    st.info("Generating response from Gemini...")

    response = model.generate_content(prompt)

    # ===================== DISPLAY PAPERS =====================
    st.subheader("Top Retrieved Papers")

    for p in top_papers:
        st.markdown(f"""
        **{p['title']}**

        {p['abstract'][:300]}...
        ---
        """)

    # ===================== OUTPUT =====================
    st.subheader("AI Research Output")
    st.write(response.text)
