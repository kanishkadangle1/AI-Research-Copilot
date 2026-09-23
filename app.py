import os
import requests
import xml.etree.ElementTree as ET

import streamlit as st
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from google import genai


# ===================== STREAMLIT CONFIG =====================

st.set_page_config(
    page_title="AI Research Copilot",
    layout="wide"
)


# ===================== GEMINI CONFIG =====================

client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)


# ===================== UI =====================

st.title("AI Research Copilot (RAG Powered)")
st.write("A hybrid retrieval + LLM research assistant")


# ===================== ARXIV FETCH =====================

def fetch_arxiv_papers(topic, max_results=5):

    url = (
        f"http://export.arxiv.org/api/query?"
        f"search_query=all:{topic}"
        f"&start=0"
        f"&max_results={max_results}"
    )

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    root = ET.fromstring(response.content)

    papers = []

    for entry in root.findall(
        "{http://www.w3.org/2005/Atom}entry"
    ):

        title = entry.find(
            "{http://www.w3.org/2005/Atom}title"
        ).text

        summary = entry.find(
            "{http://www.w3.org/2005/Atom}summary"
        ).text

        papers.append({
            "title": title.strip(),
            "abstract": summary.strip()
        })

    return papers


# ===================== HYBRID RAG =====================

def hybrid_retrieve(papers, query, top_k=3):

    if not papers:
        return []

    docs = [
        p["title"] + " " + p["abstract"]
        for p in papers
    ]

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    tfidf_matrix = vectorizer.fit_transform(
        docs + [query]
    )

    query_vec = tfidf_matrix[-1]
    doc_vecs = tfidf_matrix[:-1]

    scores = (
        doc_vecs @ query_vec.T
    ).toarray().flatten()

    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        papers[i]
        for i in top_indices
    ]


# ===================== INPUT =====================

topic = st.text_input(
    "Enter Research Topic",
    placeholder="e.g. Large Language Models"
)


# ===================== MAIN FLOW =====================

if st.button("Generate Research Report"):

    if not topic.strip():
        st.warning("Please enter a research topic.")
        st.stop()

    # -------------------------
    # FETCH PAPERS
    # -------------------------

    st.info("Fetching papers...")

    try:
        papers = fetch_arxiv_papers(
            topic,
            max_results=5
        )
    except Exception as e:
        st.error(
            f"Could not fetch papers from arXiv: {e}"
        )
        st.stop()

    if not papers:
        st.warning(
            "No research papers were found for this topic."
        )
        st.stop()


    # -------------------------
    # RAG RETRIEVAL
    # -------------------------

    st.info("Running RAG retrieval...")

    top_papers = hybrid_retrieve(
        papers,
        topic,
        top_k=3
    )

    context = ""

    for p in top_papers:

        context += f"""
Title: {p['title']}

Abstract:
{p['abstract']}

-------------------
"""


    # -------------------------
    # PROMPT
    # -------------------------

    prompt = f"""
You are an expert academic research assistant.

Use ONLY the research paper information provided
in the context below.

Do not invent papers, findings, statistics,
authors, citations, or claims that are not supported
by the provided context.

CONTEXT:
{context}

USER RESEARCH TOPIC:
{topic}

TASK:

1. Literature Review
Provide a concise synthesis of the retrieved papers.

2. Key Insights
Identify the major findings, approaches, and themes.

3. Research Gaps
Identify limitations or areas that appear insufficiently
explored based only on the provided papers.

4. Future Scope
Suggest reasonable future research directions.

5. Research Questions
Generate 3 research questions based on the identified gaps.

Keep the response structured, academic, and clear.
"""


    # -------------------------
    # GEMINI
    # -------------------------

    st.info("Generating response from Gemini...")

    try:

        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt
        )

    except Exception as e:

        st.error(
            f"Gemini API error: {e}"
        )

        st.stop()


    # -------------------------
    # DISPLAY PAPERS
    # -------------------------

    st.subheader("Top Retrieved Papers")

    for p in top_papers:

        st.markdown(
            f"""
            **{p['title']}**

            {p['abstract'][:500]}...

            ---
            """
        )


    # -------------------------
    # OUTPUT
    # -------------------------

    st.subheader("AI Research Output")

    st.write(response.text)
