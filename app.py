import time
import requests
import xml.etree.ElementTree as ET

import streamlit as st
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from google import genai


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Research Copilot",
    layout="wide"
)


# ============================================================
# PROFESSIONAL THEME
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #0B1220;
    color: #E5E7EB;
}

h1, h2, h3 {
    color: #F8FAFC;
}

.stTextInput input {
    background-color: #111827;
    color: #F8FAFC;
    border: 1px solid #334155;
    border-radius: 10px;
}

.stButton > button {
    background-color: #2563EB;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #1D4ED8;
}

[data-testid="stExpander"] {
    background-color: #111827;
    border: 1px solid #334155;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# GEMINI CONFIG
# ============================================================

try:

    client = genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"]
    )

except Exception:

    st.error(
        "Gemini API configuration error."
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    "<h1 style='text-align:center;'>AI Research Copilot</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center; color:#94A3B8; font-size:18px;'>"
    "Discover research. Understand evidence. Find what comes next."
    "</p>",
    unsafe_allow_html=True
)


# ============================================================
# ARXIV PAPER RETRIEVAL
# ============================================================

def fetch_arxiv_papers(
    topic,
    max_results=5
):

    url = (
        "https://export.arxiv.org/api/query?"
        f"search_query=all:{topic}"
        f"&start=0"
        f"&max_results={max_results}"
    )

    response = requests.get(
        url,
        timeout=30
    )

    response.raise_for_status()

    root = ET.fromstring(
        response.content
    )

    namespace = {
        "atom": "http://www.w3.org/2005/Atom"
    }

    papers = []

    for entry in root.findall(
        "atom:entry",
        namespace
    ):

        title_element = entry.find(
            "atom:title",
            namespace
        )

        abstract_element = entry.find(
            "atom:summary",
            namespace
        )

        if (
            title_element is None
            or abstract_element is None
        ):
            continue

        title = (
            title_element.text or ""
        ).strip()

        abstract = (
            abstract_element.text or ""
        ).strip()

        papers.append(
            {
                "title": title,
                "abstract": abstract
            }
        )

    return papers


# ============================================================
# TF-IDF RETRIEVAL
# ============================================================

def hybrid_retrieve(
    papers,
    query,
    top_k=3
):

    if not papers:
        return []

    documents = [
        paper["title"] + " " + paper["abstract"]
        for paper in papers
    ]

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    matrix = vectorizer.fit_transform(
        documents + [query]
    )

    query_vector = matrix[-1]

    document_vectors = matrix[:-1]

    scores = (
        document_vectors @ query_vector.T
    ).toarray().flatten()

    top_indices = np.argsort(
        scores
    )[::-1][:top_k]

    return [
        papers[index]
        for index in top_indices
    ]


# ============================================================
# GEMINI GENERATION
# ============================================================

def generate_with_gemini(prompt):

    model_name = "gemini-2.5-flash"

    for attempt in range(3):

        try:

            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )

            if response and response.text:

                return response.text

            raise RuntimeError(
                "Gemini returned an empty response."
            )

        except Exception as e:

            error_message = str(e)

            if (
                "503" in error_message
                or "UNAVAILABLE" in error_message
            ):

                if attempt < 2:

                    wait_time = 2 ** attempt

                    time.sleep(
                        wait_time
                    )

                else:

                    raise RuntimeError(
                        "Gemini is temporarily unavailable. "
                        "Please try again later."
                    )

            else:

                raise RuntimeError(
                    f"Gemini API error: {e}"
                )


# ============================================================
# RESEARCH INPUT
# ============================================================

topic = st.text_input(
    "Research topic",
    placeholder=(
        "e.g. Large language models for healthcare, "
        "AI agents, quantum computing..."
    )
)


# ============================================================
# RESEARCH DEPTH
# ============================================================

research_mode = st.radio(
    "Research depth",
    [
        "Quick",
        "Standard",
        "Deep"
    ],
    horizontal=True
)


# ============================================================
# START RESEARCH
# ============================================================

if st.button(
    "Start Research",
    type="primary"
):

    # ========================================================
    # VALIDATE TOPIC
    # ========================================================

    if not topic.strip():

        st.warning(
            "Please enter a research topic."
        )

        st.stop()


    # ========================================================
    # FETCH PAPERS
    # ========================================================

    st.info(
        f"Research mode: {research_mode} | "
        "Fetching academic papers..."
    )

    try:

        if research_mode == "Quick":

            max_papers = 5

        elif research_mode == "Standard":

            max_papers = 10

        else:

            max_papers = 20

        papers = fetch_arxiv_papers(
            topic,
            max_results=max_papers
        )

    except Exception as e:

        st.error(
            f"Could not fetch papers from arXiv: {e}"
        )

        st.stop()


    # ========================================================
    # CHECK PAPERS
    # ========================================================

    if not papers:

        st.warning(
            "No research papers were found."
        )

        st.stop()


    # ========================================================
    # RETRIEVAL
    # ========================================================

    st.info(
        f"Research mode: {research_mode} | "
        "Analyzing paper relevance..."
    )

    try:

        if research_mode == "Quick":

            top_k = 3

        elif research_mode == "Standard":

            top_k = 5

        else:

            top_k = 10

        top_papers = hybrid_retrieve(
            papers,
            topic,
            top_k=top_k
        )

    except Exception as e:

        st.error(
            f"Retrieval error: {e}"
        )

        st.stop()


    # ========================================================
    # BUILD CONTEXT
    # ========================================================

    context = ""

    for index, paper in enumerate(
        top_papers,
        start=1
    ):

        context += f"""

PAPER {index}

Title:
{paper['title']}

Abstract:
{paper['abstract']}

--------------------------------------------------
"""


        # ========================================================
    # GEMINI PROMPT
    # ========================================================

    prompt = f"""
You are an expert academic research assistant.

Use ONLY the research paper information provided below.

Do not invent papers, authors, statistics,
experimental results, or citations.

Research Topic:
{topic}

Research Mode:
{research_mode}

Retrieved Papers:
{context}

Generate:

## 1. Literature Review

Give a concise synthesis of the retrieved papers.

## 2. Key Insights

List the major insights from the papers.

## 3. Research Gaps

Identify research gaps based ONLY on the
retrieved papers.

## 4. Future Scope

Suggest possible future research directions.
Clearly distinguish suggestions from findings.

## 5. Research Questions

Generate exactly 3 research questions based
on the identified research gaps.

## 6. Conclusion

Give a short academic conclusion.

Use clear academic language and structured Markdown.
"""


    # ========================================================
    # GEMINI GENERATION
    # ========================================================

    st.info(
        "AI is synthesizing the research evidence..."
    )

    try:

        result = generate_with_gemini(
            prompt
        )

    except Exception as e:

        st.error(
            str(e)
        )

        st.stop()


    # ========================================================
    # DISPLAY RETRIEVED PAPERS
    # ========================================================

    st.subheader(
        "Top Retrieved Papers"
    )

    for index, paper in enumerate(
        top_papers,
        start=1
    ):

        with st.expander(
            f"{index}. {paper['title']}"
        ):

            st.write(
                paper["abstract"]
            )


    # ========================================================
    # DISPLAY AI OUTPUT
    # ========================================================

    st.subheader(
        "AI Research Output"
    )

    st.markdown(
        result
    )
