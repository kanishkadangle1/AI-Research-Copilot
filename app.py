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
# GEMINI CONFIGURATION
# ============================================================

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

except Exception:
    st.error(
        "GEMINI_API_KEY was not found in Streamlit Secrets."
    )
    st.stop()


try:
    client = genai.Client(
        api_key=GEMINI_API_KEY
    )

except Exception as e:
    st.error(
        f"Could not initialize Gemini: {e}"
    )
    st.stop()


# ============================================================
# UI
# ============================================================

st.title("AI Research Copilot")

st.write(
    "A research assistant for discovering and synthesizing academic papers."
)


# ============================================================
# ARXIV PAPER RETRIEVAL
# ============================================================

def fetch_arxiv_papers(topic, max_results=5):

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

    papers = []

    namespace = {
        "atom": "http://www.w3.org/2005/Atom"
    }

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

        title = title_element.text or ""
        abstract = abstract_element.text or ""

        papers.append(
            {
                "title": title.strip(),
                "abstract": abstract.strip()
            }
        )

    return papers


# ============================================================
# TF-IDF PAPER RETRIEVAL
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

    tfidf_matrix = vectorizer.fit_transform(
        documents + [query]
    )

    query_vector = tfidf_matrix[-1]

    document_vectors = tfidf_matrix[:-1]

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

    try:

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        if response is None:
            raise RuntimeError(
                "Gemini returned an empty response."
            )

        if not response.text:
            raise RuntimeError(
                "Gemini returned no text."
            )

        return response.text

    except Exception as e:

        raise RuntimeError(
            f"Gemini API error: {e}"
        )


# ============================================================
# USER INPUT
# ============================================================

topic = st.text_input(
    "Research Topic",
    placeholder="e.g. Large Language Models"
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if st.button(
    "Generate Research Report",
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
        "Fetching academic papers..."
    )

    try:

        papers = fetch_arxiv_papers(
            topic,
            max_results=5
        )

    except requests.exceptions.Timeout:

        st.error(
            "The arXiv service took too long to respond. "
            "Please try again in a moment."
        )

        st.stop()

    except requests.exceptions.RequestException:

        st.error(
            "Could not connect to the academic paper service. "
            "Please try again in a moment."
        )

        st.stop()

    except ET.ParseError:

        st.error(
            "The paper service returned an unexpected response."
        )

        st.stop()

    except Exception:

        st.error(
            "Unable to retrieve research papers right now."
        )

        st.stop()


    # ========================================================
    # CHECK PAPERS
    # ========================================================

    if not papers:

        st.warning(
            "No research papers were found for this topic."
        )

        st.stop()


    # ========================================================
    # RAG RETRIEVAL
    # ========================================================

    st.info(
        "Finding the most relevant papers..."
    )

    try:

        top_papers = hybrid_retrieve(
            papers,
            topic,
            top_k=3
        )

    except Exception:

        st.error(
            "Unable to rank the retrieved papers."
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

Generate exactly 3 research questions based on
the identified research gaps.

## 6. Conclusion

Provide a short academic conclusion.

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
            f"AI generation failed: {e}"
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
