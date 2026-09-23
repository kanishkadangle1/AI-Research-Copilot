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
    st.error("Gemini API configuration error.")
    st.stop()


# ============================================================
# UI
# ============================================================

st.title("AI Research Copilot")
st.write("Research intelligence for discovering, analyzing, and connecting academic knowledge.")


# ============================================================
# ARXIV RETRIEVAL
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

    root = ET.fromstring(response.content)

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

        papers.append({
            "title": (
                title_element.text or ""
            ).strip(),

            "abstract": (
                abstract_element.text or ""
            ).strip()
        })

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

                    time.sleep(wait_time)

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
# INPUT
# ============================================================

topic = st.text_input(
    "Enter Research Topic",
    placeholder="e.g. Large Language Models"
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if st.button(
    "Generate Research Report",
    type="primary"
):

    # --------------------------------------------------------
    # VALIDATE INPUT
    # --------------------------------------------------------

    if not topic.strip():

        st.warning(
            "Please enter a research topic."
        )

        st.stop()


    # --------------------------------------------------------
    # FETCH PAPERS
    # --------------------------------------------------------

    st.info(
        "📄 Fetching papers from arXiv..."
    )

    try:

        papers = fetch_arxiv_papers(
            topic,
            max_results=5
        )

    except Exception:

        st.error(
            "Could not fetch papers from arXiv. "
            "Please try again."
        )

        st.stop()


    if not papers:

        st.warning(
            "No research papers were found."
        )

        st.stop()


    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    st.info(
        "🔎 Running RAG retrieval..."
    )

    top_papers = hybrid_retrieve(
        papers,
        topic,
        top_k=3
    )


    # --------------------------------------------------------
    # BUILD CONTEXT
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are an expert academic research assistant.

Use ONLY the research paper information provided below.

Do not invent papers, authors, statistics,
experimental results, or citations.

Research Topic:
{topic}

Retrieved Papers:
{context}

Generate:

## 1. Literature Review

Give a concise synthesis of the retrieved papers.

## 2. Key Insights

List the major insights.

## 3. Research Gaps

Identify gaps based only on the retrieved papers.

## 4. Future Scope

Suggest possible future research directions.

## 5. Research Questions

Generate exactly 3 research questions.

## 6. Conclusion

Give a short academic conclusion.

Use clear academic language and Markdown.
"""


    # --------------------------------------------------------
    # GEMINI
    # --------------------------------------------------------

    st.info(
        "🤖 AI is analyzing the retrieved papers..."
    )

    try:

        result = generate_with_gemini(
            prompt
        )

    except Exception as e:

        st.error(str(e))
        st.stop()


    # --------------------------------------------------------
    # DISPLAY PAPERS
    # --------------------------------------------------------

    st.subheader(
        "📚 Top Retrieved Papers"
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


    # --------------------------------------------------------
    # DISPLAY AI OUTPUT
    # --------------------------------------------------------

    st.subheader(
        "🧠 AI Research Output"
    )

    st.markdown(
        result
    )
