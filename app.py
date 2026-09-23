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
    page_icon="📚",
    layout="wide"
)


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

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

st.title("📚 AI Research Copilot")
st.write(
    "A hybrid retrieval + LLM research assistant"
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

        papers.append({
            "title": title.strip(),
            "abstract": abstract.strip()
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

    model_name = "gemini-2.5-flash"

    st.info(
        f"Using Gemini model: `{model_name}`"
    )

    # Try up to 3 times if Gemini temporarily returns 503
    for attempt in range(3):

        try:

            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )

            if response is None or not response.text:

                raise RuntimeError(
                    "Gemini returned an empty response."
                )

            return response.text

        except Exception as e:

            error_message = str(e)

            if (
                "503" in error_message
                or "UNAVAILABLE" in error_message
            ):

                if attempt < 2:

                    wait_time = 2 ** attempt

                    st.warning(
                        "Gemini is temporarily busy. "
                        f"Retrying in {wait_time} seconds..."
                    )

                    time.sleep(wait_time)

                else:

                    raise RuntimeError(
                        "Gemini is temporarily unavailable. "
                        "Please try again in a few minutes."
                    )

            else:

                raise RuntimeError(
                    f"Gemini API error: {e}"
                )


# ============================================================
# USER INPUT
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

    except Exception as e:

        st.error(
            f"Could not fetch papers from arXiv: {e}"
        )

        st.stop()


    if not papers:

        st.warning(
            "No research papers were found."
        )

        st.stop()


    # --------------------------------------------------------
    # RAG RETRIEVAL
    # --------------------------------------------------------

    st.info(
        "🔎 Running RAG retrieval..."
    )

    try:

        top_papers = hybrid_retrieve(
            papers,
            topic,
            top_k=3
        )

    except Exception as e:

        st.error(
            f"Retrieval error: {e}"
        )

        st.stop()


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

Do not invent:
- papers
- authors
- statistics
- results
- citations
- facts

If the papers do not provide enough evidence,
clearly say so.

Research Topic:
{topic}

Retrieved Papers:
{context}

Create a structured academic research report with:

## 1. Literature Review

Summarize the main research themes,
approaches, and findings.

## 2. Key Insights

List the major insights from the papers.

## 3. Research Gaps

Identify possible gaps based only on
the retrieved papers.

## 4. Future Scope

Suggest reasonable future research directions.

## 5. Research Questions

Generate exactly 3 research questions.

## 6. Conclusion

Give a short academic conclusion.

Keep the response clear, concise,
academic, and evidence-based.
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

        st.error(
            str(e)
        )

        st.stop()


    # --------------------------------------------------------
    # DISPLAY PAPERS
    # --------------------------------------------------------

# ============================================================
# DISPLAY RETRIEVED PAPERS
# ============================================================

st.subheader("📚 Top Retrieved Papers")

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


# ============================================================
# DISPLAY AI OUTPUT
# ============================================================

st.subheader("🧠 AI Research Output")

st.markdown(
    result
)
