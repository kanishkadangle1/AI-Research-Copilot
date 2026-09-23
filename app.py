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
# GEMINI CONFIG
# ============================================================

try:
    client = genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"]
    )

except Exception as e:
    st.error(
        "Gemini API key is not configured correctly. "
        "Please check Streamlit Secrets."
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
# ARXIV FETCH
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

    atom_namespace = {
        "atom": "http://www.w3.org/2005/Atom"
    }

    for entry in root.findall(
        "atom:entry",
        atom_namespace
    ):

        title_element = entry.find(
            "atom:title",
            atom_namespace
        )

        summary_element = entry.find(
            "atom:summary",
            atom_namespace
        )

        if title_element is None or summary_element is None:
            continue

        title = title_element.text or ""
        summary = summary_element.text or ""

        papers.append({
            "title": title.strip(),
            "abstract": summary.strip()
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
        p["title"] + " " + p["abstract"]
        for p in papers
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

    retrieved_papers = [
        papers[i]
        for i in top_indices
    ]

    return retrieved_papers


# ============================================================
# GEMINI GENERATION WITH RETRY + FALLBACK
# ============================================================

def generate_with_gemini(prompt):

    # Primary model + fallback models.
    #
    # If a model becomes temporarily unavailable,
    # the application will try the next model.

    models = [
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash"
    ]

    last_error = None

    for model_name in models:

        st.write(
            f"Using Gemini model: `{model_name}`"
        )

        # Retry each model up to 3 times
        for attempt in range(3):

            try:

                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )

                # Make sure Gemini returned actual text.
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

                last_error = e

                error_message = str(e)

                # ------------------------------------------------
                # TEMPORARY 503 / UNAVAILABLE
                # ------------------------------------------------

                if (
                    "503" in error_message
                    or "UNAVAILABLE" in error_message
                ):

                    # Exponential backoff:
                    #
                    # Attempt 1 -> 2 seconds
                    # Attempt 2 -> 4 seconds
                    # Attempt 3 -> 8 seconds

                    wait_time = 2 ** attempt

                    if attempt < 2:

                        st.warning(
                            f"{model_name} is temporarily "
                            f"unavailable. Retrying in "
                            f"{wait_time} seconds..."
                        )

                        time.sleep(
                            wait_time
                        )

                    else:

                        st.warning(
                            f"{model_name} is still unavailable "
                            "after multiple attempts. "
                            "Trying another Gemini model..."
                        )

                else:

                    # For errors such as invalid API keys,
                    # invalid requests, permission errors, etc.,
                    # retrying will not normally solve the issue.

                    raise RuntimeError(
                        f"Gemini API error: {e}"
                    )

    # ========================================================
    # ALL MODELS FAILED
    # ========================================================

    raise RuntimeError(
        "Gemini is temporarily unavailable. "
        "All configured models were unavailable. "
        f"Last error: {last_error}"
    )


# ============================================================
# INPUT
# ============================================================

topic = st.text_input(
    "Enter Research Topic",
    placeholder="e.g. Large Language Models"
)


# ============================================================
# MAIN FLOW
# ============================================================

if st.button(
    "Generate Research Report",
    type="primary"
):

    # ========================================================
    # VALIDATE INPUT
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
        "📄 Fetching research papers from arXiv..."
    )

    try:

        papers = fetch_arxiv_papers(
            topic,
            max_results=5
        )

    except requests.exceptions.Timeout:

        st.error(
            "The arXiv request timed out. "
            "Please try again."
        )

        st.stop()

    except requests.exceptions.RequestException as e:

        st.error(
            f"Could not connect to arXiv: {e}"
        )

        st.stop()

    except ET.ParseError:

        st.error(
            "arXiv returned an unexpected response. "
            "Please try again."
        )

        st.stop()

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
            "No research papers were found for "
            f"the topic: {topic}"
        )

        st.info(
            "Try a broader research topic such as "
            "'Large Language Models', 'Artificial Intelligence', "
            "or 'Computer Vision'."
        )

        st.stop()


    # ========================================================
    # RAG RETRIEVAL
    # ========================================================

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
    # PROMPT
    # ========================================================

    prompt = f"""
You are an expert academic research assistant.

Your task is to analyze the research papers provided
in the context and produce a structured research synthesis.

IMPORTANT RULES:

1. Use ONLY the information contained in the provided context.
2. Do not invent papers.
3. Do not invent authors.
4. Do not invent statistics.
5. Do not invent experimental results.
6. Do not create citations that are not present in the context.
7. Clearly distinguish between information supported by the
   papers and reasonable future research suggestions.
8. If the available papers are insufficient to support a claim,
   explicitly say that the available evidence is insufficient.

RESEARCH TOPIC:

{topic}


RETRIEVED RESEARCH PAPERS:

{context}


TASK:

## 1. Literature Review

Provide a concise academic synthesis of the retrieved papers.

Explain:

- What the papers study
- The main approaches used
- The common themes
- Important differences between the papers


## 2. Key Insights

Identify the most important insights from the retrieved papers.

Use clear bullet points.


## 3. Research Gaps

Identify research gaps or limitations that can reasonably
be inferred from the provided papers.

Do not invent unsupported gaps.


## 4. Future Scope

Suggest possible future research directions.

Clearly distinguish these suggestions from findings
reported in the papers.


## 5. Research Questions

Generate exactly 3 research questions based on the
identified research gaps.


## 6. Short Conclusion

Provide a short academic conclusion summarizing
the overall research landscape.


Keep the response:

- Academic
- Structured
- Clear
- Concise
- Evidence-based
- Free from hallucinated information
"""


    # ========================================================
    # GEMINI GENERATION
    # ========================================================

    st.info(
        "🤖 AI is analyzing the retrieved research papers..."
    )

    try:

        result = generate_with_gemini(
            prompt
        )

    except Exception as e:

        st.error(
            str(e)
        )

        st.info(
            "Please wait a few moments and try again. "
            "Temporary Gemini capacity issues can occur."
        )

        st.stop()


    # ========================================================
    # DISPLAY RETRIEVED PAPERS
    # ========================================================

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


    # ========================================================
    # DISPLAY AI OUTPUT
    # ========================================================

    st.subheader(
        "🧠 AI Research Output"
    )

    st.markdown(
        result
    )
