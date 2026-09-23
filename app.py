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

        papers.append(
            {
                "title": title.strip(),
                "abstract": abstract.strip()
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
# GET AVAILABLE GEMINI MODELS
# ============================================================

def get_available_gemini_models():

    try:

        available_models = []

        for model in client.models.list():

            model_name = model.name

            if model_name.startswith("models/"):
                model_name = model_name.replace(
                    "models/",
                    "",
                    1
                )

            # Store every model returned by the API.
            # We will inspect the names and prefer Flash.
            available_models.append(
                model_name
            )

        return available_models

    except Exception as e:

        st.error(
            f"Could not retrieve Gemini models: {e}"
        )

        return []


# ============================================================
# SELECT GEMINI MODEL
# ============================================================

def select_gemini_model():

    models = get_available_gemini_models()

    if not models:
        return None

    # Prefer Flash models because they are generally
    # appropriate for a fast research assistant.

    flash_models = [
        model
        for model in models
        if "flash" in model.lower()
    ]

    if flash_models:

        # Sort so newer-looking model names
        # are considered first.

        flash_models.sort(
            reverse=True
        )

        return flash_models[0]

    # If no Flash model is available,
    # use the first model returned by Gemini.

    return models[0]


# ============================================================
# GEMINI GENERATION
# ============================================================

def generate_with_gemini(prompt):

    model_name = select_gemini_model()

    if not model_name:

        raise RuntimeError(
            "No Gemini model was found for your API key."
        )

    st.info(
        f"Using Gemini model: `{model_name}`"
    )

    max_attempts = 4

    for attempt in range(max_attempts):

        try:

            response = client.models.generate_content(
                model=model_name,
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

            error_message = str(e)

            # ==================================================
            # TEMPORARY 503 ERROR
            # ==================================================

            if (
                "503" in error_message
                or "UNAVAILABLE" in error_message
            ):

                if attempt < max_attempts - 1:

                    wait_time = 2 ** attempt

                    st.warning(
                        "Gemini is temporarily unavailable. "
                        f"Retrying in {wait_time} seconds..."
                    )

                    time.sleep(
                        wait_time
                    )

                else:

                    raise RuntimeError(
                        "Gemini is currently experiencing "
                        "high demand or temporary capacity "
                        "limitations. Please try again later."
                    )

            else:

                raise RuntimeError(
                    f"Gemini API error: {e}"
                )


# ============================================================
# OPTIONAL MODEL DIAGNOSTIC
# ============================================================

with st.expander(
    "🔧 Gemini API Diagnostics"
):

    st.write(
        "Use this section to check which Gemini "
        "models are available to your API key."
    )

    if st.button(
        "Check Available Gemini Models"
    ):

        models = get_available_gemini_models()

        if models:

            st.success(
                "Gemini models available to your API key:"
            )

            for model in models:

                st.write(
                    f"- `{model}`"
                )

        else:

            st.error(
                "No Gemini models were returned."
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
        "📄 Fetching papers from arXiv..."
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
            f"Could not fetch papers: {e}"
        )

        st.stop()


    # ========================================================
    # CHECK PAPER RESULTS
    # ========================================================

    if not papers:

        st.warning(
            f"No research papers were found for: {topic}"
        )

        st.info(
            "Try a broader topic, for example: "
            "'Large Language Models', "
            "'Artificial Intelligence', or "
            "'Computer Vision'."
        )

        st.stop()


    # ========================================================
    # RETRIEVAL
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
    # GEMINI PROMPT
    # ========================================================

    prompt = f"""
You are an expert academic research assistant.

You are helping a student or researcher understand
the current research landscape around a topic.

IMPORTANT RULES:

1. Use ONLY the research paper information provided
   in the context below.

2. Do NOT invent papers.

3. Do NOT invent authors.

4. Do NOT invent statistics.

5. Do NOT invent experimental results.

6. Do NOT create fake citations.

7. If the retrieved papers do not provide enough
   evidence for a statement, clearly say so.

8. Separate existing findings from your suggested
   future research directions.

RESEARCH TOPIC:

{topic}


RETRIEVED PAPERS:

{context}


Please produce the following:


## 1. Literature Review

Give a concise academic synthesis of the retrieved papers.

Discuss:

- Main research areas
- Common themes
- Approaches used
- Important differences between the papers


## 2. Key Insights

List the major insights from the retrieved papers.


## 3. Research Gaps

Identify possible research gaps based ONLY on the
provided papers.


## 4. Future Scope

Suggest possible future research directions.

Clearly indicate that these are suggestions rather
than findings directly reported by the papers.


## 5. Research Questions

Generate exactly 3 research questions based on
the identified research gaps.


## 6. Conclusion

Provide a short academic conclusion.


Use clear academic language and structured Markdown.
"""


    # ========================================================
    # GENERATE AI RESPONSE
    # ========================================================

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

        st.info(
            "Please wait a little while and try again."
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
