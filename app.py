import json
import os
from datetime import datetime

import streamlit as st
import requests
import xml.etree.ElementTree as ET
import google.generativeai as genai

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="AI Research Copilot",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== UI STYLE =====================
st.markdown("""
<style>

.stApp {
    background: #f4f7ff;
    color: #0f172a;
}

h1 {
    font-size: 38px;
    font-weight: 800;
    color: #0f172a;
}

.subtitle {
    color: #475569;
    font-size: 16px;
    margin-bottom: 15px;
}

input {
    border-radius: 10px !important;
    border: 1px solid #cbd5e1 !important;
    padding: 10px !important;
}

.stButton > button {
    background: #2563eb;
    color: white;
    border-radius: 10px;
    padding: 10px 16px;
    font-weight: 600;
    border: none;
}

section[data-testid="stSidebar"] {
    background: #0b1220;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

.sidebar-card {
    background: rgba(255,255,255,0.08);
    padding: 12px;
    border-radius: 12px;
    margin-bottom: 10px;
}

.paper-card {
    background: white;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    margin-bottom: 10px;
}

.paper-title {
    font-weight: 700;
    color: #0f172a;
}

.paper-text {
    font-size: 13px;
    color: #475569;
}

</style>
""", unsafe_allow_html=True)

# ===================== GEMINI =====================
genai.configure(api_key="GEMINI API KEY")

# FIXED MODEL (IMPORTANT)
model = genai.GenerativeModel("gemini-1.5-flash")

# ===================== ARXIV =====================
def fetch_arxiv_papers(topic, max_results=5):
    url = f"http://export.arxiv.org/api/query?search_query=all:{topic}&start=0&max_results={max_results}"

    response = requests.get(url, timeout=10)
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

# ===================== HISTORY =====================
HISTORY_FILE = "history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_history(entry):
    history = load_history()
    history.append(entry)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

# ===================== LOGIN =====================
if "user" not in st.session_state:
    st.session_state.user = None

if st.session_state.user is None:
    st.title("AI Research Copilot")
    st.write("Login to continue")

    username = st.text_input("Enter your name")

    if st.button("Login"):
        if username:
            st.session_state.user = username.capitalize()
            st.rerun()
        else:
            st.warning("Enter name")

    st.stop()

name = st.session_state.user

# ===================== SIDEBAR =====================
st.sidebar.markdown(f"""
<div class="sidebar-card">
<h3>Hey {name}</h3>
<p>Welcome back</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("History")

history = load_history()

for item in reversed(history[-10:]):
    if st.sidebar.button(item["topic"]):
        st.session_state.selected = item

if "selected" in st.session_state:
    st.subheader("Previous Research")
    st.write(st.session_state.selected["response"])

# ===================== MAIN UI =====================
st.markdown("<h1>AI Research Copilot</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Generate research insights from academic papers</div>", unsafe_allow_html=True)

topic = st.text_input("Enter Research Topic", placeholder="e.g. Generative AI in Education")

# ===================== GENERATE =====================
if st.button("Generate Research Report"):

    if not topic:
        st.warning("Enter topic")
        st.stop()

    st.info("Fetching papers...")

    try:
        papers = fetch_arxiv_papers(topic, 5)
    except Exception as e:
        st.error("Failed to fetch papers")
        st.exception(e)
        st.stop()

    if not papers:
        st.error("No papers found")
        st.stop()

    # LIMIT PAPERS (IMPORTANT FIX)
    papers = papers[:3]

    st.session_state.papers = papers

    research_text = ""

    for p in papers:
        research_text += f"""
Title: {p['title']}
Abstract: {p['abstract']}
-------------------------
"""

    # TRIM PROMPT SIZE (IMPORTANT FIX)
    research_text = research_text[:20000]

    prompt = f"""
You are a research assistant.

Use these papers:

{research_text}

Generate:
1. Literature Review
2. Major Themes
3. Research Gaps
4. Future Scope
5. Research Questions
"""

    st.info("Generating insights...")

    try:
        response = model.generate_content(prompt)
        result_text = response.text

    except Exception as e:
        st.error("Gemini API failed")
        st.exception(e)
        st.stop()

    # ===================== PAPERS =====================
    st.subheader("Research Papers")

    for p in papers:
        st.markdown(f"""
<div class="paper-card">
<div class="paper-title">{p['title']}</div>
<div class="paper-text">{p['abstract'][:300]}...</div>
</div>
""", unsafe_allow_html=True)

    # ===================== OUTPUT =====================
    st.subheader("AI Insights")
    st.write(result_text)

    # ===================== SAVE HISTORY =====================
    save_history({
        "user": name,
        "topic": topic,
        "response": result_text,
        "timestamp": str(datetime.now())
    })

# ===================== CHAT WITH PAPERS =====================
st.subheader("Chat with Papers")

if "papers" in st.session_state:

    question = st.text_input("Ask something about these papers")

    if st.button("Ask AI"):

        context = ""

        for p in st.session_state.papers:
            context += f"""
Title: {p['title']}
Abstract: {p['abstract']}
-------------------
"""

        context = context[:20000]

        chat_prompt = f"""
You are a research assistant.

Answer ONLY using the given papers.

PAPERS:
{context}

QUESTION:
{question}

Give a clear academic answer.
"""

        try:
            chat_response = model.generate_content(chat_prompt)
            st.write(chat_response.text)

        except Exception as e:
            st.error("Chat failed")
            st.exception(e)
