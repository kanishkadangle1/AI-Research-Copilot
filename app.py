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

/* Title */
h1 {
    font-size: 38px;
    font-weight: 800;
    color: #0f172a;
}

/* Subtitle */
.subtitle {
    color: #475569;
    font-size: 16px;
    margin-bottom: 15px;
}

/* Input */
input {
    border-radius: 10px !important;
    border: 1px solid #cbd5e1 !important;
    padding: 10px !important;
}

/* Button */
.stButton > button {
    background: #2563eb;
    color: white;
    border-radius: 10px;
    padding: 10px 16px;
    font-weight: 600;
    border: none;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0b1220;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Sidebar card */
.sidebar-card {
    background: rgba(255,255,255,0.08);
    padding: 12px;
    border-radius: 12px;
    margin-bottom: 10px;
}

/* Paper card */
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
model = genai.GenerativeModel("gemini-2.5-flash")

# ===================== ARXIV =====================
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

    papers = fetch_arxiv_papers(topic, 5)

    st.session_state.papers = papers  # IMPORTANT for chat

    research_text = ""

    for p in papers:
        research_text += f"""
Title: {p['title']}
Abstract: {p['abstract']}
-------------------------
"""

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

    response = model.generate_content(prompt)

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
    st.write(response.text)

    # ===================== SAVE HISTORY =====================
    save_history({
        "user": name,
        "topic": topic,
        "response": response.text,
        "timestamp": str(datetime.now())
    })

# ===================== CHAT WITH PAPERS (PHASE C) =====================
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

        chat_prompt = f"""
You are a research assistant.

Answer ONLY using the given papers.

PAPERS:
{context}

QUESTION:
{question}

Give a clear academic answer.
"""

        chat_response = model.generate_content(chat_prompt)

        st.write(chat_response.text)
