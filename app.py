"""
AI Research Assistant — Streamlit Frontend
Production-grade UI with multi-PDF support, citations, and evaluation metrics.
"""

import time
import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000/api/v1"
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Sora:wght@300;400;600;700&display=swap');

:root {
    --bg: #0d1117;
    --surface: #161b22;
    --border: #30363d;
    --accent: #58a6ff;
    --accent2: #3fb950;
    --warn: #d29922;
    --text: #c9d1d9;
    --muted: #8b949e;
}

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.stApp { background-color: var(--bg); }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

/* Metric cards */
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: var(--accent); }
.metric-value { font-size: 2rem; font-weight: 700; color: var(--accent); }
.metric-label { font-size: 0.8rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; }

/* Citation cards */
.citation-card {
    background: #1c2128;
    border-left: 3px solid var(--accent);
    border-radius: 0 6px 6px 0;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.88rem;
}
.citation-header {
    display: flex;
    justify-content: space-between;
    font-weight: 600;
    color: var(--accent);
    margin-bottom: 6px;
}
.citation-excerpt { color: var(--muted); font-style: italic; }

/* Score badge */
.score-high { color: #3fb950; font-weight: 700; }
.score-med  { color: #d29922; font-weight: 700; }
.score-low  { color: #f85149; font-weight: 700; }

/* Chat bubbles */
.chat-user {
    background: #1f2937;
    border-radius: 12px 12px 2px 12px;
    padding: 12px 16px;
    margin: 8px 0 8px 20%;
    border: 1px solid var(--border);
}
.chat-assistant {
    background: #0d2137;
    border-radius: 12px 12px 12px 2px;
    padding: 12px 16px;
    margin: 8px 20% 8px 0;
    border: 1px solid #1a3a5c;
}

/* Buttons */
.stButton > button {
    background: var(--accent) !important;
    color: #0d1117 !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-family: 'Sora', sans-serif !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* File uploader */
[data-testid="stFileUploadDropzone"] {
    border: 2px dashed var(--border) !important;
    border-radius: 8px !important;
    background: var(--surface) !important;
}

/* Code font */
code { font-family: 'JetBrains Mono', monospace !important; }

/* Header */
.app-header {
    padding: 24px 0 16px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 24px;
}
.app-title {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #58a6ff, #3fb950);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.app-subtitle { color: var(--muted); font-size: 0.9rem; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)


# ── State ─────────────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents" not in st.session_state:
    st.session_state.documents = []
if "last_citations" not in st.session_state:
    st.session_state.last_citations = []
if "last_metrics" not in st.session_state:
    st.session_state.last_metrics = {}


# ── API Helpers ───────────────────────────────────────────────────────────────
def api_get(path: str):
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def api_post(path: str, json=None, files=None, data=None):
    try:
        r = requests.post(f"{API_BASE}{path}", json=json, files=files, data=data, timeout=120)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def api_delete(path: str):
    try:
        r = requests.delete(f"{API_BASE}{path}", timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def refresh_documents():
    result = api_get("/documents/")
    st.session_state.documents = result.get("documents", [])

def get_or_create_session():
    if not st.session_state.session_id:
        result = api_post("/sessions/", json={"name": f"Session {int(time.time())}"})
        st.session_state.session_id = result.get("session_id")
    return st.session_state.session_id

def confidence_color(score: float) -> str:
    if score >= 0.75: return "score-high"
    if score >= 0.50: return "score-med"
    return "score-low"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔬 AI Research Assistant")
    st.markdown("---")

    # Upload
    st.markdown("#### 📄 Upload Documents")
    uploaded = st.file_uploader(
        "Drop PDF files here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    tags_input = st.text_input("Tags (comma-separated)", placeholder="ml, deep-learning")

    if uploaded and st.button("⬆️  Process PDFs", use_container_width=True):
        session_id = get_or_create_session()
        progress = st.progress(0)
        for i, f in enumerate(uploaded):
            with st.spinner(f"Processing {f.name}..."):
                result = api_post(
                    "/documents/upload",
                    files={"file": (f.name, f.getvalue(), "application/pdf")},
                    data={"session_id": session_id, "tags": tags_input},
                )
                if "error" in result:
                    st.error(f"❌ {f.name}: {result['error']}")
                else:
                    st.success(
                        f"✅ {f.name}: {result['chunk_count']} chunks, "
                        f"{result['page_count']} pages "
                        f"({result['processing_time_ms']:.0f}ms)"
                    )
            progress.progress((i + 1) / len(uploaded))
        refresh_documents()

    st.markdown("---")

    # Document list
    st.markdown("#### 📚 Indexed Documents")
    if st.button("🔄 Refresh", use_container_width=True):
        refresh_documents()

    docs = st.session_state.documents
    if docs:
        for doc in docs:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"📄 `{doc['filename'][:22]}...`" if len(doc['filename']) > 25 else f"📄 `{doc['filename']}`")
            with col2:
                if st.button("🗑", key=f"del_{doc['doc_id']}", help="Delete"):
                    result = api_delete(f"/documents/{doc['doc_id']}")
                    if "error" not in result:
                        st.success("Deleted")
                        refresh_documents()
                        st.rerun()
    else:
        st.markdown('<p style="color:#8b949e;font-size:0.85rem;">No documents indexed yet.</p>', unsafe_allow_html=True)

    st.markdown("---")

    # Settings
    st.markdown("#### ⚙️ Settings")
    top_k = st.slider("Top-K Retrieval", 1, 10, 5)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)
    show_eval = st.checkbox("Show Evaluation Metrics", value=True)

    if st.button("🆕 New Session", use_container_width=True):
        st.session_state.session_id = None
        st.session_state.messages = []
        st.session_state.last_citations = []
        st.session_state.last_metrics = {}
        st.rerun()


# ── Main Area ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div class="app-title">AI Research Assistant</div>
  <div class="app-subtitle">RAG-powered Q&A · Semantic Search · Source Citations · Confidence Scoring</div>
</div>
""", unsafe_allow_html=True)

tab_chat, tab_eval, tab_docs = st.tabs(["💬 Chat", "📊 Evaluation", "🗂 Documents"])


# ── TAB 1: Chat ───────────────────────────────────────────────────────────────
with tab_chat:
    # Chat history
    chat_col, info_col = st.columns([2, 1])

    with chat_col:
        st.markdown("#### Conversation")
        chat_container = st.container(height=420)
        with chat_container:
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.markdown(f'<div class="chat-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-assistant">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

        # Input
        with st.form("chat_form", clear_on_submit=True):
            col_q, col_b = st.columns([5, 1])
            with col_q:
                question = st.text_input("Ask a question...", label_visibility="collapsed",
                                         placeholder="What are the key findings in the uploaded papers?")
            with col_b:
                submitted = st.form_submit_button("Send →")

        if submitted and question.strip():
            session_id = get_or_create_session()
            st.session_state.messages.append({"role": "user", "content": question})

            with st.spinner("Thinking..."):
                payload = {
                    "question": question,
                    "session_id": session_id,
                    "top_k": top_k,
                    "temperature": temperature,
                }
                result = api_post("/chat/", json=payload)

            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                answer = result.get("answer", "")
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.session_state.last_citations = result.get("citations", [])
                st.session_state.last_metrics = {
                    "confidence_score": result.get("confidence_score", 0),
                    "confidence_label": result.get("confidence_label", ""),
                    "retrieved_chunks": result.get("retrieved_chunks", 0),
                    "processing_time_ms": result.get("processing_time_ms", 0),
                    "model_used": result.get("model_used", ""),
                }
                st.rerun()

    with info_col:
        # Metrics panel
        if st.session_state.last_metrics:
            m = st.session_state.last_metrics
            score = m.get("confidence_score", 0)
            css_class = confidence_color(score)
            st.markdown("#### Last Response")
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-value {css_class}">{score:.0%}</div>
              <div class="metric-label">Confidence</div>
            </div>""", unsafe_allow_html=True)
            st.markdown(f"**Label:** {m.get('confidence_label', 'N/A')}")
            st.markdown(f"**Sources:** {m.get('retrieved_chunks', 0)} chunks")
            st.markdown(f"**Time:** {m.get('processing_time_ms', 0):.0f}ms")
            st.markdown(f"**Model:** `{m.get('model_used', 'N/A')}`")

        # Citations panel
        if st.session_state.last_citations:
            st.markdown("#### Citations")
            for i, c in enumerate(st.session_state.last_citations, 1):
                score = c["relevance_score"]
                css_class = confidence_color(score)
                st.markdown(f"""
                <div class="citation-card">
                  <div class="citation-header">
                    <span>Source {i} · {c['filename']}</span>
                    <span class="{css_class}">{score:.0%}</span>
                  </div>
                  <div>📄 Page {c.get('page_number', '?')}</div>
                  <div class="citation-excerpt">"{c['excerpt'][:180]}..."</div>
                </div>""", unsafe_allow_html=True)


# ── TAB 2: Evaluation ─────────────────────────────────────────────────────────
with tab_eval:
    st.markdown("#### Retrieval Quality Evaluation")
    st.markdown(
        "Run a question through the RAG pipeline and inspect retrieval metrics "
        "(MRR, Precision@K, Recall@K, NDCG)."
    )

    with st.form("eval_form"):
        eval_q = st.text_area("Evaluation Question", height=80,
                               placeholder="Enter a question to evaluate retrieval quality...")
        eval_k = st.slider("Top-K", 1, 10, 5, key="eval_k")
        run_eval = st.form_submit_button("▶ Run Evaluation")

    if run_eval and eval_q.strip():
        with st.spinner("Running evaluation..."):
            result = api_post("/evaluation/", json={
                "question": eval_q,
                "top_k": eval_k,
            })

        if "error" in result:
            st.error(result["error"])
        else:
            metrics = result.get("retrieval_metrics", {})

            # Metric grid
            c1, c2, c3, c4 = st.columns(4)
            cards = [
                (c1, "MRR", metrics.get("mrr", 0), "Mean Reciprocal Rank"),
                (c2, "P@K", metrics.get("precision_at_k", 0), "Precision@K"),
                (c3, "R@K", metrics.get("recall_at_k", 0), "Recall@K"),
                (c4, "NDCG", metrics.get("ndcg", 0), "Norm. DCG"),
            ]
            for col, name, val, desc in cards:
                with col:
                    css = confidence_color(val)
                    st.markdown(f"""
                    <div class="metric-card">
                      <div class="metric-value {css}">{val:.3f}</div>
                      <div class="metric-label">{name}</div>
                      <div style="font-size:0.75rem;color:#8b949e">{desc}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("---")
            col_left, col_right = st.columns([1, 1])

            with col_left:
                # Radar chart
                cats = ["MRR", "Precision@K", "Recall@K", "NDCG", "Avg Relevance"]
                vals = [
                    metrics.get("mrr", 0),
                    metrics.get("precision_at_k", 0),
                    metrics.get("recall_at_k", 0),
                    metrics.get("ndcg", 0),
                    metrics.get("avg_relevance_score", 0),
                ]
                fig = go.Figure(go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=cats + [cats[0]],
                    fill="toself",
                    fillcolor="rgba(88,166,255,0.15)",
                    line=dict(color="#58a6ff", width=2),
                ))
                fig.update_layout(
                    polar=dict(
                        bgcolor="#161b22",
                        radialaxis=dict(range=[0, 1], gridcolor="#30363d",
                                        tickfont=dict(color="#8b949e")),
                        angularaxis=dict(gridcolor="#30363d",
                                         tickfont=dict(color="#c9d1d9")),
                    ),
                    paper_bgcolor="#0d1117",
                    plot_bgcolor="#0d1117",
                    margin=dict(t=30, b=30),
                    height=320,
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_right:
                # Citation relevance bar
                cits = result.get("citations", [])
                if cits:
                    df = pd.DataFrame([{
                        "Source": f"{c['filename'][:20]} p.{c.get('page_number','?')}",
                        "Relevance": c["relevance_score"],
                    } for c in cits])
                    fig2 = px.bar(df, x="Relevance", y="Source", orientation="h",
                                  color="Relevance",
                                  color_continuous_scale=["#f85149", "#d29922", "#3fb950"],
                                  range_color=[0, 1])
                    fig2.update_layout(
                        paper_bgcolor="#0d1117",
                        plot_bgcolor="#161b22",
                        font=dict(color="#c9d1d9"),
                        margin=dict(t=10, b=10),
                        height=320,
                        showlegend=False,
                        coloraxis_showscale=False,
                    )
                    st.plotly_chart(fig2, use_container_width=True)

            # Answer
            st.markdown("**Generated Answer:**")
            st.info(result.get("answer", ""))

            st.markdown(f"""
            **Confidence:** `{result.get('answer_confidence', 0):.1%}` · 
            **Latency:** `{metrics.get('latency_ms', 0):.0f}ms` · 
            **Retrieved:** `{metrics.get('retrieved_count', 0)} chunks`
            """)


# ── TAB 3: Documents ──────────────────────────────────────────────────────────
with tab_docs:
    st.markdown("#### Indexed Document Library")
    if st.button("🔄 Refresh Library"):
        refresh_documents()

    docs = st.session_state.documents
    if not docs:
        st.info("No documents indexed yet. Upload PDFs using the sidebar.")
    else:
        df = pd.DataFrame([{
            "Filename": d["filename"],
            "Doc ID": d["doc_id"][:12] + "...",
            "Session": d.get("session_id", "")[:8] or "—",
            "Tags": d.get("tags", "") or "—",
        } for d in docs])
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Document deep-dive
        st.markdown("---")
        st.markdown("#### 🔍 Document Preview")
        doc_options = {d["filename"]: d["doc_id"] for d in docs}
        selected_name = st.selectbox("Select document", list(doc_options.keys()))
        if selected_name:
            doc_id = doc_options[selected_name]
            chunks_result = api_get(f"/documents/{doc_id}/chunks?limit=5&offset=0")
            if "error" not in chunks_result:
                st.markdown(f"Showing first 5 chunks for **{selected_name}**")
                for i, chunk in enumerate(chunks_result.get("chunks", []), 1):
                    with st.expander(f"Chunk {i} — Page {chunk['metadata'].get('page_number', '?')} ({chunk['metadata'].get('char_count', '?')} chars)"):
                        st.text(chunk["text"])
