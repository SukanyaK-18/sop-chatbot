from __future__ import annotations

import os
import tempfile

import streamlit as st

from sop_chatbot.config import ChatbotConfig
from sop_chatbot.index import SOPIndex
from sop_chatbot.ingester import DocumentIngester
from sop_chatbot.models import DocumentNotFoundError, IngestError, LLMError, QueryTimeoutError
from sop_chatbot.query_engine import QueryEngine
from sop_chatbot.session import SessionContext

st.set_page_config(page_title="SOP Chatbot", page_icon="🤖", layout="wide")

# ---------------------------------------------------------------------------
# Custom CSS for styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* Animated bot avatar */
@keyframes bounce {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-8px); }
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}

.bot-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 20px 0;
    margin-bottom: 10px;
}

.bot-avatar {
    font-size: 3.5rem;
    animation: bounce 2s ease-in-out infinite;
    display: inline-block;
}

.bot-title {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #4472C4, #1ABC9C);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}

.bot-subtitle {
    font-size: 0.95rem;
    color: #888;
    margin-top: 4px;
}

/* Status indicator */
.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #1ABC9C;
    animation: pulse 1.5s ease-in-out infinite;
    margin-right: 6px;
}

/* Chat messages styling */
[data-testid="stChatMessage"] {
    border-radius: 12px;
    margin-bottom: 8px;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
}

section[data-testid="stSidebar"] .stMarkdown h1 {
    color: #4472C4;
    font-size: 1.1rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Bot Header with animated avatar
# ---------------------------------------------------------------------------
st.markdown("""
<div class="bot-header">
    <div class="bot-avatar">🤖</div>
    <div>
        <p class="bot-title">SOP Chatbot</p>
        <p class="bot-subtitle"><span class="status-dot"></span>Online — Ask me anything about your SOP documents</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Bootstrap shared components once per session
# ---------------------------------------------------------------------------

@st.cache_resource(ttl=60)  # Refresh every 60 seconds
def init_components():
    config = ChatbotConfig()
    index = SOPIndex(config)
    ingester = DocumentIngester(index, config)
    session = SessionContext(config)
    engine = QueryEngine(index, session, config)
    return ingester, engine, session

ingester, engine, session_ctx = init_components()

SESSION_ID = "ui"

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, Good Day! 👋 How may I help you? Upload an SOP document and ask me anything about it."}
    ]

# ---------------------------------------------------------------------------
# Sidebar — document management
# ---------------------------------------------------------------------------

with st.sidebar:
    # Load API key from Streamlit secrets (set during deployment) or fall back to manual input
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        # Check Streamlit secrets
        try:
            api_key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            api_key = ""

    if not api_key:
        # Only show input if key is not pre-configured
        st.title("🔑 API Key")
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Get a free key at console.groq.com",
        )
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key
            st.success("API key set ✓", icon="✅")
        else:
            st.warning("Enter your Groq API key to enable querying.")
        st.divider()
    else:
        os.environ["GROQ_API_KEY"] = api_key

    st.title("📂 Documents")

    uploaded = st.file_uploader(
        "Upload SOP document",
        type=["txt", "md", "docx"],
        help="Supported formats: .txt, .md, .docx",
    )

    if uploaded and st.button("Ingest", use_container_width=True):
        suffix = os.path.splitext(uploaded.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        try:
            result = ingester.ingest(tmp_path)
            # Rename internally to the original file name
            from sop_chatbot.models import Chunk
            chunks = ingester.get_chunks(os.path.basename(tmp_path))
            if chunks and chunks[0].source != uploaded.name:
                # re-upsert with correct source name
                ingester._index.delete_by_source(os.path.basename(tmp_path))
                renamed = [
                    Chunk(
                        text=c.text,
                        source=uploaded.name,
                        section_id=c.section_id,
                        chunk_index=c.chunk_index,
                    )
                    for c in chunks
                ]
                ingester._index.upsert_chunks(renamed)
            st.success(f"✅ Ingested **{uploaded.name}** — {result.chunk_count} chunks")
        except IngestError as e:
            st.error(str(e))
        finally:
            os.unlink(tmp_path)

    st.divider()
    st.subheader("Ingested documents")
    docs = ingester.list_documents()
    if docs:
        for doc in sorted(docs):
            col1, col2 = st.columns([4, 1])
            col1.write(f"📄 {doc}")
            if col2.button("✕", key=f"remove_{doc}", help=f"Remove {doc}"):
                try:
                    ingester.remove_document(doc)
                    st.rerun()
                except DocumentNotFoundError as e:
                    st.error(str(e))
    else:
        st.caption("No documents ingested yet.")

    st.divider()
    if st.button("🗑️ Clear all documents", use_container_width=True):
        ingester.clear()
        st.rerun()

    if st.button("🔄 Reset conversation", use_container_width=True):
        session_ctx.reset(SESSION_ID)
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

# Main chat area (header already rendered above via HTML)

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your SOPs...", disabled=not api_key):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = engine.query(prompt, SESSION_ID)

                response_md = result.answer

                if result.sources:
                    response_md += f"\n\n*Confidence: {result.confidence_score:.0%}*"

                st.markdown(response_md)
                st.session_state.messages.append({"role": "assistant", "content": response_md})

            except QueryTimeoutError:
                msg = "⚠️ The query timed out. Please try again."
                st.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
            except LLMError as e:
                msg = f"⚠️ LLM error: {e}"
                st.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
