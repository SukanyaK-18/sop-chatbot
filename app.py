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

st.set_page_config(page_title="SOP Chatbot", page_icon="📋", layout="wide")

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
    st.session_state.messages = []  # list of {"role": "user"|"assistant", "content": str}

# ---------------------------------------------------------------------------
# Sidebar — document management
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("API Key")
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

st.title("📋 SOP Chatbot")
st.caption("Ask questions about your ingested Standard Operating Procedure documents.")

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

                # Display any images referenced in the chunks used
                displayed_images = set()
                for chunk in result.chunks_used:
                    if "[IMAGE_PATH:" in chunk.text:
                        # Extract image path from the chunk
                        for line in chunk.text.split("\n"):
                            if line.startswith("[IMAGE_PATH:") and line.endswith("]"):
                                img_path = line[len("[IMAGE_PATH:"):-1]
                                if img_path not in displayed_images and os.path.exists(img_path):
                                    st.image(img_path, caption=f"From: {chunk.source}")
                                    displayed_images.add(img_path)

                st.session_state.messages.append({"role": "assistant", "content": response_md})

            except QueryTimeoutError:
                msg = "⚠️ The query timed out. Please try again."
                st.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
            except LLMError as e:
                msg = f"⚠️ LLM error: {e}"
                st.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
