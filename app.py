"""
app.py - Streamlit UI for RAG Bot

Run:
    streamlit run app.py
"""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

import llm
import rag

load_dotenv()

st.set_page_config(page_title="RAG Bot", page_icon="🤖", layout="wide")
st.title("RAG Bot - Any Model, Free Tools")
st.caption("Upload your documents, pick any free model, ask questions grounded in your docs.")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("1. Choose a model")

    provider_names = list(llm.PROVIDERS.keys())
    provider_labels = {k: llm.PROVIDERS[k]["label"] for k in provider_names}

    provider = st.selectbox(
        "Provider",
        options=provider_names,
        index=provider_names.index("groq") if "groq" in provider_names else 0,
        format_func=lambda k: provider_labels[k],
    )

    models = llm.PROVIDERS[provider]["models"]
    model = st.selectbox("Model", options=models)

    # Also allow a custom model id (useful for Ollama / OpenRouter)
    custom_model = st.text_input("Or type any model id (optional)", value="")
    if custom_model.strip():
        model = custom_model.strip()

    api_key = None
    if llm.PROVIDERS[provider].get("requires_key"):
        env_var = llm.PROVIDERS[provider].get("env_var", "")
        existing = os.getenv(env_var, "")
        api_key = st.text_input(
            f"{llm.PROVIDERS[provider]['label']} API Key",
            value="",
            type="password",
            placeholder=f"Paste key or set {env_var} in .env",
            help=f"You can also set {env_var} in a .env file (see .env.example).",
        )
        if not api_key and existing:
            api_key = existing
            st.success(f"Using {env_var} from .env")
    else:
        st.info("Ollama runs locally — no API key needed. Make sure `ollama serve` is running.")

    st.divider()
    st.header("2. Add documents")

    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, MD, or DOCX",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
    )

    if st.button("📥 Process documents", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Upload at least one file first.")
        else:
            total_chunks = 0
            with st.spinner("Reading and indexing documents..."):
                for f in uploaded_files:
                    try:
                        text = rag.load_document(f.getvalue(), f.name)
                        if not text.strip():
                            st.warning(f"No text found in {f.name}")
                            continue
                        n = rag.add_document(text, f.name)
                        total_chunks += n
                        st.success(f"{f.name}: {n} chunks")
                    except Exception as exc:
                        st.error(f"{f.name}: {exc}")
            st.success(f"Done. Added {total_chunks} chunks.")

    doc_count = rag.count_documents()
    st.metric("Chunks in database", doc_count)
    if doc_count == 0:
        st.warning("No documents indexed yet. Chat will work, but without document context.")

    if st.button("🗑️ Clear database", use_container_width=True):
        rag.clear_collection()
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Free tools: Streamlit + ChromaDB + pypdf. Models via Ollama / Groq / Gemini / OpenRouter.")

# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources used"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**{i}. {src['source']}**")
                    st.write(src["text"][:800])

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve context
    sources = []
    context_text = ""
    if rag.count_documents() > 0:
        try:
            sources = rag.search(prompt, n_results=4)
            if sources:
                context_text = "\n\n---\n\n".join(
                    f"[Source: {s['source']}]\n{s['text']}" for s in sources
                )
        except Exception as exc:
            st.error(f"Search failed: {exc}")
    else:
        st.warning("No documents uploaded yet — answering without document context. Upload files in the sidebar for RAG answers.")

    system_prompt = (
        "You are a helpful assistant. Answer using ONLY the provided context when it is relevant. "
        "If the context does not contain the answer, say you don't know based on the documents. "
        "Be concise and cite the source name when you use it."
    )

    if context_text:
        full_prompt = f"Context from documents:\n\n{context_text}\n\nQuestion: {prompt}\n\nAnswer:"
    else:
        full_prompt = prompt

    with st.chat_message("assistant"):
        with st.spinner(f"Thinking with {provider} / {model}..."):
            try:
                answer = llm.chat(
                    provider=provider,
                    model=model,
                    prompt=full_prompt,
                    api_key=api_key,
                    system=system_prompt,
                )
            except Exception as exc:
                answer = f"⚠️ {exc}"
        st.markdown(answer)
        if sources:
            with st.expander("Sources used"):
                for i, src in enumerate(sources, 1):
                    st.markdown(f"**{i}. {src['source']}**")
                    st.write(src["text"][:800])

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
