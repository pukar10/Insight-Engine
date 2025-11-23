"""
app.py - Streamlit UI for Insight Engine.

Run this with:
    streamlit run app/app.py
"""

import streamlit as st

from backend.search import search

# Try to import optional chatty mode (local LLM via Ollama).
# If this fails (e.g. ollama not installed), we just disable that feature.
try:
    from backend.answer import answer_question
    HAS_LLM = True
except Exception:
    HAS_LLM = False

# Basic Streamlit page config (title, favicon etc.)
st.set_page_config(page_title="Insight Engine", page_icon="🧠")

st.title("🧠 Insight Engine")

st.write(
    "Search and explore your own notes, PDFs and text files.\n\n"
    "How to use this:\n"
    "1. Put some `.txt`, `.md` or `.pdf` files into the `data/` folder.\n"
    "2. Run `python backend/ingest.py` to build the index.\n"
    "3. Then come back here and ask questions about your documents."
)

# Text input for the user's question
question = st.text_input(
    "Ask a question about your documents:",
    placeholder="Example: What did I write about job offers?",
)

# Slider to control how many chunks we retrieve
num_results = st.slider(
    "How many snippets to retrieve?",
    min_value=1,
    max_value=10,
    value=5,
)

# Toggle for using the local LLM (if available)
use_llm = st.checkbox(
    "Generate a full answer using a local LLM (Ollama)",
    value=False,
    help="Requires Ollama installed and running, and a model pulled (e.g. `ollama pull llama3.2`).",
)

if use_llm and not HAS_LLM:
    st.warning(
        "Local LLM mode is not available. "
        "Make sure `backend/answer.py` exists, dependencies are installed, "
        "and Ollama is running with a model pulled."
    )

# Only run search when the user types a question
if question:
    st.subheader("🔍 Matching snippets")

    with st.spinner("Searching in your documents..."):
        hits = search(question, n_results=num_results)

    if not hits:
        st.info(
            "No results found.\n\n"
            "Did you run `python backend/ingest.py` and add files to the `data/` folder?"
        )
    else:
        # Show each snippet in an expandable box
        for hit in hits:
            header = f"{hit['source']} (chunk {hit['chunk_index']})"
            with st.expander(header):
                st.write(hit["text"])
                if hit["distance"] is not None:
                    st.caption(f"distance: {hit['distance']:.4f}")

        # If the user wants a full answer and we have LLM support, call it
        if use_llm and HAS_LLM:
            st.subheader("🤖 Answer (local LLM)")

            with st.spinner("Asking the local model (via Ollama)..."):
                answer_text, used_chunks = answer_question(
                    question,
                    n_context_chunks=num_results,
                )

            st.write(answer_text)

            # Optional: show which chunks were used to build the answer
            with st.expander("Show context used for this answer"):
                for hit in used_chunks:
                    header = f"{hit['source']} (chunk {hit['chunk_index']})"
                    st.markdown(f"**{header}**")
                    st.write(hit["text"])
                    st.markdown("---")
