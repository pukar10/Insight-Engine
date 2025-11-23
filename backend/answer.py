"""
answer.py

Optional module that uses a local LLM via Ollama to generate answers
based on chunks retrieved from Chroma.

You only need this if you want "chatty" answers, not just search.

Requires:
    - Ollama installed on the host OS
    - A model pulled, e.g.:  ollama pull llama3.2
"""

from textwrap import dedent
from typing import Tuple, List, Dict

import ollama

from backend.search import search


def build_prompt(question: str, chunks: List[Dict]) -> str:
    """
    Build a prompt that tells the model to answer using only the given chunks.
    """
    # Use only the 'text' field from each chunk
    context_parts = [c["text"] for c in chunks]
    context = "\n\n---\n\n".join(context_parts)

    prompt = dedent(
        f"""
        You are an assistant that answers questions using ONLY the context below.
        If the context does not contain the answer, say:
        "I don't know based on these documents."

        Context:
        {context}

        Question: {question}

        Answer in a concise way.
        """
    ).strip()

    return prompt


def answer_question(
    question: str,
    n_context_chunks: int = 5,
    model_name: str = "llama3.2",
) -> Tuple[str, List[Dict]]:
    """
    High-level function:

    1. Search for relevant chunks for this question.
    2. Build a prompt that includes those chunks.
    3. Ask the local Ollama model to answer.
    4. Return (answer_text, chunks_used).
    """
    # Step 1: retrieve top chunks from our vector database
    chunks = search(question, n_results=n_context_chunks)

    # Step 2: build a prompt using those chunks
    prompt = build_prompt(question, chunks)

    # Step 3: send prompt to Ollama
    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )

    answer_text = response["message"]["content"]

    return answer_text, chunks


if __name__ == "__main__":
    # Simple test (requires Ollama running and a model pulled)
    q = "What do my notes say about project X?"
    ans, used_chunks = answer_question(q)
    print("Answer:\n", ans)
    print("\nNumber of chunks used:", len(used_chunks))
