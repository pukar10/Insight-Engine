"""
ingest.py

Read documents from the 'data/' folder, break them into chunks,
create embeddings for each chunk, and store everything in a local
Chroma database under 'db/'.

Run this file whenever you add or change documents:
    python backend/ingest.py
"""

import os
from pathlib import Path
import uuid

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from tqdm import tqdm

# Folder where your input documents live
DATA_DIR = Path("data")

# Folder where Chroma will store its data
DB_DIR = "db"

# Name of the collection (like a table name inside Chroma)
COLLECTION_NAME = "notes"


# ---------- File loading functions ---------- #

def load_txt(path: Path) -> str:
    """Read a .txt file and return its text."""
    return path.read_text(encoding="utf-8", errors="ignore")


def load_md(path: Path) -> str:
    """Read a .md (Markdown) file and return its text."""
    return path.read_text(encoding="utf-8", errors="ignore")


def load_pdf(path: Path) -> str:
    """Read a .pdf file and return all text from all pages."""
    reader = PdfReader(str(path))
    pages_text = []

    for page in reader.pages:
        # extract_text() may return None, so we default to ""
        text = page.extract_text() or ""
        pages_text.append(text)

    # Join text from all pages into one big string
    return "\n".join(pages_text)


# Map file extension -> loader function
LOADERS = {
    ".txt": load_txt,
    ".md": load_md,
    ".pdf": load_pdf,
}


def find_documents(data_dir: Path):
    """
    Walk through data_dir and all its subfolders.
    For each supported file, yield (file_path, file_text).

    This is a generator, so it produces one file at a time instead
    of loading everything into memory at once.
    """
    for root, _, files in os.walk(data_dir):
        for filename in files:
            ext = Path(filename).suffix.lower()
            if ext in LOADERS:
                path = Path(root) / filename
                loader = LOADERS[ext]
                text = loader(path)
                yield path, text


def split_into_chunks(text: str, max_chars: int = 800, overlap: int = 200):
    """
    Split a long string into smaller overlapping pieces ("chunks").

    max_chars:  maximum size of each chunk (in characters)
    overlap:    how many characters to keep as overlap with the previous chunk

    Example with max_chars=10, overlap=2:
      chunk 1: text[0:10]
      chunk 2: text[8:18]
      chunk 3: text[16:26]
    """
    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end]
        chunks.append(chunk)

        if end == n:
            # We reached the end of the text
            break

        # Move the window forward, but keep "overlap" characters
        start = max(0, end - overlap)

    return chunks


def main():
    """Main function: create or rebuild the Chroma index."""
    if not DATA_DIR.exists():
        print(f"Folder '{DATA_DIR}' does not exist. Please create it and add some files.")
        return

    # Turn the generator into a list so we can count the docs
    docs = list(find_documents(DATA_DIR))

    if not docs:
        print(f"No .txt, .md or .pdf files found in '{DATA_DIR}'.")
        return

    print(f"Found {len(docs)} files. Creating index...")

    # Create a Chroma client that stores data in the 'db/' folder
    client = chromadb.PersistentClient(path=DB_DIR)

    # Tell Chroma which embedding model to use
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"  # popular, light model for semantic search
    )

    # For simplicity, delete any old collection and start fresh
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        # It's okay if the collection does not exist yet
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    all_ids = []       # list of chunk IDs
    all_chunks = []    # list of chunk texts
    all_metadata = []  # list of metadata dicts

    # Process each document, split into chunks, and collect everything
    for path, full_text in tqdm(docs, desc="Processing files"):
        chunks = split_into_chunks(full_text)
        for chunk_index, chunk_text in enumerate(chunks):
            # Give each chunk a unique ID
            chunk_id = str(uuid.uuid4())

            all_ids.append(chunk_id)
            all_chunks.append(chunk_text)
            all_metadata.append(
                {
                    "source": str(path),        # which file this chunk came from
                    "chunk_index": chunk_index  # which number chunk in that file
                }
            )

    print(f"Adding {len(all_chunks)} chunks to Chroma...")

    # This call:
    #  - computes embeddings for each chunk (using embedding_fn)
    #  - stores the embeddings + text + metadata in the Chroma DB
    collection.add(
        ids=all_ids,
        documents=all_chunks,
        metadatas=all_metadata,
    )

    print("✅ Done! Embedding index is stored in the 'db/' folder.")


# This runs main() if we call: python backend/ingest.py
if __name__ == "__main__":
    main()
