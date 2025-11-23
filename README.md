# Insight-engine
Turn your messy notes and PDFs into a tiny, local “second brain”.

Ask questions in a simple web UI and get back:
- Relevant snippets from your files
- (Optionally but **highly** recommended) a full AI answer powered by a local LLM

All local. All free. No API keys. No cloud.



## ✨ What it does

- Indexes your `.txt`, `.md`, and `.pdf` files in `data/`
- Lets you search semantically (not just exact keywords)
- Shows you where each answer came from (file + chunk)
- Optional: uses a local model (via Ollama) to summarize and answer in full sentences



## 🛠 Tech

- **Python**
- **Streamlit** – web UI
- **Chroma** – local vector database
- **sentence-transformers** – embeddings (`all-MiniLM-L6-v2`)
- **Ollama + phi4 (optional)** – local LLM for chatty answers



### 🚀 local setup
---

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add files to index (data/)

# 3. Build index
python backend/ingest.py

# 4. Run app
streamlit run app.py

# (Optional) Install ollama server with phi4 LLM

```
