# RAG Bot — Any Model, Free Tools

A small Retrieval-Augmented Generation (RAG) chatbot you can run locally for free.
Upload your own documents (PDF, TXT, MD, DOCX), then ask questions — the bot answers
using your documents as context. Switch between models/providers from a dropdown,
no code changes needed.

## Free stack

| Part | Tool | Cost |
|---|---|---|
| UI | Streamlit | Free |
| Vector DB | ChromaDB (local, file-based) | Free |
| Embeddings | Chroma default — `all-MiniLM-L6-v2` (runs locally) | Free |
| PDF / DOCX | pypdf / python-docx | Free |
| Local LLM | Ollama — Llama 3.1, Mistral, Gemma 2, Phi-3 | Free, no key |
| Cloud LLMs | Groq, Google Gemini, OpenRouter free tiers | Free tier / free models |

## Setup

You need Python 3.10+.

```bash
cd rag-bot

# 1. Create a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# Mac / Linux
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional, for cloud models) copy the env template and add keys
cp .env.example .env
# then edit .env and paste your keys
```

### Optional: Ollama for fully local, no-key models

1. Install from https://ollama.com
2. Pull a model and start it:

```bash
ollama pull llama3.1
ollama serve
```

Then pick **Ollama** in the app sidebar — no API key needed.

## How to get free API keys

- **Groq** — Sign up at https://console.groq.com — go to *API Keys* → *Create API Key*.
  Very fast, generous free tier. Set `GROQ_API_KEY` in `.env`.
- **Google Gemini** — Go to https://aistudio.google.com/apikey — *Create API Key*.
  Free tier for Gemini Flash models. Set `GEMINI_API_KEY` in `.env`.
- **OpenRouter** — Sign up at https://openrouter.ai — *Keys* → *Create Key*.
  Models ending in `:free` cost nothing. Set `OPENROUTER_API_KEY` in `.env`.

You can also paste a key directly in the app sidebar instead of using `.env`.

## Run

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

1. Pick a provider + model in the sidebar
2. Upload documents and click **Process documents**
3. Ask questions in the chat box — click *Sources used* under an answer to see the chunks it used

## How to use in VS Code

1. Open the `rag-bot` folder in VS Code: *File → Open Folder*
2. Open a terminal in VS Code: *Terminal → New Terminal*
3. Run the setup commands above in that terminal
4. To debug / step through code: open `app.py`, `rag.py`, or `llm.py`, set breakpoints,
   and use the built-in Python debugger — or just edit and save, Streamlit auto-reloads
5. Recommended VS Code extensions: **Python** (Microsoft), **GitLens** (optional)

Using Git in VS Code: the *Source Control* tab (branch icon) shows changed files —
type a message and click commit there, no command line needed.

## Project structure

```
rag-bot/
├── app.py              # Streamlit chat UI
├── rag.py              # Load (PDF/TXT/MD/DOCX), chunk, ChromaDB search
├── llm.py              # Any-model layer: Ollama / Groq / Gemini / OpenRouter
├── requirements.txt
├── .env.example        # Copy to .env and add your free keys
├── .gitignore
├── data/               # Put sample documents here if you like
│   └── .gitkeep
└── chroma_db/          # Created automatically on first run (git-ignored)
```

## How RAG works here

1. **Load** — `rag.load_document()` extracts text from your file
2. **Chunk** — `rag.chunk_text()` splits it into ~1000-char chunks with 200-char overlap
3. **Embed + store** — Chroma embeds each chunk locally and stores it in `chroma_db/`
4. **Retrieve** — your question is embedded, top 4 most similar chunks are fetched
5. **Generate** — context + question are sent to the model you selected in `llm.chat()`

## How to add any new model

Open `llm.py` and add the model name to the provider's `models` list in `PROVIDERS`:

```python
"groq": {
    ...
    "models": [
        "llama-3.1-8b-instant",
        "your-new-model-id",   # <-- just add a line
    ],
},
```

Or simply type any model id in the app's *"Or type any model id"* sidebar field —
no code change at all. To add a whole new OpenAI-compatible provider, copy the
`groq` entry, change `base_url` / `env_var`, and it works through the same path.

## Notes

- First run downloads the embedding model (~80 MB) once — after that it's fully local.
- Answers are only as good as your documents. If a doc has no text (scanned PDF image),
  you'll need OCR first.
- Keep `.env` out of Git — it's already in `.gitignore`.
