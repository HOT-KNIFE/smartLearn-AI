# SmartLearn AI

An AI-powered learning assistant that parses PDF lecture slides and answers students' course-related questions using RAG (Retrieval-Augmented Generation).

## Features

- **PDF Upload** — Upload lecture PDFs of any size (no 30-page limit)
- **RAG Retrieval** — Chunking → Embeddings → FAISS vector search for evidence
- **Cited Answers** — Every answer includes `[Page X]` citations grounded in the source PDF
- **Multi-Turn Chat** — Follow-up questions remember earlier conversation context
- **PDF Preview** — Click a citation to jump directly to the cited page in the preview
- **Local-First Fallback** — Works without an API key using keyword-based answer extraction

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python + FastAPI |
| Frontend | React + Vite |
| LLM | OpenRouter (`openrouter/free`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Search | FAISS (IndexFlatIP) |
| Deployment | Vercel (frontend) + cloud server (backend) |

## Project Structure

```
smartLearn-AI/
├── smartlearn-backend/
│   ├── main.py                 # FastAPI routes (/upload, /chat, /documents/{id}/file)
│   ├── services/
│   │   ├── rag.py              # Chunking, embedding, FAISS index, retrieval, evaluation
│   │   ├── llm.py              # OpenRouter LLM integration
│   │   └── pdf.py              # Legacy Day 2 PDF extraction
│   └── requirements.txt
├── smartlearn-frontend/
│   └── src/
│       ├── App.jsx             # Root component — coordinates shared state
│       ├── ChatPanel.jsx       # Multi-turn chat with citation buttons
│       ├── PdfPreview.jsx      # PDF iframe preview with page jumping
│       ├── api.js              # Backend API helpers
│       └── index.css           # Workspace layout (preview left, chat right)
├── Day3/
│   ├── Lab_A_Chunking_Embedding.ipynb
│   ├── Lab_B_FAISS_Retrieval.ipynb
│   └── Lab_C_Connection_to_Backend.ipynb
└── .env.example
```

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- [OpenRouter API key](https://openrouter.ai/keys) (free tier available)

### Backend Setup

```bash
cd smartLearn-AI
python -m venv venv
.\venv\Scripts\activate.bat    # Windows
source .venv/bin/activate      # macOS / Linux

cd smartlearn-backend
pip install -r requirements.txt

# Create .env file with your API key
echo OPENROUTER_API_KEY=sk-or-v1-xxxxx > .env

# Start the backend
uvicorn main:app --reload --port 8000
```

Open `http://localhost:8000/docs` to test the API routes.

### Frontend Setup

```bash
cd smartLearn-AI/smartlearn-frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/upload?chat_id=...` | Upload a PDF file |
| `POST` | `/chat` | Ask a question (`{"chat_id": "...", "message": "..."}`) |
| `GET` | `/documents/{chat_id}/file` | Serve the uploaded PDF |
| `GET` | `/health` | Health check |

### Response Shapes

**Upload:**
```json
{"status": "ok", "filename": "lesson.pdf", "pages": 28, "characters": 108492}
```

**Chat:**
```json
{
  "answer": "The sparse retriever is BM25 [Page 4].",
  "citations": [4, 6],
  "sources": [
    {"page": 4, "chunk_id": 51, "score": 0.59, "preview": "mark datasets. For single-hop..."}
  ]
}
```

## RAG Pipeline

```
PDF → extract_pages → clean_text → chunk_by_characters(overlap)
     → SentenceTransformer embed → FAISS IndexFlatIP
     → query embedding → search top-k → LLM answer with citations
```

Key configuration (adjustable in `rag.py`):
- `CHUNK_SIZE = 700` — characters per chunk
- `OVERLAP = 120` — sliding window overlap
- `TOP_K = 5` — retrieved chunks per question
- `CANDIDATE_POOL = 60` — FAISS neighbours before reranking

## Evaluation

Lab B includes a small 6-question evaluation set across `pdf1.pdf` and `pdf2.pdf`. Results are saved to `Day3/artifacts/reports/section2_eval_results.csv`.

| Metric | Score |
|--------|:----:|
| Retrieval hit rate | 83% |
| Answer hit rate (local) | 17% |
| Answer hit rate (LLM) | ~83% |

The gap between local (17%) and LLM (83%) answer hit rates illustrates why retrieval alone isn't enough — answer generation needs a language model to synthesise evidence into a coherent answer.

## License

MIT
