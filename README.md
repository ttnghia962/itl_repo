# CV RAG Assistant

A candidate CV search and Q&A system built with RAG (Retrieval-Augmented Generation).
Upload CVs (PDF), extract structured information with an LLM, index them in a vector
store, and query them using natural language.

## Architecture

| Component | Technology |
|-----------|-----------|
| Backend   | FastAPI + Uvicorn (Python 3.11) |
| LLM       | Groq API (`llama-3.3-70b-versatile`) |
| Vector store | ChromaDB |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| PDF extraction | pdfplumber, pdf2image, pytesseract (OCR) |
| Frontend  | React 18 + Vite + TypeScript |

```
itl_repo/
├── backend/          # FastAPI app, services, ingestion, evaluation
│   └── app/
│       ├── main.py           # FastAPI entrypoint
│       ├── config.py         # settings (reads .env)
│       ├── routers/          # /api/cvs, /api/query
│       ├── services/         # embeddings, vector_store, llm_*, pipeline
│       ├── ingestion/        # seed data loader
│       └── evaluation/       # retrieval evaluation
├── frontend/         # React + Vite UI
├── data/             # sample CV data (PDF)
└── .env              # API keys and configuration (backend)
```

## Prerequisites

- **Python 3.11+**
- **Node.js 18+** (20+ recommended)
- **Groq API key** — sign up at https://console.groq.com
- **Tesseract OCR** and **Poppler** (required for scanned/image-based CVs, used for OCR):
  - Windows: install [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) and [Poppler](https://github.com/oschwartz10612/poppler-windows/releases), and add both to `PATH`.

## 1. Configure `.env` (backend)

Create a `.env` file at the **repo root** (`backend/app/config.py` loads it from there).

The backend uses a **round-robin pool of Groq API keys**, not a single `GROQ_API_KEY`.
Keys must be named `GROQ_API_KEY1`, `GROQ_API_KEY2`, `GROQ_API_KEY3`, ... (any number of
consecutive, uniquely-numbered keys). At least one key is required.

```env
GROQ_API_KEY1=gsk_your_first_key_here
GROQ_API_KEY2=gsk_your_second_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

> A plain `GROQ_API_KEY` (no number suffix) is **not** read by the backend and will be
> silently ignored — make sure each key has a numeric suffix.

## 2. Run the backend

```bash
cd backend

# Create and activate a virtualenv
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt

# Start the API (from backend/)
uvicorn app.main:app --reload --port 8000
```

The backend runs at **http://localhost:8000**
- Health check: http://localhost:8000/api/health → `{"status":"ok"}`
- Swagger docs: http://localhost:8000/docs

## 3. Load sample data (optional)

Load the sample CVs under `data/` into the vector store so you can search immediately
without uploading files manually:

```bash
cd backend
python -m app.ingestion.ingest_seed_data
```

> The seed data directory is configured in `backend/app/config.py` (`seed_data_dir`,
> default `data/departments/INFORMATION-TECHNOLOGY`). If that path doesn't match the
> actual location of the PDFs under `data/` in your checkout, update `seed_data_dir`
> before running the script.

This script is **idempotent** — re-running it skips CVs that were already loaded.

## 4. Run the frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend runs at **http://localhost:5173** (the backend already has CORS enabled
for this origin).

If the backend runs on a different host/port, create a `frontend/.env` file (Vite only
reads env files from the `frontend/` directory, not the repo root):

```env
VITE_API_BASE_URL=http://localhost:8000
```

## 5. Usage

1. Open http://localhost:5173
2. **Upload a CV**: choose a PDF file → the system extracts information with the LLM and indexes it.
3. **Search**: type a natural-language query (e.g. *"Find candidates with Python and cloud experience"*).
4. Review the ranked results and open the original CV file.

## Testing

Backend (pytest):
```bash
cd backend
pytest
```

Frontend (vitest):
```bash
cd frontend
npm test
```

## RAG quality evaluation

```bash
cd backend
python -m app.evaluation.run_evaluation
```

Precision@K / Recall@K results are written to `docs/RAG_EVALUATION_REPORT.md`.

## Main API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/cvs/upload` | Upload & index a CV (PDF) |
| `GET`  | `/api/cvs` | List indexed CVs |
| `GET`  | `/api/cvs/{cv_id}` | Get details of a CV |
| `GET`  | `/api/cvs/{cv_id}/file` | Download the original PDF |
| `POST` | `/api/query` | Query CVs using natural language |
| `GET`  | `/api/health` | Health check |

## Troubleshooting

- **Groq error / 401**: verify your `GROQ_API_KEY<N>` values in the root `.env` are correct, numbered, and within quota.
- **OCR / scanned PDF errors**: make sure Tesseract and Poppler are installed and on `PATH`.
- **Frontend can't reach the API**: confirm the backend is running on `:8000` and check `frontend/.env`'s `VITE_API_BASE_URL`.
- **First run is slow**: `sentence-transformers` downloads the embedding model on first use.
