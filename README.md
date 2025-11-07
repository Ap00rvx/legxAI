# Legal BERT XAI: QA + Lawyer Recommendation

This project provides a lightweight, explainable assistant for Indian legal queries:

- Retrieval-augmented QA over bundled corpora (`data/*.json`)
- Extractive BERT QA (`deepset/bert-base-cased-squad2`)
- Token-level explanations via Integrated Gradients (Captum)
- Lawyer-type recommendation from query with rationale hints

## What’s included

- Data loaders for `constitution_qa.json`, `ipc_qa.json`, `crpc_qa.json`, and `train.jsonl`.
- Sentence-embedding index using a BERT-based dense retriever (`msmarco-bert-base-dot-v5`).
- QA model for answer extraction.
- XAI: per-token importance highlighting for the top answer.
- CLI for quick usage.

## Setup (Windows PowerShell)

1. Create/activate a virtual environment (optional but recommended):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

Note: The first run will download pretrained models from Hugging Face.

## Try it

Ask a question (with explanation):

```powershell
python -m legal_xai.app_cli ask "What is Article 14 about?" --top-k 5 --explain
```

Recommend lawyer types:

```powershell
python -m legal_xai.app_cli recommend "Police arrested my brother without warrant and won’t tell us why"
```

## How it works

- Retrieval: builds a dense embedding index over the provided corpora (filters very short entries). Top-k passages are retrieved for a query.
- QA: runs a BERT QA head on each retrieved passage and returns the highest-confidence answer.
- XAI: uses Integrated Gradients over the input embeddings to attribute importance to tokens, returning normalized scores for the context tokens used to extract the answer.
- Recommendation: rule-based hints + keyword patterns map the query to likely practice areas (e.g., criminal, constitutional). This can be upgraded to an embedding-based classifier.

## Upgrading recommendation to model-based (optional)

- Replace the rule-based `infer_practice_areas` with an embedding similarity over curated practice area descriptions using the same retriever model.
- Optionally fine-tune a text classifier (BERT) on labeled queries → practice areas and use LIME/SHAP for explanations.

## Ethics and limitations

- This tool is for educational/assistance purposes only and is not legal advice.
- Model outputs may be inaccurate; always consult a qualified advocate.
- Explanations (token attributions) are indicative and not causal.

## Project layout

- `legal_xai/data_loading.py` — load sources and simple rule-based area hints
- `legal_xai/embedding.py` — BERT-based sentence embeddings and index
- `legal_xai/retrieval.py` — build/retrieve from the index
- `legal_xai/qa.py` — extractive QA (shared model via `model_registry.py`)
- `legal_xai/explain.py` — Integrated Gradients token importances (shared QA model)
- `legal_xai/recommend.py` — recommend lawyer types
- `legal_xai/model_registry.py` — central caching of QA & embedding models
- `legal_xai/app_cli.py` — CLI entrypoints
- `legal_xai/api.py` — FastAPI service with `/ask` (streaming or JSON)

## API (FastAPI)

Start the API (defaults to index limit 200 for faster cold start):

```powershell
.\.venv\Scripts\Activate.ps1
$env:LEGAL_XAI_INDEX_LIMIT=200
uvicorn legal_xai.api:app --host 127.0.0.1 --port 8000 --reload
```

Health:

```powershell
curl "http://127.0.0.1:8000/health"
```

Ask (streaming via Server‑Sent Events):

```powershell
curl "http://127.0.0.1:8000/ask?q=What%20is%20Article%2014%3F&top_k=3&explain=true&stream=true&links=5"
```

Ask (non‑streaming JSON):

```powershell
curl "http://127.0.0.1:8000/ask?q=What%20is%20Article%2014%3F&top_k=3&explain=false&stream=false&links=5"
```

### Response shape (`/ask` when stream=false)

```json
{
	"top-answer": "string",
	"explaination-text": "string",
	"links": [ { "label": "link-1", "type": "link", "url": "...", "text": "..." } ],
	"related-topics": "comma-separated areas",
	"related-questions": ["...", "..."]
}
```

Notes:
- Out‑of‑context detection: query is out‑of‑scope if it lacks legal cues AND similarity is low. High-priority legal terms (e.g., “supreme court”) force in‑scope.
- Even out‑of‑context responses still return India‑biased search links.
- Web links prefer Indian legal domains (indiankanoon.org, *.gov.in) and fall back to Google/DDG/Bing search links if scraping is blocked.
- Streaming events order: `retrieval` → `answers` → `web` → `explanation` (optional) → `final` → `done`.

## Developer notes

- Models centralized via `model_registry.py` to avoid duplicate memory usage.
- India biasing applied in link queries; thresholds & keyword lists tunable in `api.py`.
- Out‑of‑scope logic combines keyword heuristics & similarity with adaptive threshold for short queries.

## Next steps

- Add `/recommend` API endpoint.
- Persist embedding index to disk to skip recomputation.
- Add tests: retrieval accuracy, out‑of‑scope detection, explanation fallback.

## Deploy to Render

You can deploy the API as a Render Web Service using the included `render.yaml`.

1) Push this repo to GitHub/GitLab.

2) In Render, click “New +” → “Web Service” → “Build and deploy from a Git repository”, select your repo.

3) If Render auto-detects the Python environment, you can accept it, or choose “Use Render.yaml” to auto-apply settings from `render.yaml`.

If configuring manually, use:
- Environment: Python
- Build Command: `pip install --upgrade pip setuptools wheel && pip install -r requirements.txt`
- Start Command: `uvicorn legal_xai.api:app --host 0.0.0.0 --port $PORT`
- Environment Variables:
	- `PYTHONUNBUFFERED=1`
	- `LEGAL_XAI_INDEX_LIMIT=200` (tune lower for faster cold starts)
	- Optional caches: `HF_HOME=/opt/render/project/.cache/huggingface`, `SENTENCE_TRANSFORMERS_HOME=/opt/render/project/.cache/sentence-transformers`

Notes:
- First deploy may take longer due to model downloads. Subsequent deploys are faster if you enable a persistent disk (see `render.yaml` comments) to cache models.
- The service listens on `$PORT` per Render’s requirement.
- You can scale the instance plan if you need more memory for larger indexes.
