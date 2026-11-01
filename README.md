# 🍝 Eataly AI

Internal Retrieval-Augmented Generation (RAG) Assistant for Eataly Staff

Eataly AI is an internal chatbot that helps staff access operational, HR, menu, wine, and procedural information from internal PDF documentation. The system uses a Retrieval-Augmented Generation pipeline built with LangChain, Chroma, and OpenAI models, served by a FastAPI backend with the "Il Mercato" web UI.

All code contained is proprietary and IP of Hasaan Mohsin.

---


## Use the chatbot!

### Access the app directly via

```
http://eataly-LoadB-Im6ruFOAYdDe-1295347466.ca-central-1.elb.amazonaws.com
```
---

## Features

* PDF and structured menu (JSON/JSONL) ingestion
* Semantic search using OpenAI embeddings
* GPT-powered contextual responses
* Real-time streaming responses
* Source chips under every grounded answer (document + page, opens the PDF)
* Every question and thumbs up/down rating logged to S3, queryable with Athena
* Follow-up questions use recent conversation history
* Built-in guardrails:

  * No employee personal data disclosure
  * No order placement or availability checks
  * Escalation for out-of-scope queries

---

## Architecture

1. Discover `.pdf`, `.json` and `.jsonl` files in `eataly_ai_knowledge_base/`
2. Structured menu records (JSON/JSONL) become one document each in the `menu_items` collection
3. Other PDFs are split page by page into overlapping chunks (1000 size, 150 overlap) in `restaurant_documents`
4. Embed with `text-embedding-3-small` into persistent Chroma (`chroma_db/`), incrementally: only new or changed sources are embedded
5. Follow-up questions ("does it contain pork?") are rewritten into standalone questions using the conversation, then routed to the menu and/or document collection; top-k is retrieved from each
6. Generate response with `gpt-5-nano`
7. Stream output to the browser over Server-Sent Events (`/api/chat`)
8. Deployed on AWS ECS Fargate behind a load balancer (see [docs/aws-setup.md](docs/aws-setup.md))

---

## 📂 Project Structure

```
.
├── backend/
│   ├── app.py            # FastAPI app: /api/chat, /api/feedback, /api/health, /api/documents
│   ├── ingest.py         # S3 download, PDF + JSON/JSONL ingestion, incremental Chroma sync
│   ├── retrieval.py      # routing between menu_items and restaurant_documents
│   ├── rag.py            # models, follow-up rewriting, prompt, citations
│   ├── events.py         # question log + feedback records to S3 (or ./logs locally)
│   └── source_registry.json  # duplicate-handling overrides (superseded / index_anyway PDFs)
├── frontend/index.html   # Il Mercato chat UI (single file, no build step)
├── eataly_ai_knowledge_base/  # PDFs + eataly_menu_enriched.json
├── evals/                # 50-question golden set + run_evals.py
├── tests/                # pytest suite (offline: fake embeddings, synthetic PDFs)
├── docs/                 # AWS setup, menu data notes, original build spec and UI guide
├── infra/aws/            # CloudFormation: foundation.yaml, app.yaml
├── .github/workflows/    # build + deploy to AWS on push to main
├── Dockerfile
├── requirements.txt      # app dependencies (also used by the Docker image)
└── requirements-dev.txt  # + pytest
```

## Run locally

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env      # add OPENAI_API_KEY (and LANGSMITH_API_KEY)
uvicorn backend.app:app --port 8000
```

Open http://localhost:8000. On startup the app syncs the knowledge base into `chroma_db/`; the first run embeds
everything (about 40 s), later runs only embed what changed (about 2 s). `/api/health` turns 200 when ready.
Add `?mock` to the URL to try the UI with canned answers and no backend calls.

---

## Knowledge base and ingestion

Drop files into `eataly_ai_knowledge_base/` (subfolders are fine):

| Format | Collection | Chunking |
|---|---|---|
| `.json` (array of records) / `.jsonl` (one record per line) | `menu_items` | One menu record per document, never split. Text is built from name, menu, section, description, price(s), dietary labels, notes, guest wording and allergen assessment; structured fields are stored as metadata. |
| `.pdf` | `restaurant_documents` | Page by page, `PDF_CHUNK_SIZE` / `PDF_CHUNK_OVERLAP` characters; short pages stay whole. |

```
python -m backend.ingest            # sync: embed new/changed sources, delete removed ones
python -m backend.ingest --rebuild  # drop both collections and re-embed everything
```

The app runs the same sync on startup, so restarting the server is enough after adding a file. Stop the server
before running the command by hand (both would write to `chroma_db/`).

**Sync policy.** Every document stores a content hash; Chroma is the source of truth for what is indexed.

* A changed JSON record is re-embedded on its own; records removed from a file are deleted.
* A changed PDF has all its chunks replaced; a deleted PDF has its chunks removed.
* A file that fails to parse keeps its previously indexed content (an error is logged) instead of being wiped.
* Changing the embedding model or PDF chunk settings rebuilds the affected collection automatically.
* `chroma_db/manifest.json` reports the last sync: every file, its SHA-256, record IDs, collection and status
  (`indexed`, `represented_by_json`, `superseded`, `needs_review`, `no_text`, `error`).

**Duplicates between menu PDFs and JSON.** Menu records name the PDF they were extracted from (`source_file`).
Those PDFs are *represented by JSON*: kept on disk so citation chips open the original page, but not embedded,
so the same dish is never retrieved twice. Ingestion also confirms this by checking how many of the records'
item names appear in the PDF text. Any other PDF whose text contains a large share of a menu's item names is
indexed but marked `needs_review` with a warning, never silently dropped. Resolve it in
`backend/source_registry.json`: list it under `superseded` (do not embed) or `index_anyway` (embed, stop warning).

**Retrieval routing.** A keyword router (menu words and dish names vs. policy/procedure words) picks a route.
Both collections are always searched: the primary one returns `RETRIEVAL_K` results and the other
`RETRIEVAL_SECONDARY_K`; unclear questions take `RETRIEVAL_BOTH_K` from each, interleaved. Results are ordered
by route rather than by score, because scores from different collections aren't comparable.

**Allergen data.** Menu allergens are inferred from listed ingredients and are not verified by the restaurant.
They are labelled that way in the text the model reads, kept in separate metadata fields
(`allergens_inferred`, `allergens_possible`, `allergens_verified=false`) and never used as "safe for" filters.
The prompt keeps listed ingredients, inferred allergens, possible allergens and verified information apart, and
never treats an unlisted ingredient as absent.

## Question log and feedback

Every answered question is saved as `queries/YYYY/MM/DD/<message_id>.json` (question, rewritten search query,
route, retrieved sources with scores, answer, fallback flag, latency) and every rating as
`feedback/YYYY/MM/DD/<message_id>.json`, in S3 on AWS or `./logs/` locally. On AWS both are Athena tables with
saved queries; see [docs/aws-setup.md](docs/aws-setup.md#everyday-tasks).

## Tests and evaluation

```
pytest                                # offline: fake embeddings, synthetic PDFs, no OpenAI calls
python -m evals.run_evals             # retrieval metrics on evals/eataly_eval_50.csv (embedding calls only)
python -m evals.run_evals --answers   # + answers from the app model, graded by EVAL_JUDGE_MODEL (default gpt-5-mini)
```

Results are written to `evals/results/`. Metrics: Recall@5, Hit@5, MRR, duplicate retrieval rate and retrieval
latency; with `--answers` also answer accuracy, citation correctness, allergen safety and end-to-end latency.
The golden set covers the menu only; the PDF document collection has no evaluation cases yet.

---

## Configuration

Environment variables (all optional):

| Variable | Default | Purpose |
|---|---|---|
| `KNOWLEDGE_BASE_DIR` | `eataly_ai_knowledge_base` | Source folder |
| `CHROMA_DIR` | `chroma_db` | Persistent Chroma storage + `manifest.json` |
| `SOURCE_REGISTRY` | `backend/source_registry.json` | Duplicate-handling overrides |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embeddings for both collections |
| `PDF_CHUNK_SIZE` / `PDF_CHUNK_OVERLAP` | `1000` / `150` | PDF chunking |
| `RETRIEVAL_K` / `RETRIEVAL_SECONDARY_K` / `RETRIEVAL_BOTH_K` | `5` / `2` / `4` | Results per collection by route |
| `RETRIEVAL_SCORE_THRESHOLD` | unset | Minimum relevance (0-1) to keep a result; leave unset until evaluated |
| `CHAT_LIMIT` / `DAILY_CHAT_LIMIT` | `20` / `2000` | Rate limits (per visitor per 5 min / per day) |
| `EVENTS_S3_URI` | bucket of `KNOWLEDGE_BASE_S3_URI` | Where `queries/` and `feedback/` are written; unset locally means `./logs/` |

Change the chat model in `chat_model()` in `backend/rag.py`.

---

## Safety Design

The assistant is restricted to staff use and enforces:

* Context-grounded responses only
* Manager escalation for unknown answers
* Privacy protection
* No operational authority

---

## Observability

LangSmith tracing is enabled via `@traceable` decorators for debugging and monitoring.

---

## 📜 License

Internal use only.
