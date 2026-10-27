# 🍝 Eataly AI

Internal Retrieval-Augmented Generation (RAG) Assistant for Eataly Staff

Eataly AI is an internal chatbot that helps staff access operational, HR, menu, wine, and procedural information from internal PDF documentation. The system uses a Retrieval-Augmented Generation pipeline built with LangChain, Chroma, and OpenAI models, served by a FastAPI backend with the "Il Mercato" web UI.

All code contained is proprietary and IP of Hasaan Mohsin.

---


## Use the chatbot!

### Access the app directly via

```
http://eataly-LoadB-Im6ruFOAYdDe-1295347466.ca-central-1.elb.amazonaws.com   (AWS, new UI)
https://eataly-ai.streamlit.app/                                             (legacy Streamlit UI)
```
---

## Features

* PDF knowledge base ingestion
* Semantic search using OpenAI embeddings
* GPT-powered contextual responses
* Real-time streaming responses
* Source chips under every grounded answer (document + page, opens the PDF)
* Thumbs up/down feedback saved to S3
* Follow-up questions use recent conversation history
* Built-in guardrails:

  * No employee personal data disclosure
  * No order placement or availability checks
  * Escalation for out-of-scope queries

---

## Architecture

1. Load PDFs from `eataly_ai_knowledge_base/`
2. Split into overlapping chunks (512 size, 50 overlap)
3. Generate embeddings with `text-embedding-3-small`
4. Store in Chroma vector database
5. Retrieve top-k relevant chunks
6. Generate response with `gpt-5-nano`
7. Stream output to the browser over Server-Sent Events (`/api/chat`)
8. Deployed on AWS ECS Fargate behind a load balancer (see [docs/aws-setup.md](docs/aws-setup.md)); the legacy Streamlit app still runs on Streamlit Community Cloud

---

## 📂 Project Structure

```
.
├── backend/              # FastAPI app: /api/chat, /api/feedback, /api/health, /api/documents
├── frontend/index.html   # Il Mercato chat UI (single file, no build step)
├── chatbot.py            # shared PDF loading, chunking, embeddings, model
├── streamlit_app.py      # legacy Streamlit UI (Streamlit Cloud)
├── eataly_ai_knowledge_base/
├── infra/aws/            # CloudFormation: foundation.yaml, app.yaml
├── .github/workflows/    # build + deploy to AWS on push to main
├── Dockerfile
├── requirements-api.txt  # Docker / FastAPI dependencies
└── requirements.txt      # Streamlit Cloud dependencies
```

## Run locally

```
pip install -r requirements-api.txt
cp .env.example .env      # add OPENAI_API_KEY (and LANGSMITH_API_KEY)
uvicorn backend.app:app --port 8501
```

Open http://localhost:8501. The first answer waits until the PDFs are indexed (`/api/health` turns 200).
Add `?mock` to the URL to try the UI with canned answers and no backend calls.

---

## Configuration

* Adjust chunking in `get_chunks()`
* Modify retrieval depth with `RETRIEVAL_K` in `backend/app.py` (`k` in `chatbot_stream()` for Streamlit)
* Rate limits: `CHAT_LIMIT` (per visitor per 5 min, default 20) and `DAILY_CHAT_LIMIT` (default 2000) environment variables
* Change model in `get_llm()`

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
