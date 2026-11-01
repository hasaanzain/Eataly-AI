# app.py — HTTP API + static front end for the Eataly staff assistant.
# Run locally: uvicorn backend.app:app --port 8000   (the Docker image serves on 8501)

import json
import logging
import os
import threading
import time
import uuid
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from backend import ingest, rag
from backend.events import EventStore
from backend.retrieval import Retriever


load_dotenv()  # local runs; in AWS the variables come from the task definition

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("eataly.api")

FRONTEND = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
HISTORY_TURNS = 6

# Abuse/cost protection for an app with no login (per client IP, plus a daily ceiling).
CHAT_LIMIT, CHAT_WINDOW_S = int(os.getenv("CHAT_LIMIT", "20")), 300
FEEDBACK_LIMIT, FEEDBACK_WINDOW_S = 60, 300
DAILY_CHAT_LIMIT = int(os.getenv("DAILY_CHAT_LIMIT", "2000"))

CONTENT_SECURITY_POLICY = (
    "default-src 'self'; script-src 'self' 'unsafe-inline'; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; "
    "frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
)


# ---------- In-process state ----------

class KnowledgeBase:
    """Synced once per process in a background thread; /api/health reports 503 until ready.
    The sync is incremental, so only new or changed sources are embedded on startup."""

    def __init__(self):
        self.status = "loading"
        self.retriever = None
        self.llm = None
        self.rewriter = None
        self.files = {}
        self.counts = {}

    def build(self):
        started = time.monotonic()
        try:
            config = ingest.IngestConfig.from_env()
            ingest.sync_from_s3(config.kb_dir)
            stores, report = ingest.sync(config)
            self.retriever = Retriever(stores)
            self.counts = {name: c["count"] for name, c in report["collections"].items()}
            self.files = {p.name: p for p in ingest.discover(config.kb_dir) if p.suffix.lower() == ".pdf"}
            self.llm = rag.chat_model()
            self.rewriter = rag.rewrite_model()
            self.status = "ready"
            log.info("knowledge base ready: %s, %d PDFs servable, in %.1fs",
                     self.counts, len(self.files), time.monotonic() - started)
        except Exception:
            self.status = "error"
            log.exception("knowledge base build failed")


class RateLimiter:
    def __init__(self, limit, window_s):
        self.limit, self.window_s = limit, window_s
        self.hits = {}
        self.lock = threading.Lock()

    def allow(self, key):
        now = time.monotonic()
        with self.lock:
            if len(self.hits) > 10_000:
                self.hits = {k: q for k, q in self.hits.items() if q and q[-1] > now - self.window_s}
            q = self.hits.setdefault(key, deque())
            while q and q[0] <= now - self.window_s:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(now)
            return True


class DailyLimit:
    def __init__(self, limit):
        self.limit, self.day, self.count = limit, date.today(), 0
        self.lock = threading.Lock()

    def allow(self):
        with self.lock:
            if date.today() != self.day:
                self.day, self.count = date.today(), 0
            if self.count >= self.limit:
                return False
            self.count += 1
            return True


class RecentAnswers:
    """Keeps recent answers so feedback events can include the question, answer and citations."""

    def __init__(self, size=1000):
        self.size, self.items = size, OrderedDict()
        self.lock = threading.Lock()

    def put(self, message_id, record):
        with self.lock:
            self.items[message_id] = record
            while len(self.items) > self.size:
                self.items.popitem(last=False)

    def get(self, message_id):
        with self.lock:
            return self.items.get(message_id)


kb = KnowledgeBase()
chat_limiter = RateLimiter(CHAT_LIMIT, CHAT_WINDOW_S)
feedback_limiter = RateLimiter(FEEDBACK_LIMIT, FEEDBACK_WINDOW_S)
daily_chats = DailyLimit(DAILY_CHAT_LIMIT)
recent_answers = RecentAnswers()
event_store = EventStore()


@asynccontextmanager
async def lifespan(app):
    threading.Thread(target=kb.build, name="kb-build", daemon=True).start()
    yield


app = FastAPI(title="Eataly Staff Assistant", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "same-origin"
    return response


def client_ip(request):
    # The ALB appends the real client address as the last X-Forwarded-For entry.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


def new_id(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def save_query(message_id, record):
    # The query log must never break a chat, so failures are only logged.
    try:
        event_store.save("queries", message_id, record)
    except Exception:
        log.exception("query log failed message=%s", message_id)


# ---------- Schemas ----------

class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = Field(default=None, max_length=64)
    history: list[Turn] = Field(default_factory=list, max_length=20)
    topic: str | None = Field(default=None, max_length=32)


class Citation(BaseModel):
    document: str = Field(max_length=200)
    page: str | None = Field(default=None, max_length=20)


class FeedbackRequest(BaseModel):
    # Becomes part of the stored file name, so only plain ID characters are accepted.
    message_id: str = Field(max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    rating: Literal["up", "down"] | None
    comment: str | None = Field(default=None, max_length=1000)
    # Sent by the browser as a fallback for answers given before a server restart.
    question: str | None = Field(default=None, max_length=2000)
    answer: str | None = Field(default=None, max_length=20000)
    citations: list[Citation] = Field(default_factory=list, max_length=10)


# ---------- Routes ----------

@app.get("/", include_in_schema=False)
def home():
    return FileResponse(
        FRONTEND,
        media_type="text/html",
        headers={"Cache-Control": "no-cache", "Content-Security-Policy": CONTENT_SECURITY_POLICY, "X-Frame-Options": "DENY"},
    )


@app.get("/api/health")
def health():
    body = {"status": kb.status, "documents": len(kb.files), "indexed": kb.counts}
    return JSONResponse(body, status_code=200 if kb.status == "ready" else 503)


@app.post("/api/chat")
def chat(req: ChatRequest, request: Request):
    if not chat_limiter.allow(client_ip(request)) or not daily_chats.allow():
        raise HTTPException(429, "Too many questions at once. Try again in a moment.")
    if kb.status != "ready":
        raise HTTPException(503, "The knowledge base is still loading. Try again in a minute.")
    message = req.message.strip()
    if not message:
        raise HTTPException(422, "Message is empty.")

    history = req.history[-HISTORY_TURNS:]
    conversation_id = req.conversation_id or new_id("c")
    message_id = new_id("m")

    def events():
        started = time.monotonic()
        parts, query, route, docs = [], None, None, []

        def query_record(status, fallback, citations):
            return {
                "conversation_id": conversation_id,
                "status": status,
                "question": message,
                "search_query": query,
                "route": route,
                "answer": "".join(parts),
                "fallback": fallback,
                "citations": citations,
                "retrieved": rag.retrieval_record(docs),
                "history_turns": len(history),
                "topic": req.topic,
                "latency_ms": round((time.monotonic() - started) * 1000),
            }

        try:
            query = rag.standalone_question(kb.rewriter, message, history)
            route = kb.retriever.route(query)
            docs = kb.retriever.retrieve(query, route)
            for text in rag.stream_answer(kb.llm, rag.build_messages(message, history, docs)):
                parts.append(text)
                yield sse("token", {"text": text})
            answer = "".join(parts)
            grounded = bool(answer.strip()) and not rag.is_fallback(answer)
            sources = rag.to_sources(docs) if grounded else []
            citations = [{"document": s["title"], "page": s["page"]} for s in sources]
            yield sse("sources", {"sources": sources})
            recent_answers.put(message_id, {
                "conversation_id": conversation_id,
                "question": message,
                "answer": answer,
                "citations": citations,
                "topic": req.topic,
            })
            save_query(message_id, query_record("ok", not grounded, citations))
            yield sse("done", {"message_id": message_id, "conversation_id": conversation_id})
            log.info(
                "chat ok conversation=%s message=%s ms=%d chars=%d route=%s sources=%d fallback=%s topic=%s",
                conversation_id, message_id, (time.monotonic() - started) * 1000, len(answer), route,
                len(sources), not grounded, req.topic,
            )
        except Exception:
            log.exception("chat failed conversation=%s message=%s", conversation_id, message_id)
            save_query(message_id, query_record("error", None, []))
            yield sse("error", {"message": "Something went wrong. Try again."})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/feedback", status_code=204)
def feedback(req: FeedbackRequest, request: Request):
    if not feedback_limiter.allow(client_ip(request)):
        raise HTTPException(429, "Too many requests. Try again in a moment.")
    record = recent_answers.get(req.message_id)
    if record is not None:
        record = {**record, "record_source": "server"}
    elif req.question and req.answer:
        record = {
            "conversation_id": None,
            "question": req.question,
            "answer": req.answer,
            "citations": [c.model_dump() for c in req.citations],
            "topic": None,
            "record_source": "browser",
        }
    else:
        raise HTTPException(404, "Unknown message.")
    event_store.save("feedback", req.message_id, {
        "rating": req.rating,
        "comment": req.comment,
        **record,
    })
    log.info("feedback saved message=%s rating=%s", req.message_id, req.rating)
    return Response(status_code=204)


@app.get("/api/documents/{filename}")
def document(filename: str):
    path = kb.files.get(filename)
    if path is None:
        raise HTTPException(404, "Document not found.")
    return FileResponse(path, media_type="application/pdf", filename=filename, content_disposition_type="inline")
