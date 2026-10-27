# app.py — HTTP API + static front end for the Eataly staff assistant.
# Run: uvicorn backend.app:app --host 0.0.0.0 --port 8501

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

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

import chatbot
from backend import rag
from backend.feedback import FeedbackStore


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("eataly.api")

FRONTEND = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
HISTORY_TURNS = 6
RETRIEVAL_K = 4

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
    """Built once per process in a background thread; /api/health reports 503 until ready."""

    def __init__(self):
        self.status = "loading"
        self.vectordb = None
        self.llm = None
        self.files = {}

    def build(self):
        started = time.monotonic()
        try:
            self.vectordb = chatbot.get_vectordb(chatbot.path)
            self.files = {p.name: p for p in chatbot.get_doc_names(chatbot.path)}
            self.llm = chatbot.get_llm()
            self.status = "ready"
            log.info("knowledge base ready: %d documents in %.1fs", len(self.files), time.monotonic() - started)
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
feedback_store = FeedbackStore()


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


# ---------- Schemas ----------

class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = Field(default=None, max_length=64)
    history: list[Turn] = Field(default_factory=list, max_length=20)
    topic: str | None = Field(default=None, max_length=32)


class FeedbackRequest(BaseModel):
    message_id: str = Field(max_length=64)
    rating: Literal["up", "down"] | None
    comment: str | None = Field(default=None, max_length=1000)


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
    body = {"status": kb.status, "documents": len(kb.files)}
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
        parts = []
        try:
            docs = rag.retrieve(kb.vectordb, rag.retrieval_query(message, history), k=RETRIEVAL_K)
            for text in rag.stream_answer(kb.llm, rag.build_messages(message, history, docs)):
                parts.append(text)
                yield sse("token", {"text": text})
            answer = "".join(parts)
            grounded = bool(answer.strip()) and not rag.is_fallback(answer)
            sources = rag.to_sources(docs) if grounded else []
            yield sse("sources", {"sources": sources})
            recent_answers.put(message_id, {
                "conversation_id": conversation_id,
                "question": message,
                "answer": answer,
                "citations": [{"document": s["title"], "page": s["page"]} for s in sources],
                "topic": req.topic,
            })
            yield sse("done", {"message_id": message_id, "conversation_id": conversation_id})
            log.info(
                "chat ok conversation=%s message=%s ms=%d chars=%d sources=%d fallback=%s topic=%s",
                conversation_id, message_id, (time.monotonic() - started) * 1000, len(answer),
                len(sources), not grounded, req.topic,
            )
        except Exception:
            log.exception("chat failed conversation=%s message=%s", conversation_id, message_id)
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
    if record is None:
        raise HTTPException(404, "Unknown message.")
    feedback_store.save({
        "message_id": req.message_id,
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
