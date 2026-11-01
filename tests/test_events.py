import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from backend import app as api
from backend import rag
from backend.events import EventStore


def read_events(root, kind):
    files = sorted((root / kind).rglob("*.json"))
    # Athena's JSON reader needs one object per line, so every file must be a single line.
    assert all(len(f.read_text().splitlines()) == 1 for f in files)
    return {f.stem: json.loads(f.read_text()) for f in files}


def test_store_writes_one_line_json_under_kind_and_date(tmp_path):
    store = EventStore(bucket="", local_dir=tmp_path)
    key = store.save("queries", "m_abc", {"question": "Was ist das?"})
    assert key.startswith("queries/") and key.endswith("/m_abc.json")
    [(name, event)] = read_events(tmp_path, "queries").items()
    assert name == "m_abc" and event["message_id"] == "m_abc" and event["question"] == "Was ist das?"
    assert event["created_at"]


def test_store_rejects_unknown_kind(tmp_path):
    with pytest.raises(ValueError):
        EventStore(bucket="", local_dir=tmp_path).save("documents", "m_abc", {})


class FakeRetriever:
    def route(self, query):
        return "menu"

    def retrieve(self, query, route):
        return [Document(id="t-001", page_content="Carbonara Classica", metadata={
            "source_type": "menu_record", "doc_id": "t-001", "name": "Carbonara Classica", "menu": "main",
            "source_file": "Test Menu.pdf", "source_page": 2, "score": 0.81})]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "event_store", EventStore(bucket="", local_dir=tmp_path))
    monkeypatch.setattr(api.kb, "status", "ready")
    monkeypatch.setattr(api.kb, "retriever", FakeRetriever())
    monkeypatch.setattr(api.kb, "llm", RunnableLambda(lambda messages: AIMessage("It has guanciale.")))
    monkeypatch.setattr(api.kb, "rewriter", None)
    return TestClient(api.app)  # no `with`: skips the startup knowledge-base build


def test_chat_logs_query_and_feedback_links_to_it(client, tmp_path):
    response = client.post("/api/chat", json={"message": "what is in the carbonara?", "topic": "menu"})
    assert response.status_code == 200 and "event: done" in response.text
    [(message_id, query)] = read_events(tmp_path, "queries").items()
    assert query["status"] == "ok" and query["fallback"] is False
    assert query["question"] == query["search_query"] == "what is in the carbonara?"
    assert query["route"] == "menu" and query["answer"] == "It has guanciale."
    assert query["retrieved"] == [{"id": "t-001", "type": "menu_record", "title": "Carbonara Classica",
                                   "page": "2", "score": 0.81}]
    assert query["citations"] == [{"document": "Carbonara Classica · Main menu", "page": "2"}]

    assert client.post("/api/feedback", json={"message_id": message_id, "rating": "down"}).status_code == 204
    feedback = read_events(tmp_path, "feedback")[message_id]
    assert feedback["rating"] == "down" and feedback["question"] == "what is in the carbonara?"


def test_failed_chat_is_still_logged(client, tmp_path, monkeypatch):
    def broken(messages):
        raise RuntimeError("API down")
    monkeypatch.setattr(api.kb, "llm", RunnableLambda(broken))
    assert "event: error" in client.post("/api/chat", json={"message": "hello"}).text
    [query] = read_events(tmp_path, "queries").values()
    assert query["status"] == "error" and query["question"] == "hello" and query["answer"] == ""


def test_feedback_id_cannot_escape_the_log_folder(client):
    response = client.post("/api/feedback", json={"message_id": "../../documents/x", "rating": "up",
                                                  "question": "q", "answer": "a"})
    assert response.status_code == 422


def test_retrieval_record_covers_pdf_chunks():
    doc = Document(id="pdf:Uniform Policy.pdf:p0:c0", page_content="Aprons are black.", metadata={
        "source_type": "pdf", "source": "kb/Uniform Policy.pdf", "document_title": "Uniform Policy",
        "page_number": 1, "score": 0.5})
    assert rag.retrieval_record([doc]) == [{"id": "pdf:Uniform Policy.pdf:p0:c0", "type": "pdf",
                                            "title": "Uniform Policy", "page": "1", "score": 0.5}]
