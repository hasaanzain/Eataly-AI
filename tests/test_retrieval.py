import pytest

from backend.ingest import sync
from backend.retrieval import RetrievalConfig, Retriever


@pytest.fixture
def retriever(kb, embeddings):
    stores, _ = sync(kb, embeddings)
    return Retriever(stores, RetrievalConfig(primary_k=3, secondary_k=1, both_k=2))


@pytest.mark.parametrize("question, route", [
    ("How much is the tiramisu?", "menu"),
    ("What is in Carbonara Classica?", "menu"),          # dish name, no menu keyword
    ("Is the margherita gluten free?", "menu"),
    ("What is the uniform policy for servers?", "documents"),
    ("How do I clock in on Dayforce?", "documents"),
    ("Which pasta should I recommend, and what are the steps for serving it?", "both"),
    ("Tell me something interesting", "both"),
])
def test_route(retriever, question, route):
    assert retriever.route(question) == route


def test_menu_question_puts_menu_records_first(retriever):
    docs = retriever.retrieve("What is in Carbonara Classica?")
    assert docs[0].metadata["item_id"] == "t-001"
    assert [d.metadata["source_type"] for d in docs] == ["menu_record"] * 3 + ["pdf"]


def test_document_question_puts_pdf_chunks_first(retriever):
    docs = retriever.retrieve("What is the uniform policy for servers?")
    assert docs[0].metadata["source_file"] == "Uniform Policy.pdf"
    assert docs[-1].metadata["source_type"] == "menu_record"


def test_question_needing_both_sources_gets_evidence_from_each(retriever):
    docs = retriever.retrieve("Carbonara price and the apron uniform policy for the shift", route="both")
    kinds = [d.metadata["source_type"] for d in docs]
    assert kinds == ["menu_record", "pdf", "menu_record", "pdf"]
    assert any(d.metadata.get("item_id") == "t-001" for d in docs)
    assert any("apron" in d.page_content for d in docs)


def test_metadata_filter_and_scores(retriever):
    docs = retriever.retrieve("wine", route="menu", menu_filter={"category": "Pasta"})
    assert all("score" in d.metadata for d in docs)
    assert all(d.metadata.get("category") == "Pasta" for d in docs if d.metadata["source_type"] == "menu_record")


def test_score_threshold_drops_weak_matches(kb, embeddings):
    stores, _ = sync(kb, embeddings)
    strict = Retriever(stores, RetrievalConfig(score_threshold=0.99))
    assert strict.retrieve("zzz unrelated words") == []


def test_no_duplicate_menu_evidence_from_the_printed_menu(retriever):
    docs = retriever.retrieve("Carbonara Classica Margherita Tiramisu", route="both")
    assert all(d.metadata["source_file"] != "Test_Menu_Main_2026.01.15.pdf"
               for d in docs if d.metadata["source_type"] == "pdf")
