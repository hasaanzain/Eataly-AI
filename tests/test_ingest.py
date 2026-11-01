import json

from backend.ingest import DOCS_COLLECTION, MENU_COLLECTION, indexed, sync
from tests.conftest import MENU_PDF, base_records, record, write_pdf


def menu_ids(stores):
    return set(indexed(stores[MENU_COLLECTION]))


def doc_metadata(stores):
    return list(indexed(stores[DOCS_COLLECTION]).values())


def test_pdf_ingestion_keeps_file_and_page_metadata(kb, embeddings):
    stores, report = sync(kb, embeddings)
    chunks = doc_metadata(stores)
    assert chunks and {md["source_file"] for md in chunks} == {"Uniform Policy.pdf"}
    assert {md["page_number"] for md in chunks} == {1, 2}
    assert all(md["doc_id"].startswith("pdf:Uniform Policy.pdf:p") for md in chunks)
    assert report["files"]["Uniform Policy.pdf"]["status"] == "indexed"


def test_pdf_chunk_ids_are_stable_across_rebuilds(kb, embeddings):
    first = set(indexed(sync(kb, embeddings)[0][DOCS_COLLECTION]))
    second = set(indexed(sync(kb, embeddings, rebuild=True)[0][DOCS_COLLECTION]))
    assert first == second


def test_json_and_jsonl_records_become_one_document_each(kb, embeddings):
    stores, report = sync(kb, embeddings)
    assert menu_ids(stores) == {"t-001", "t-002", "t-003", "t-004", "t-100"}
    assert report["files"]["menu.json"]["record_ids"] == ["t-001", "t-002", "t-003", "t-004"]
    assert report["files"]["extras.jsonl"]["record_ids"] == ["t-100"]


def test_long_menu_record_is_not_split(kb, embeddings):
    long = record("t-200", "Tagliere Grande", "cured meats, " * 60, 40)  # far longer than chunk_size=200
    (kb.kb_dir / "long.jsonl").write_text(json.dumps(long) + "\n", encoding="utf-8")
    stores, _ = sync(kb, embeddings)
    got = stores[MENU_COLLECTION].get(ids=["t-200"])
    assert len(got["ids"]) == 1 and got["documents"][0].count("cured meats") == 60


def test_menu_metadata_is_chroma_safe_and_complete(kb, embeddings):
    stores, _ = sync(kb, embeddings)
    md = stores[MENU_COLLECTION].get(ids=["t-001"])["metadatas"][0]
    assert md["name"] == "Carbonara Classica" and md["category"] == "Pasta" and md["menu"] == "main"
    assert md["price"] == 29.0 and md["source_file"] == MENU_PDF and md["source_page"] == 1
    assert md["menu_version"] == "2026-01-15" and md["source_type"] == "menu_record"
    assert md["allergens_verified"] is False and md["allergens_inferred"] == "wheat_gluten,egg,milk"
    assert json.loads(md["allergen_assessment"])["needs_staff_confirmation"] is True
    wine = stores[MENU_COLLECTION].get(ids=["t-003"])
    assert "price" not in wine["metadatas"][0]  # null price is dropped, not stored as None
    assert "glass $14, bottle $56" in wine["documents"][0]


def test_allergens_are_labelled_unverified_and_absence_is_not_claimed(kb, embeddings):
    stores, _ = sync(kb, embeddings)
    docs = dict(zip(*[stores[MENU_COLLECTION].get(ids=["t-001", "t-002"])[k] for k in ("ids", "documents")]))
    assert "NOT verified by the restaurant" in docs["t-001"]
    assert "wheat gluten, egg, milk" in docs["t-001"]
    assert "none inferred" in docs["t-002"] and "may still be present" in docs["t-002"]


def test_pdf_represented_by_json_is_kept_but_not_embedded(kb, embeddings):
    stores, report = sync(kb, embeddings)
    assert MENU_PDF not in {md["source_file"] for md in doc_metadata(stores)}
    entry = report["files"][MENU_PDF]
    assert entry["status"] == "represented_by_json" and entry["represented_by"] == ["extras.jsonl", "menu.json"]
    assert entry["content_check"].startswith("3/")


def test_undeclared_pdf_duplicating_menu_is_flagged_and_still_indexed(kb, embeddings):
    write_pdf(kb.kb_dir / "Old Menu Copy.pdf", ["Carbonara Classica\nMargherita Tradizionale\nTiramisu della Casa"])
    stores, report = sync(kb, embeddings)
    entry = report["files"]["Old Menu Copy.pdf"]
    assert entry["status"] == "needs_review" and MENU_PDF in entry["possible_duplicate_of"]
    assert "Old Menu Copy.pdf" in {md["source_file"] for md in doc_metadata(stores)}


def test_registry_superseded_pdf_is_not_indexed(kb, embeddings):
    sync(kb, embeddings)
    kb.registry_path.write_text(json.dumps({"superseded": {"Uniform Policy.pdf": "replaced"}}), encoding="utf-8")
    stores, report = sync(kb, embeddings)
    assert doc_metadata(stores) == []
    assert report["files"]["Uniform Policy.pdf"]["status"] == "superseded"


def test_unchanged_sync_embeds_nothing(kb, embeddings):
    sync(kb, embeddings)
    embeddings.embedded.clear()
    _, report = sync(kb, embeddings)
    assert embeddings.embedded == []
    assert report["changes"] == {MENU_COLLECTION: {"upserted": 0, "deleted": 0},
                                 DOCS_COLLECTION: {"upserted": 0, "deleted": 0}}


def test_updating_one_record_reembeds_only_that_record(kb, embeddings):
    sync(kb, embeddings)
    records = base_records()
    records[1]["price"] = 23
    (kb.kb_dir / "menu.json").write_text(json.dumps(records), encoding="utf-8")
    embeddings.embedded.clear()
    stores, report = sync(kb, embeddings)
    assert len(embeddings.embedded) == 1 and "Margherita" in embeddings.embedded[0]
    assert report["changes"][MENU_COLLECTION] == {"upserted": 1, "deleted": 0}
    assert stores[MENU_COLLECTION].get(ids=["t-002"])["metadatas"][0]["price"] == 23


def test_removed_record_and_deleted_pdf_are_removed_from_index(kb, embeddings):
    sync(kb, embeddings)
    (kb.kb_dir / "menu.json").write_text(json.dumps(base_records()[:3]), encoding="utf-8")
    (kb.kb_dir / "Uniform Policy.pdf").unlink()
    stores, report = sync(kb, embeddings)
    assert "t-004" not in menu_ids(stores)
    assert doc_metadata(stores) == []
    assert report["changes"][MENU_COLLECTION]["deleted"] == 1


def test_changed_pdf_replaces_its_chunks(kb, embeddings):
    sync(kb, embeddings)
    write_pdf(kb.kb_dir / "Uniform Policy.pdf", ["Uniform policy, revised: aprons are green."])
    stores, _ = sync(kb, embeddings)
    texts = stores[DOCS_COLLECTION].get()["documents"]
    assert texts == ["Uniform policy, revised: aprons are green."]


def test_broken_json_keeps_previously_indexed_records(kb, embeddings):
    sync(kb, embeddings)
    (kb.kb_dir / "menu.json").write_text("{not json", encoding="utf-8")
    stores, report = sync(kb, embeddings)
    assert report["files"]["menu.json"]["status"] == "error"
    assert {"t-001", "t-002", "t-003", "t-004"} <= menu_ids(stores)


def test_chunk_setting_change_rebuilds_document_collection(kb, embeddings):
    sync(kb, embeddings)
    kb.chunk_size = 80
    embeddings.embedded.clear()
    stores, _ = sync(kb, embeddings)
    assert embeddings.embedded and all("Carbonara" not in t for t in embeddings.embedded)  # menu untouched
    assert all(len(t) <= 80 for t in stores[DOCS_COLLECTION].get()["documents"])


def test_textless_pdf_is_reported_once_and_not_reread(kb, embeddings, monkeypatch):
    write_pdf(kb.kb_dir / "Scanned Label.pdf", [""])
    _, report = sync(kb, embeddings)
    assert report["files"]["Scanned Label.pdf"]["status"] == "no_text"
    calls = []
    monkeypatch.setattr("backend.ingest.pdf_documents", lambda *a: calls.append(a) or [])
    _, report = sync(kb, embeddings)
    assert calls == [] and report["files"]["Scanned Label.pdf"]["status"] == "no_text"


def test_manifest_records_files_hashes_and_counts(kb, embeddings):
    sync(kb, embeddings)
    manifest = json.loads((kb.chroma_dir / "manifest.json").read_text())
    assert manifest["collections"][MENU_COLLECTION]["count"] == 5
    assert len(manifest["files"]["menu.json"]["sha256"]) == 64
    assert manifest["collections"][DOCS_COLLECTION]["settings"]["chunk_size"] == 200
