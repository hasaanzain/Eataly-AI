# ingest.py — builds and incrementally syncs the two Chroma collections from the knowledge base.
# Run: python -m backend.ingest            (embed only what changed since the last sync)
#      python -m backend.ingest --rebuild  (drop both collections and re-embed everything)
#
# menu_items            one document per structured menu record from .json / .jsonl files
# restaurant_documents  PDF pages split into overlapping chunks
#
# Chroma itself is the source of truth for what is indexed: every document stores a content hash,
# so a sync compares hashes, upserts what changed and deletes what disappeared. manifest.json in the
# Chroma directory is a readable report of the last sync (files, hashes, record IDs, status).

import argparse
import hashlib
import json
import logging
import os
import re
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from chromadb.errors import NotFoundError
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


log = logging.getLogger("eataly.ingest")

MENU_COLLECTION = "menu_items"
DOCS_COLLECTION = "restaurant_documents"
SUFFIXES = {".pdf", ".json", ".jsonl"}
# Bump when the way documents are built changes, so existing collections are re-embedded.
SCHEMA_VERSION = "1"
# An undeclared PDF that contains this share of a menu's item names is flagged for review.
REVIEW_OVERLAP = 0.3
REVIEW_MIN_ITEMS = 3


@dataclass
class IngestConfig:
    kb_dir: Path
    chroma_dir: Path
    registry_path: Path
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 1000
    chunk_overlap: int = 150

    @classmethod
    def from_env(cls):
        return cls(
            kb_dir=Path(os.getenv("KNOWLEDGE_BASE_DIR", "eataly_ai_knowledge_base")),
            chroma_dir=Path(os.getenv("CHROMA_DIR", "chroma_db")),
            registry_path=Path(os.getenv("SOURCE_REGISTRY", Path(__file__).with_name("source_registry.json"))),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            chunk_size=int(os.getenv("PDF_CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("PDF_CHUNK_OVERLAP", "150")),
        )

    def collection_settings(self, name):
        # Changing any of these makes the stored embeddings stale, so the collection is rebuilt.
        settings = {"schema_version": SCHEMA_VERSION, "embedding_model": self.embedding_model}
        if name == DOCS_COLLECTION:
            settings |= {"chunk_size": self.chunk_size, "chunk_overlap": self.chunk_overlap}
        return settings


# ---------- Helpers ----------

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(text):
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def clean(value):
    return " ".join(str(value).split()) if value not in (None, "") else ""


def money(value):
    return str(int(value)) if float(value).is_integer() else f"{value:.2f}"


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def menu_version(source_file):
    # Menu PDFs are named like EEC_PIPA_Menu_Main_2026.10.22.pdf; the print date is the version.
    match = re.search(r"(\d{4})[._-](\d{2})[._-](\d{2})", source_file or "")
    return "-".join(match.groups()) if match else ""


def discover(kb_dir):
    files = []
    for path in sorted(kb_dir.rglob("*")):
        rel = path.relative_to(kb_dir)
        if path.is_file() and path.suffix.lower() in SUFFIXES and not any(p.startswith(".") for p in rel.parts):
            files.append(path)
    return files


def sync_from_s3(kb_dir):
    # In AWS the PDFs and menu JSON live in a private S3 bucket, e.g. KNOWLEDGE_BASE_S3_URI=s3://my-bucket/documents/
    # Locally the variable is unset and the existing folder is used as-is.
    s3_uri = os.getenv("KNOWLEDGE_BASE_S3_URI")
    if not s3_uri:
        return
    import boto3

    bucket, _, prefix = s3_uri.removeprefix("s3://").partition("/")
    s3 = boto3.client("s3")
    kb_dir.mkdir(parents=True, exist_ok=True)
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith(tuple(SUFFIXES)):
                target = kb_dir / Path(key).relative_to(prefix) if prefix else kb_dir / key
                target.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket, key, str(target))


def load_registry(path):
    # Explicit overrides for the automatic duplicate handling; see source_registry.json.
    if not path.exists():
        return {"superseded": {}, "index_anyway": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {"superseded": data.get("superseded") or {}, "index_anyway": data.get("index_anyway") or []}


# ---------- Menu records (JSON / JSONL) ----------

def read_records(path):
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        records = []
        for n, line in enumerate(text.splitlines(), 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"line {n}: {e}") from e
    else:
        data = json.loads(text)
        records = data if isinstance(data, list) else data.get("records") if isinstance(data, dict) else None
    if not isinstance(records, list):
        raise ValueError("expected a list of records")
    return records


def price_text(record):
    prices = record.get("prices")
    if isinstance(prices, dict):
        listed = [f"{label.replace('_', ' ')} ${money(v)}" for label, v in prices.items() if is_number(v)]
        if listed:
            return "Prices as printed on the menu: " + ", ".join(listed)
    if is_number(record.get("price")):
        return f"Price as printed on the menu: ${money(record['price'])}"
    return "Price: not listed on the menu"


def allergen_text(assessment):
    inferred = ", ".join(clean(a).replace("_", " ") for a in assessment.get("inferred_contains") or [])
    possible = ", ".join(clean(a).replace("_", " ") for a in assessment.get("possible_unconfirmed") or [])
    if assessment.get("verified_by_restaurant") is True:
        return f"Allergens verified by the restaurant: {inferred or 'none declared'}."
    return (
        "Allergens NOT verified by the restaurant. "
        f"Inferred from the listed ingredients only: {inferred or 'none inferred'}. "
        f"Possible but unconfirmed: {possible or 'none noted'}. "
        "Ingredients or allergens not listed here may still be present; confirm with the chef before advising a guest."
    )


def menu_document(record, kb_file):
    if not isinstance(record, dict):
        raise ValueError("record is not an object")
    item_id, name = clean(record.get("id")), clean(record.get("name"))
    if not item_id or not name:
        raise ValueError("record needs an id and a name")

    menu, section = clean(record.get("menu")), clean(record.get("section"))
    source_file = clean(record.get("source_file"))
    version = menu_version(source_file)
    labels = [clean(label) for label in record.get("dietary_labels") or [] if clean(label)]
    assessment = record.get("allergen_assessment") if isinstance(record.get("allergen_assessment"), dict) else {}
    menu_title = " ".join(filter(None, [f"{menu.title()} menu" if menu else "Menu", version]))

    lines = [
        name,
        " · ".join(filter(None, [menu_title, section])),
        f"Restaurant: {clean(record.get('restaurant'))}" if record.get("restaurant") else "",
        f"Listed on the menu as: {clean(record.get('description'))}" if record.get("description") else "",
        price_text(record),
        f"Dietary labels printed on the menu: {', '.join(labels)}" if labels else "",
        f"Notes: {clean(record.get('notes'))}" if record.get("notes") else "",
        f"Suggested guest wording (not official menu copy): {clean(record.get('guest_friendly_description'))}"
        if record.get("guest_friendly_description") else "",
        allergen_text(assessment),
    ]
    page = record.get("source_page")
    metadata = {
        "doc_id": item_id,
        "source_type": "menu_record",
        "kb_file": kb_file,
        "item_id": item_id,
        "name": name,
        "restaurant": clean(record.get("restaurant")),
        "menu": menu,
        "category": section,
        "menu_version": version,
        "price": float(record["price"]) if is_number(record.get("price")) else None,
        "dietary_labels": ",".join(labels),
        "source_file": source_file,
        "source_page": page if is_number(page) else None,
        # Kept apart from the confirmed fields on purpose: inferred allergens are not declarations
        # and must never be used as a "safe for" filter.
        "allergens_inferred": ",".join(assessment.get("inferred_contains") or []),
        "allergens_possible": ",".join(assessment.get("possible_unconfirmed") or []),
        "allergens_verified": assessment.get("verified_by_restaurant") is True,
        "allergen_status": clean(assessment.get("assessment_status")),
        "allergen_assessment": json.dumps(assessment, ensure_ascii=False),
        "content_hash": hashlib.sha256((SCHEMA_VERSION + json.dumps(record, sort_keys=True)).encode()).hexdigest(),
    }
    # Chroma accepts only str/int/float/bool metadata, never None.
    metadata = {k: v for k, v in metadata.items() if v is not None and v != ""}
    return Document(page_content="\n".join(line for line in lines if line), metadata=metadata)


def sync_menu(store, files, config, report):
    desired, failed = {}, set()
    for path in files:
        if path.suffix.lower() not in (".json", ".jsonl"):
            continue
        rel = path.relative_to(config.kb_dir).as_posix()
        try:
            records = read_records(path)
        except (OSError, UnicodeDecodeError, ValueError) as e:
            # Keep the file's existing records rather than deleting them because of a typo.
            log.error("could not read %s, keeping its indexed records: %s", rel, e)
            failed.add(rel)
            report["files"][rel] = {"collection": MENU_COLLECTION, "status": "error", "error": str(e)}
            continue
        ids = []
        for n, record in enumerate(records, 1):
            try:
                doc = menu_document(record, rel)
            except ValueError as e:
                log.warning("%s record %d skipped: %s", rel, n, e)
                continue
            item_id = doc.metadata["doc_id"]
            if item_id in desired:
                log.warning("%s record %d skipped: id %s already loaded from %s",
                            rel, n, item_id, desired[item_id].metadata["kb_file"])
                continue
            desired[item_id] = doc
            ids.append(item_id)
        report["files"][rel] = {"collection": MENU_COLLECTION, "sha256": sha256(path), "status": "indexed",
                                "records": len(ids), "record_ids": ids}

    current = indexed(store)
    changed = [d for i, d in desired.items() if current.get(i, {}).get("content_hash") != d.metadata["content_hash"]]
    removed = [i for i, m in current.items() if i not in desired and m.get("kb_file") not in failed]
    if removed:
        store.delete(ids=removed)
    if changed:
        store.add_documents(changed, ids=[d.metadata["doc_id"] for d in changed])
    report["changes"][MENU_COLLECTION] = {"upserted": len(changed), "deleted": len(removed)}
    log.info("%s: %d records, %d upserted, %d deleted", MENU_COLLECTION, len(desired), len(changed), len(removed))
    return list(desired.values())


# ---------- PDFs ----------

def pdf_documents(path, rel, file_hash, splitter):
    docs = []
    for page in PyPDFLoader(str(path)).load():
        if not page.page_content.strip():
            continue
        page_index = int(page.metadata.get("page", 0))
        # Splitting page by page keeps every chunk on one page; short pages stay whole.
        for i, chunk in enumerate(splitter.split_documents([page])):
            docs.append(Document(page_content=chunk.page_content, metadata={
                "doc_id": f"pdf:{rel}:p{page_index}:c{i}",
                "source_type": "pdf",
                "kb_file": rel,
                "source": str(path),
                "source_file": path.name,
                "document_title": path.stem.strip(" ."),
                "page": page_index,
                "page_number": page_index + 1,
                "page_label": str(page.metadata.get("page_label") or page_index + 1),
                "file_hash": file_hash,
            }))
    return docs


def menu_overlap(text, names):
    # Share of a menu's item names that appear verbatim in the text. Used as a duplicate signal,
    # never on its own to drop content.
    padded = f" {normalize(text)} "
    hits = sum(f" {name} " in padded for name in names)
    return hits, len(names)


def review_matches(text, names_by_menu):
    matches = {}
    for source_file, names in names_by_menu.items():
        hits, total = menu_overlap(text, names)
        if hits >= REVIEW_MIN_ITEMS and hits / total >= REVIEW_OVERLAP:
            matches[source_file] = f"{hits}/{total}"
    return matches


def sync_documents(store, files, config, registry, menu_docs, previous, report):
    # A PDF is "represented" when JSON records name it as their source_file (explicit provenance),
    # unless the registry says to index it anyway. Represented PDFs stay on disk for citation
    # links but are not embedded, so the same menu is never searchable twice.
    represented = defaultdict(set)
    names_by_menu = defaultdict(set)
    for doc in menu_docs:
        md = doc.metadata
        if md.get("source_file"):
            represented[md["source_file"]].add(md["kb_file"])
            name = normalize(md["name"])
            if len(name) >= 5:
                names_by_menu[md["source_file"]].add(name)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", "  ", " ", ""],
    )
    current = indexed(store)
    ids_by_file, hash_by_file = defaultdict(list), {}
    for doc_id, md in current.items():
        ids_by_file[md.get("kb_file")].append(doc_id)
        hash_by_file[md.get("kb_file")] = md.get("file_hash")

    keep, upserted, deleted = set(), 0, 0
    for path in files:
        if path.suffix.lower() != ".pdf":
            continue
        rel, name = path.relative_to(config.kb_dir).as_posix(), path.name
        digest = sha256(path)
        entry = {"sha256": digest}
        report["files"][rel] = entry
        cached = previous.get(rel, {}) if previous.get(rel, {}).get("sha256") == digest else {}

        if name in registry["superseded"]:
            entry |= {"status": "superseded", "reason": registry["superseded"][name]}
            continue
        if name in represented and name not in registry["index_anyway"]:
            entry |= {"status": "represented_by_json", "represented_by": sorted(represented[name])}
            if "content_check" in cached:
                entry["content_check"] = cached["content_check"]
            else:
                hits, total = menu_overlap(" ".join(p.page_content for p in PyPDFLoader(str(path)).load()),
                                           names_by_menu[name])
                entry["content_check"] = f"{hits}/{total} item names found in the PDF text"
                if hits / max(total, 1) < REVIEW_OVERLAP:
                    log.warning("%s is declared as the source of %d menu records but only %d of their names "
                                "were found in its text (image-only PDF?); review the provenance", rel, total, hits)
            continue

        keep.add(rel)
        entry["collection"] = DOCS_COLLECTION
        # Unchanged: same hash as the indexed chunks, or a known text-less PDF that produced none.
        if hash_by_file.get(rel) == digest or cached.get("status") == "no_text":
            entry |= {k: v for k, v in cached.items() if k in ("status", "possible_duplicate_of")}
            entry.setdefault("status", "indexed")
            entry["chunks"] = len(ids_by_file[rel])
            continue
        try:
            docs = pdf_documents(path, rel, digest, splitter)
        except Exception as e:  # pypdf raises many error types for damaged files
            log.exception("could not read %s, keeping its indexed chunks", rel)
            entry |= {"status": "error", "error": str(e)}
            continue

        reviewed = name in registry["index_anyway"]
        matches = {} if reviewed else review_matches(" ".join(d.page_content for d in docs), names_by_menu)
        if matches:
            # Possible duplicate of structured menu data: index it, but surface it for a human.
            log.warning("%s looks like it duplicates menu records %s; indexed anyway. Add it to "
                        "source_registry.json (superseded or index_anyway) once reviewed", rel, matches)
            entry |= {"status": "needs_review", "possible_duplicate_of": matches}
        else:
            entry["status"] = "indexed" if docs else "no_text"
        if not docs:
            log.warning("%s has no extractable text (scanned or image-only?)", rel)
        if ids_by_file[rel]:
            store.delete(ids=ids_by_file[rel])
            deleted += len(ids_by_file[rel])
        if docs:
            store.add_documents(docs, ids=[d.metadata["doc_id"] for d in docs])
            upserted += len(docs)
        entry["chunks"] = len(docs)

    stale = [doc_id for kb_file, ids in ids_by_file.items() if kb_file not in keep for doc_id in ids]
    if stale:
        store.delete(ids=stale)
        deleted += len(stale)
    report["changes"][DOCS_COLLECTION] = {"upserted": upserted, "deleted": deleted}
    log.info("%s: %d PDFs indexed, %d chunks upserted, %d deleted", DOCS_COLLECTION, len(keep), upserted, deleted)


# ---------- Collections ----------

def indexed(store):
    got = store.get(include=["metadatas"])
    return {doc_id: md or {} for doc_id, md in zip(got["ids"], got["metadatas"])}


def open_collection(client, name, embeddings, config, rebuild=False):
    settings = config.collection_settings(name)
    try:
        stored = client.get_collection(name).metadata or {}
    except NotFoundError:
        stored = None
    if stored is not None and (rebuild or {k: stored.get(k) for k in settings} != settings):
        log.info("rebuilding %s (%s)", name, "--rebuild" if rebuild else "embedding or chunk settings changed")
        client.delete_collection(name)
    return Chroma(
        collection_name=name,
        embedding_function=embeddings,
        client=client,
        collection_metadata=settings,
        collection_configuration={"hnsw": {"space": "cosine"}},
    )


def sync(config: IngestConfig, embeddings: Embeddings | None = None, rebuild=False):
    """Bring both collections in line with the knowledge base folder. Returns (stores, report)."""
    started = time.monotonic()
    if embeddings is None:
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(model=config.embedding_model)

    config.chroma_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.chroma_dir / "manifest.json"
    try:
        previous = json.loads(manifest_path.read_text(encoding="utf-8")).get("files", {})
    except (OSError, ValueError):
        previous = {}

    client = chromadb.PersistentClient(path=str(config.chroma_dir))
    stores = {name: open_collection(client, name, embeddings, config, rebuild)
              for name in (MENU_COLLECTION, DOCS_COLLECTION)}
    files = discover(config.kb_dir)
    report = {"files": {}, "changes": {}}

    menu_docs = sync_menu(stores[MENU_COLLECTION], files, config, report)
    sync_documents(stores[DOCS_COLLECTION], files, config, load_registry(config.registry_path),
                   menu_docs, previous, report)

    report = {
        "synced_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration_s": round(time.monotonic() - started, 1),
        "collections": {name: {"settings": config.collection_settings(name),
                               "count": client.get_collection(name).count()} for name in stores},
        **report,
    }
    manifest_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return stores, report


def main():
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="Sync the knowledge base into Chroma.")
    parser.add_argument("--rebuild", action="store_true", help="drop both collections and re-embed everything")
    args = parser.parse_args()
    config = IngestConfig.from_env()
    _, report = sync(config, rebuild=args.rebuild)
    print(json.dumps({k: report[k] for k in ("collections", "changes", "duration_s")}, indent=2))
    review = {f: e for f, e in report["files"].items() if e.get("status") in ("needs_review", "error", "no_text")}
    if review:
        print("Needs attention:", json.dumps(review, indent=2))
    print(f"Manifest: {config.chroma_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
