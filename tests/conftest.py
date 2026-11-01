# Shared fixtures: a tiny synthetic knowledge base and an offline embedding model,
# so the tests never call OpenAI or touch the real documents.

import hashlib
import json
import math
import re

import pytest
from langchain_core.embeddings import Embeddings

from backend.ingest import IngestConfig


class KeywordEmbeddings(Embeddings):
    """Bag-of-words vectors: texts sharing words are similar. Counts every text it embeds."""

    dims = 256

    def __init__(self):
        self.embedded = []

    def _vector(self, text):
        vec = [0.0] * self.dims
        for word in re.findall(r"[a-z]+", text.lower()):
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dims] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts):
        self.embedded.extend(texts)
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


def write_pdf(path, pages):
    """Write a minimal valid PDF with one Helvetica text block per page."""
    def escape(s):
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    objects = [b"<< /Type /Catalog /Pages 2 0 R >>"]
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(len(pages)))
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for i, text in enumerate(pages):
        lines = " ".join(f"({escape(line)}) Tj T*" for line in text.split("\n"))
        stream = f"BT /F1 10 Tf 40 760 Td 12 TL {lines} ET".encode("latin-1")
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                       f"/Resources << /Font << /F1 3 0 R >> >> /Contents {5 + 2 * i} 0 R >>".encode())
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for n, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{off:010d} 00000 n \n".encode() for off in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))


MENU_PDF = "Test_Menu_Main_2026.01.15.pdf"


def record(item_id, name, description, price=None, **extra):
    return {
        "id": item_id, "restaurant": "Test Trattoria", "menu": "main", "section": "Pasta",
        "name": name, "price": price, "description": description, "dietary_labels": [], "notes": "",
        "source_file": MENU_PDF, "source_page": 1,
        "guest_friendly_description": f"Try {name}.",
        "allergen_assessment": {
            "inferred_contains": [], "possible_unconfirmed": [],
            "assessment_status": "ingredient_inference_only",
            "verified_by_restaurant": False, "needs_staff_confirmation": True,
        },
        **extra,
    }


def base_records():
    carbonara = record("t-001", "Carbonara Classica", "Rigatoni, guanciale, eggs, pecorino romano, black pepper", 29.0)
    carbonara["allergen_assessment"]["inferred_contains"] = ["wheat_gluten", "egg", "milk"]
    return [
        carbonara,
        record("t-002", "Margherita Tradizionale", "San Marzano tomato sauce, mozzarella, basil", 21),
        record("t-003", "Lambrusco Rosato", "Sparkling red wine", None, prices={"glass": 14, "bottle": 56}),
        record("t-004", "Tiramisu della Casa", "Mascarpone, espresso soaked savoiardi, cocoa", 13.5),
    ]


@pytest.fixture
def kb(tmp_path):
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    (kb_dir / "menu.json").write_text(json.dumps(base_records()), encoding="utf-8")
    (kb_dir / "extras.jsonl").write_text(
        json.dumps(record("t-100", "Focaccia Bianca", "Focaccia, olive oil, sea salt", 7)) + "\n", encoding="utf-8")
    # The printed menu the JSON came from: must stay on disk but must not be embedded.
    write_pdf(kb_dir / MENU_PDF, ["Carbonara Classica 29\nMargherita Tradizionale 21\nTiramisu della Casa 13.5"])
    write_pdf(kb_dir / "Uniform Policy.pdf", [
        "Uniform policy. All servers wear a black apron and black non-slip shoes on every shift.",
        "Clocking in. Clock in on Dayforce at the start of each shift before greeting guests.",
    ])
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"superseded": {}, "index_anyway": []}), encoding="utf-8")
    config = IngestConfig(kb_dir=kb_dir, chroma_dir=tmp_path / "chroma", registry_path=registry,
                          chunk_size=200, chunk_overlap=20)
    return config


@pytest.fixture
def embeddings():
    return KeywordEmbeddings()
