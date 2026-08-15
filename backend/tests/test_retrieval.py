"""Retrieval quality checks for the RAG store.

These double as the evaluation harness for the retrieval stage: each case is a
query a farmer might realistically type, paired with the document that should be
retrieved. Run with:

    .venv/bin/python -m pytest backend/tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.store import KnowledgeStore  # noqa: E402

#: (query, document_id that must appear in the top-3 results)
RETRIEVAL_CASES = [
    ("how does lumpy skin disease spread between cattle", "lsd-transmission"),
    ("what flies carry the virus", "lsd-transmission"),
    ("can my cow catch it from another cow", "lsd-transmission"),
    ("what do the lumps look like", "lsd-clinical-signs"),
    ("how big are the nodules and where do they appear", "lsd-clinical-signs"),
    ("my cow has a fever and swollen glands", "lsd-clinical-signs"),
    ("is it ringworm or lumpy skin disease", "lsd-differential"),
    ("other skin diseases that look similar", "lsd-differential"),
    ("rain scald crusts along the back", "lsd-differential"),
    ("what test confirms lumpy skin disease", "lsd-laboratory"),
    ("what sample should I send to the laboratory", "lsd-laboratory"),
    ("is there a cure or medicine", "lsd-management"),
    ("should I vaccinate my herd", "lsd-management"),
    ("do I have to report this to anyone", "lsd-management"),
    ("what virus causes it", "lsd-etiology"),
    ("can humans get lumpy skin disease", "lsd-etiology"),
    ("how accurate is an AI photo diagnosis", "lsd-screening-limits"),
]


@pytest.fixture(scope="module")
def store() -> KnowledgeStore:
    return KnowledgeStore().load()


def test_corpus_loads(store: KnowledgeStore) -> None:
    documents, chunks = store.stats
    assert documents >= 7, "expected the full curated corpus"
    assert chunks >= 30, "expected the corpus to chunk on section headings"


@pytest.mark.parametrize(("query", "expected_document"), RETRIEVAL_CASES)
def test_expected_document_in_top_3(
    store: KnowledgeStore, query: str, expected_document: str
) -> None:
    hits = store.search(query, top_k=3, min_score=0.0)
    retrieved = [hit.document_id for hit in hits]
    assert expected_document in retrieved, (
        f"{query!r} -> {retrieved} (wanted {expected_document})"
    )


def test_scores_are_ordered(store: KnowledgeStore) -> None:
    hits = store.search("lumpy skin disease nodules", top_k=6, min_score=0.0)
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)


def test_empty_query_returns_nothing(store: KnowledgeStore) -> None:
    assert store.search("   ") == []


def test_chunk_ids_are_unique(store: KnowledgeStore) -> None:
    ids = [chunk.chunk_id for chunk in store.chunks]
    assert len(ids) == len(set(ids))


# -- retrieval agent (the diagnostic path) ---------------------------------


def test_diagnostic_retrieval_is_diverse() -> None:
    """The synthesis agent needs breadth, not many chunks of one document.

    Documents differ in length (the differential note has 11 chunks against 4
    for most), so without a per-document cap the longest one takes nearly every
    slot and the verdict is written without the clinical-signs or confirmation
    material.
    """
    from app.agents.retrieval import run_retrieval

    result = run_retrieval(user_message="My cow has lumps all over her body")
    documents = {passage.document_id for passage in result.passages}

    assert len(documents) >= 4, f"retrieval collapsed onto {documents}"
    counts: dict[str, int] = {}
    for passage in result.passages:
        counts[passage.document_id] = counts.get(passage.document_id, 0) + 1
    # Standing queries top up after the cap, so allow a little headroom.
    assert max(counts.values()) <= 5, counts


def test_diagnostic_retrieval_always_includes_safety_context() -> None:
    """Confirmation pathway and screening limits must be present regardless of
    what the dynamic query happens to match."""
    from app.agents.retrieval import run_retrieval

    result = run_retrieval(user_message="the cow looks fine to me")
    documents = {passage.document_id for passage in result.passages}
    assert "lsd-laboratory" in documents
    assert "lsd-screening-limits" in documents


def test_query_expansion_reaches_clinical_vocabulary() -> None:
    from app.rag.store import _expand_query

    expanded = _expand_query("how do the lumps spread")
    assert "transmission" in expanded
    assert "nodule" in expanded
