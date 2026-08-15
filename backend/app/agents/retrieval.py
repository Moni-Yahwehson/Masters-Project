"""Agent 4 — knowledge retrieval (the R in RAG).

Builds a retrieval query from whatever evidence is already available (the
farmer's message, the visual findings, the CNN verdict) and pulls the most
relevant passages from the curated veterinary corpus. Those passages are the
only material the synthesis agent is permitted to cite.
"""

from __future__ import annotations

from typing import Optional

from ..config import get_settings
from ..rag.store import KnowledgeStore, get_store
from ..schemas import CnnResult, LesionResult, RetrievalResult, RetrievedPassage

# Always pulled in alongside the dynamic query so the report can never be
# generated without the safety framing and the confirmation pathway.
_STANDING_QUERIES = [
    "differential diagnosis of nodular skin conditions in cattle",
    "laboratory confirmation and notification of lumpy skin disease",
    "limitations of image-based screening and CNN classifiers",
]


def build_query(
    user_message: str = "",
    lesions: Optional[LesionResult] = None,
    cnn: Optional[CnnResult] = None,
) -> str:
    parts: list[str] = ["lumpy skin disease cattle"]
    if user_message.strip():
        parts.append(user_message.strip())
    if lesions is not None:
        if lesions.lesions_present:
            parts.append("skin nodules lesions clinical signs")
            parts.extend(lesion.description for lesion in lesions.lesions[:4])
            parts.extend(
                lesion.body_region
                for lesion in lesions.lesions[:4]
                if lesion.body_region and lesion.body_region != "unspecified"
            )
            if lesions.distribution:
                parts.append(lesions.distribution)
            if lesions.nodule_morphology:
                parts.append(lesions.nodule_morphology)
            parts.extend(lesions.other_clinical_signs[:4])
        else:
            parts.append("healthy cattle skin no lesions differential diagnosis")
    if cnn is not None and cnn.available and cnn.aggregate_label:
        parts.append(cnn.aggregate_label)
    return " ".join(part for part in parts if part).strip()


#: At most this many passages from any one document in the main result set.
#: Documents differ a lot in length — the differential-diagnosis note has 11
#: chunks against 4 for most others — so an uncapped top-k lets the longest
#: document occupy nearly every slot. The synthesis agent needs breadth
#: (clinical signs *and* differentials *and* the confirmation pathway), not
#: seven variations on one theme.
_MAX_PASSAGES_PER_DOCUMENT = 3


def _cap_per_document(
    passages: list[RetrievedPassage], limit: int, max_per_document: int
) -> list[RetrievedPassage]:
    counts: dict[str, int] = {}
    kept: list[RetrievedPassage] = []
    for passage in passages:
        if counts.get(passage.document_id, 0) >= max_per_document:
            continue
        counts[passage.document_id] = counts.get(passage.document_id, 0) + 1
        kept.append(passage)
        if len(kept) >= limit:
            break
    return kept


def run_retrieval(
    user_message: str = "",
    lesions: Optional[LesionResult] = None,
    cnn: Optional[CnnResult] = None,
    store: Optional[KnowledgeStore] = None,
    top_k: Optional[int] = None,
) -> RetrievalResult:
    store = store or get_store()
    query = build_query(user_message, lesions, cnn)
    limit = top_k or get_settings().retrieval_top_k

    # Cast a wider net, then cap per document and trim back to `limit`.
    candidates = store.search(query, top_k=limit * 3)
    passages = _cap_per_document(candidates, limit, _MAX_PASSAGES_PER_DOCUMENT)
    seen = {passage.chunk_id for passage in passages}

    # Top up with the standing queries so mandatory safety context is present
    # even when the dynamic query is dominated by lesion vocabulary.
    #
    # These are appended *after* the main results and deliberately not merged
    # into one global ranking: cosine scores from different queries are not
    # comparable, so sorting the combined list lets an incidental standing-query
    # hit outrank the passage that actually answers the farmer's question.
    for standing_query in _STANDING_QUERIES:
        for passage in store.search(standing_query, top_k=2):
            if passage.chunk_id not in seen:
                seen.add(passage.chunk_id)
                passages.append(passage)

    return RetrievalResult(query=query, passages=passages, strategy=store.backend)


def format_passages(result: RetrievalResult) -> str:
    """Render retrieved passages as a citable context block for the LLM."""
    if not result.passages:
        return "(no supporting literature retrieved)"
    blocks: list[str] = []
    for passage in result.passages:
        blocks.append(
            f"[{passage.chunk_id}] {passage.document_title} — {passage.section}\n"
            f"Source: {passage.source}\n"
            f"{passage.text}"
        )
    return "\n\n---\n\n".join(blocks)
