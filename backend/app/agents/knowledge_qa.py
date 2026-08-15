"""Text-only question answering over the same RAG corpus.

Used when the farmer asks a question without attaching an image ("how does LSD
spread?", "can I vaccinate a pregnant cow?"). Shares the retrieval store with the
diagnostic pipeline so both surfaces answer from identical material.
"""

from __future__ import annotations

from ..llm import LocalLLMClient
from ..rag.store import get_store
from ..schemas import KnowledgeAnswer, RetrievalResult
from .retrieval import format_passages

SYSTEM_PROMPT = """You are Anima, a veterinary knowledge assistant for cattle \
farmers and extension workers, specialising in lumpy skin disease (LSD).

Answer strictly from the retrieved passages below. Rules:
- If the passages do not cover the question, say so plainly and suggest what the \
user should ask a veterinarian instead. Do not fill the gap from memory.
- Cite the passages you use with their bracketed ids, e.g. [lsd-transmission#01], \
placed at the end of the sentence they support.
- Write in markdown for a non-specialist: short paragraphs, bullet lists where \
there is a list, bold for the key terms. No headings above level 3.
- Lead with the direct answer, then the supporting detail.
- Never give a specific drug, dose or treatment protocol — refer the user to a \
veterinarian. Never present anything as a confirmed diagnosis of a specific animal.
- If the question is about a specific sick animal, answer what you can and then \
suggest they upload photographs so the diagnostic pipeline can assess them."""


def answer_question(llm: LocalLLMClient, question: str) -> KnowledgeAnswer:
    store = get_store()
    passages = store.search(question, top_k=8)
    retrieval = RetrievalResult(
        query=question, passages=passages, strategy=store.backend
    )

    prompt = f"""QUESTION:
{question.strip()}

RETRIEVED PASSAGES:
{format_passages(retrieval)}

Answer the question now."""

    markdown = llm.text(
        system=SYSTEM_PROMPT,
        content=[{"type": "text", "text": prompt}],
        max_tokens=1280,
    )
    return KnowledgeAnswer(
        question=question,
        markdown=markdown,
        citations=passages,
        grounded=bool(passages),
    )


def fallback_answer(question: str) -> KnowledgeAnswer:
    """Retrieval-only answer used when the LLM is unavailable."""
    store = get_store()
    passages = store.search(question, top_k=4)
    if not passages:
        return KnowledgeAnswer(
            question=question,
            markdown=(
                "I could not reach the language model and found nothing in the "
                "local knowledge base matching that question. Please contact your "
                "local veterinary officer."
            ),
            citations=[],
            grounded=False,
        )
    sections = [
        "The language model is unavailable, so here are the most relevant "
        "passages from the veterinary knowledge base, unsummarised:\n"
    ]
    for passage in passages:
        sections.append(
            f"**{passage.document_title} — {passage.section}** "
            f"[{passage.chunk_id}]\n\n{passage.text}"
        )
    return KnowledgeAnswer(
        question=question,
        markdown="\n\n".join(sections),
        citations=passages,
        grounded=True,
    )
