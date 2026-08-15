"""Chat endpoint — the single surface the Vue frontend talks to.

Routes on whether images are attached: with images it runs the full diagnostic
workflow, without them it answers from the RAG corpus.
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from ..images import decode_inline_images
from ..orchestrator import answer_text_question, get_orchestrator
from ..schemas import ChatRequest, ChatResponse, DiagnosisReport
from ..session_store import StoredMessage, get_session_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])

_VERDICT_TITLES = {
    "lsd_likely": "Lumpy Skin Disease — suspected",
    "lsd_unlikely": "No LSD indicators found",
    "inconclusive": "Inconclusive — more information needed",
    "not_cattle": "Cannot assess these images",
}


class SessionSummary(BaseModel):
    id: str
    title: str
    created_at: float
    updated_at: float
    message_count: int


class SessionMessage(BaseModel):
    id: str
    role: str
    text: str = ""
    markdown: str = ""
    title: str = ""
    kind: str = "knowledge"
    image_count: int = 0
    created_at: float
    report: Optional[DiagnosisReport] = None


@router.post("", response_model=ChatResponse, summary="Send a chat message")
async def chat(request: ChatRequest) -> ChatResponse:
    if not request.message.strip() and not request.images:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Send a message, an image, or both.",
        )

    store = get_session_store()
    session = store.get_or_create(request.session_id)

    store.append(
        session.id,
        StoredMessage(
            id=f"msg-{uuid.uuid4().hex[:10]}",
            role="user",
            text=request.message,
            image_count=len(request.images),
        ),
    )

    if request.images:
        payloads = decode_inline_images(request.images)
        orchestrator = get_orchestrator()
        report = await run_in_threadpool(
            orchestrator.run,
            payloads,
            request.message,
            session.id,
            None,
        )
        response = ChatResponse(
            session_id=session.id,
            message_id=f"msg-{uuid.uuid4().hex[:10]}",
            title=_VERDICT_TITLES.get(report.verdict.value, "Analysis"),
            markdown=report.answer_markdown,
            kind="diagnosis",
            report=report,
            citations=report.retrieval.passages if report.retrieval else [],
        )
    else:
        answer = await run_in_threadpool(answer_text_question, request.message)
        response = ChatResponse(
            session_id=session.id,
            message_id=f"msg-{uuid.uuid4().hex[:10]}",
            title="Knowledge base",
            markdown=answer.markdown,
            kind="knowledge",
            citations=answer.citations,
            examples=answer.examples,
            examples_note=answer.examples_note,
        )

    store.append(
        session.id,
        StoredMessage(
            id=response.message_id,
            role="assistant",
            markdown=response.markdown,
            title=response.title,
            kind=response.kind,
            report=response.report,
        ),
    )
    return response


@router.get(
    "/sessions",
    response_model=list[SessionSummary],
    summary="List conversations",
)
async def list_sessions() -> list[SessionSummary]:
    return [
        SessionSummary(
            id=session.id,
            title=session.title,
            created_at=session.created_at,
            updated_at=session.updated_at,
            message_count=len(session.messages),
        )
        for session in get_session_store().list_sessions()
    ]


@router.get(
    "/sessions/{session_id}",
    response_model=list[SessionMessage],
    summary="Fetch a conversation transcript",
)
async def get_session(session_id: str) -> list[SessionMessage]:
    session = get_session_store().get(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found."
        )
    return [
        SessionMessage(
            id=message.id,
            role=message.role,
            text=message.text,
            markdown=message.markdown,
            title=message.title,
            kind=message.kind,
            image_count=message.image_count,
            created_at=message.created_at,
            report=message.report,
        )
        for message in session.messages
    ]


@router.delete("/sessions/{session_id}", summary="Delete a conversation")
async def delete_session(session_id: str) -> dict[str, bool]:
    return {"deleted": get_session_store().delete(session_id)}
