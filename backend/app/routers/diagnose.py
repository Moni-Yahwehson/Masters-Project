"""Diagnosis endpoints — both a plain request/response form and an SSE stream
that reports each agent's progress as the workflow runs."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ..images import decode_inline_images, decode_uploads
from ..orchestrator import get_orchestrator
from ..schemas import AgentStep, DiagnosisReport, InlineImage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/diagnose", tags=["diagnosis"])


class DiagnoseJsonRequest(BaseModel):
    message: str = ""
    images: list[InlineImage] = Field(default_factory=list)
    session_id: Optional[str] = None


@router.post("", response_model=DiagnosisReport, summary="Diagnose uploaded images")
async def diagnose_multipart(
    images: list[UploadFile] = File(..., description="One or more cattle photographs"),
    message: str = Form("", description="Optional note from the farmer"),
    session_id: Optional[str] = Form(None),
) -> DiagnosisReport:
    """Run the full six-agent workflow over a multipart image upload."""
    payloads = await decode_uploads(images)
    if not payloads:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one image is required.",
        )
    orchestrator = get_orchestrator()
    return await run_in_threadpool(
        orchestrator.run, payloads, message, session_id, None
    )


@router.post(
    "/json",
    response_model=DiagnosisReport,
    summary="Diagnose base64 data-URL images",
)
async def diagnose_json(request: DiagnoseJsonRequest) -> DiagnosisReport:
    """Same workflow, but accepting the base64 data URLs the browser produces."""
    payloads = decode_inline_images(request.images)
    if not payloads:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one image is required.",
        )
    orchestrator = get_orchestrator()
    return await run_in_threadpool(
        orchestrator.run, payloads, request.message, request.session_id, None
    )


@router.post("/stream", summary="Diagnose with live per-agent progress (SSE)")
async def diagnose_stream(request: DiagnoseJsonRequest) -> StreamingResponse:
    """Stream `agent_step` events as each agent starts and finishes, then a
    final `report` event carrying the complete `DiagnosisReport`.

    Event shape (Server-Sent Events):

        event: agent_step
        data: {"agent": "cnn_vision", "status": "completed", ...}

        event: report
        data: {"request_id": "...", "verdict": "lsd_likely", ...}
    """
    payloads = decode_inline_images(request.images)
    if not payloads:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one image is required.",
        )

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    _SENTINEL = object()

    def on_progress(step: AgentStep) -> None:
        # Called from the orchestrator's worker thread.
        loop.call_soon_threadsafe(queue.put_nowait, ("agent_step", step))

    def run_pipeline() -> None:
        orchestrator = get_orchestrator()
        try:
            report = orchestrator.run(
                payloads, request.message, request.session_id, on_progress
            )
            loop.call_soon_threadsafe(queue.put_nowait, ("report", report))
        except Exception as exc:  # noqa: BLE001 - surfaced to the client as an event
            logger.exception("Diagnosis pipeline failed")
            loop.call_soon_threadsafe(
                queue.put_nowait, ("error", {"detail": str(exc)})
            )
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, (_SENTINEL, None))

    async def event_source():
        task = asyncio.create_task(run_in_threadpool(run_pipeline))
        try:
            while True:
                event, payload = await queue.get()
                if event is _SENTINEL:
                    break
                body = (
                    payload.model_dump(mode="json")
                    if isinstance(payload, (AgentStep, DiagnosisReport))
                    else payload
                )
                yield f"event: {event}\ndata: {json.dumps(body)}\n\n"
            yield "event: done\ndata: {}\n\n"
        finally:
            await task

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # stop nginx buffering the stream
        },
    )
