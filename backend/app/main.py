"""FastAPI application entry point.

Run with:
    uvicorn app.main:app --reload --port 8000   (from the backend/ directory)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .agents.cnn_vision import get_cnn_agent
from .config import get_settings
from .llm import get_llm
from .rag.store import get_store
from .routers import chat, diagnose, knowledge
from .schemas import HealthResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()

DESCRIPTION = """
Multi-agent decision support for **Lumpy Skin Disease (LSD)** in cattle.

Six agents cooperate on every image submission:

1. **Intake triage** — is this cattle, and is the image usable?
2. **CNN screening** — the trained MobileNetV2 classifier (`cattle_lsd_detector.keras`)
3. **Lesion analysis** — a vision-language model describes the visible findings
4. **Knowledge retrieval** — RAG over a curated veterinary corpus
5. **Diagnostic synthesis** — reconciles the evidence into a cited verdict
6. **Grounding & safety review** — an adversarial check before the farmer sees it

Steps 2 and 3 run in parallel as independent evidence streams.

This service provides decision support, not a diagnosis. LSD is confirmed by
laboratory testing and is a notifiable disease.
"""

@asynccontextmanager
async def lifespan(_: FastAPI):
    documents, chunks = get_store().stats
    logger.info(
        "Knowledge base ready: %d documents / %d chunks (%s)",
        documents,
        chunks,
        get_store().backend,
    )
    if not get_llm().available:
        logger.warning("LLM unavailable: %s", get_llm().unavailable_reason)
    if not get_cnn_agent().available:
        logger.warning("CNN unavailable: %s", get_cnn_agent().unavailable_reason)
    yield


app = FastAPI(
    title="Anima — LSD Diagnostic API",
    description=DESCRIPTION,
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(diagnose.router)
app.include_router(chat.router)
app.include_router(knowledge.router)


@app.get("/api/health", response_model=HealthResponse, tags=["meta"])
async def health() -> HealthResponse:
    llm = get_llm()
    cnn = get_cnn_agent()
    store = get_store()
    documents, chunks = store.stats

    notes: list[str] = []
    llm_available = llm.available
    cnn_available = cnn.available
    if not llm_available:
        notes.append(f"LLM unavailable: {llm.unavailable_reason}")
    if not cnn_available:
        notes.append(f"CNN unavailable: {cnn.unavailable_reason}")
    if not chunks:
        notes.append(f"Knowledge base is empty at {settings.knowledge_dir}")
    notes.append(f"Retrieval backend: {store.backend}")

    return HealthResponse(
        status="ok" if (llm_available and cnn_available and chunks) else "degraded",
        llm_available=llm_available,
        llm_model=f"{settings.vision_model} (vision) / {settings.text_model} (text)",
        llm_runtime=f"local · {settings.llm_base_url}",
        cnn_available=cnn_available,
        cnn_model_path=str(settings.model_path),
        knowledge_documents=documents,
        knowledge_chunks=chunks,
        notes=notes,
    )


@app.get("/api/config", tags=["meta"])
async def config() -> dict:
    """Non-secret runtime configuration, useful for the frontend and for a
    reproducibility appendix."""
    return {
        "llm_runtime": "local",
        "llm_base_url": settings.llm_base_url,
        "vision_model": settings.vision_model,
        "text_model": settings.text_model,
        "installed_models": get_llm().installed_models,
        "decision_threshold": settings.decision_threshold,
        "uncertainty_band": settings.uncertainty_band,
        "positive_class": settings.positive_class,
        "negative_class": settings.negative_class,
        "image_size": settings.model_image_size,
        "retrieval_top_k": settings.retrieval_top_k,
        "retrieval_backend": get_store().backend,
        "max_images_per_request": settings.max_images_per_request,
    }
