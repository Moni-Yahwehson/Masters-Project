"""Pydantic contracts shared by the API, the agents and the Vue frontend.

These models are the single source of truth for the wire format. The frontend
mirrors them in `frontend/src/services/api.js`; keep the two in sync.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AgentName(str, Enum):
    TRIAGE = "triage"
    CNN_VISION = "cnn_vision"
    LESION_ANALYST = "lesion_analyst"
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    SYNTHESIS = "synthesis"
    VERIFICATION = "verification"


class AgentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    DEGRADED = "degraded"
    FAILED = "failed"


class Verdict(str, Enum):
    LSD_LIKELY = "lsd_likely"
    LSD_UNLIKELY = "lsd_unlikely"
    INCONCLUSIVE = "inconclusive"
    NOT_CATTLE = "not_cattle"


class Severity(str, Enum):
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    UNKNOWN = "unknown"


# --------------------------------------------------------------------------
# Agent-level payloads
# --------------------------------------------------------------------------


class ImageRef(BaseModel):
    """An uploaded image after normalisation."""

    id: str
    filename: Optional[str] = None
    media_type: str = "image/jpeg"
    width: Optional[int] = None
    height: Optional[int] = None
    bytes: Optional[int] = None


class TriageResult(BaseModel):
    is_cattle: bool = Field(description="Whether cattle are visible in the images.")
    subject_description: str = Field(
        description="One sentence naming what is actually in the image."
    )
    image_quality: Literal["good", "acceptable", "poor"] = "acceptable"
    quality_notes: str = ""
    skin_visible: bool = Field(
        default=True,
        description="Whether enough of the animal's skin is visible to assess lesions.",
    )
    proceed: bool = Field(
        default=True, description="Whether the diagnostic pipeline should continue."
    )
    reason: str = ""


class CnnPrediction(BaseModel):
    image_id: str
    probability_lsd: float = Field(ge=0.0, le=1.0)
    predicted_label: str
    confident: bool = Field(
        description="False when the probability sits inside the uncertainty band."
    )


class CnnResult(BaseModel):
    available: bool = True
    model_name: str = "MobileNetV2 + dense head (cattle_lsd_detector.keras)"
    predictions: list[CnnPrediction] = Field(default_factory=list)
    aggregate_probability_lsd: Optional[float] = None
    aggregate_label: Optional[str] = None
    threshold: float = 0.5
    inference_ms: Optional[float] = None
    note: str = ""


class SaliencyMap(BaseModel):
    """A Grad-CAM overlay showing which pixels drove the CNN's score."""

    image_id: str
    original_data_url: str = Field(
        description="The uploaded image, resized for display, as a data URL."
    )
    overlay_data_url: str = Field(
        description="The same frame with the saliency heat map composited on top."
    )
    probability_lsd: float = Field(ge=0.0, le=1.0)
    explains: Literal["lsd_like", "healthy_like"] = Field(
        default="lsd_like",
        description=(
            "Which call the heat map explains. The gradient is taken towards the "
            "label the model actually predicted, so on a negative call the hot "
            "regions are the evidence *against* LSD."
        ),
    )
    method: str = "grad-cam"


class VisualNeighbour(BaseModel):
    """A labelled training image close to the upload in the CNN's feature space."""

    query_image_id: str
    reference_id: str = Field(description="Filename of the dataset image.")
    label: str = Field(description="Its ground-truth folder: healthycows | lumpycows.")
    similarity: float = Field(
        ge=-1.0, le=1.0, description="Cosine similarity of the 128-d embeddings."
    )
    thumbnail_data_url: str


class ExampleImage(BaseModel):
    """A reference photograph shown alongside a text answer.

    Drawn from the labelled training set, so it illustrates what the condition
    looks like in this dataset. It is not a diagnosis and not a match against
    anything the user uploaded.
    """

    reference_id: str
    label: str
    caption: str
    thumbnail_data_url: str


class VisualEvidence(BaseModel):
    """Deterministic, human-checkable explanations of the CNN screening step.

    Produced from the classifier's own activations — no language model is
    involved, and nothing here is a diagnosis.
    """

    available: bool = False
    saliency: list[SaliencyMap] = Field(default_factory=list)
    neighbours: list[VisualNeighbour] = Field(default_factory=list)
    #: Ground-truth label counts across the returned neighbours.
    neighbour_label_counts: dict[str, int] = Field(default_factory=dict)
    #: Whether the precedent vote lands on the same side as the CNN's own score.
    neighbour_agreement: Optional[Literal["agree", "disagree", "split"]] = None
    #: Total images in the precedent index the neighbours were drawn from.
    index_size: int = 0
    #: True when the upload is itself in the training set — i.e. the model has
    #: already seen this exact image, so its score on it is not a fair test.
    #: Surfaced rather than silently dropped because it is the single most
    #: common way a demo overstates the system.
    query_in_index: bool = False
    embedding_layer: str = ""
    elapsed_ms: Optional[float] = None
    note: str = ""


class Lesion(BaseModel):
    description: str
    body_region: str = "unspecified"
    approximate_count: Optional[str] = None
    size_estimate: Optional[str] = None


class LesionResult(BaseModel):
    lesions_present: bool
    lesions: list[Lesion] = Field(default_factory=list)
    distribution: str = Field(
        default="",
        description="Whether findings are localised or generalised over the body.",
    )
    nodule_morphology: str = ""
    other_clinical_signs: list[str] = Field(default_factory=list)
    body_condition: str = "unknown"
    severity: Severity = Severity.UNKNOWN
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    observation_notes: str = ""


class RetrievedPassage(BaseModel):
    chunk_id: str
    document_id: str
    document_title: str
    source: str
    section: str = ""
    text: str
    score: float


class RetrievalResult(BaseModel):
    query: str
    passages: list[RetrievedPassage] = Field(default_factory=list)
    strategy: str = "tfidf-cosine"


class DifferentialDiagnosis(BaseModel):
    condition: str
    likelihood: Literal["high", "moderate", "low"]
    distinguishing_features: str


class SynthesisResult(BaseModel):
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    headline: str = Field(description="One-sentence answer for the farmer.")
    summary_markdown: str
    clinical_reasoning: str
    differentials: list[DifferentialDiagnosis] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    urgency: Literal["routine", "prompt", "urgent"] = "prompt"
    notifiable_disease_notice: str = ""
    citations: list[str] = Field(
        default_factory=list,
        description="chunk_ids from the retrieval agent supporting the claims.",
    )
    evidence_agreement: str = Field(
        default="",
        description="How CNN, visual and literature evidence line up or conflict.",
    )


class VerificationResult(BaseModel):
    grounded: bool = Field(
        description="Whether every clinical claim traces to retrieved literature."
    )
    unsupported_claims: list[str] = Field(default_factory=list)
    cnn_llm_agreement: Literal["agree", "disagree", "not_applicable"] = "not_applicable"
    disagreement_note: str = ""
    safety_flags: list[str] = Field(default_factory=list)
    adjusted_confidence: Optional[float] = None
    reviewer_note: str = ""


# --------------------------------------------------------------------------
# Orchestration envelope
# --------------------------------------------------------------------------


class AgentStep(BaseModel):
    agent: AgentName
    label: str
    status: AgentStatus = AgentStatus.PENDING
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    duration_ms: Optional[float] = None
    summary: str = ""
    error: Optional[str] = None
    output: Optional[dict[str, Any]] = None


class DiagnosisReport(BaseModel):
    request_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    session_id: Optional[str] = None
    user_message: str = ""
    images: list[ImageRef] = Field(default_factory=list)

    verdict: Verdict = Verdict.INCONCLUSIVE
    confidence: float = 0.0
    headline: str = ""
    answer_markdown: str = ""

    triage: Optional[TriageResult] = None
    cnn: Optional[CnnResult] = None
    visual: Optional[VisualEvidence] = None
    lesions: Optional[LesionResult] = None
    retrieval: Optional[RetrievalResult] = None
    synthesis: Optional[SynthesisResult] = None
    verification: Optional[VerificationResult] = None

    trace: list[AgentStep] = Field(default_factory=list)
    degraded: bool = False
    degradation_notes: list[str] = Field(default_factory=list)
    total_ms: Optional[float] = None


# --------------------------------------------------------------------------
# HTTP request/response shapes
# --------------------------------------------------------------------------


class InlineImage(BaseModel):
    """An image supplied as a base64 data URL (the shape the Vue client sends)."""

    data_url: str
    filename: Optional[str] = None


class ChatRequest(BaseModel):
    message: str = ""
    images: list[InlineImage] = Field(default_factory=list)
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    message_id: str
    title: str
    markdown: str
    kind: Literal["diagnosis", "knowledge", "error"] = "knowledge"
    report: Optional[DiagnosisReport] = None
    citations: list[RetrievedPassage] = Field(default_factory=list)
    examples: list[ExampleImage] = Field(default_factory=list)
    examples_note: str = ""
    created_at: datetime = Field(default_factory=_utcnow)


class KnowledgeAnswer(BaseModel):
    question: str
    markdown: str
    citations: list[RetrievedPassage] = Field(default_factory=list)
    grounded: bool = True
    examples: list[ExampleImage] = Field(default_factory=list)
    examples_note: str = ""


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    llm_available: bool
    llm_model: str
    llm_runtime: str = "local"
    cnn_available: bool
    cnn_model_path: str
    knowledge_documents: int
    knowledge_chunks: int
    notes: list[str] = Field(default_factory=list)


class DocumentInfo(BaseModel):
    document_id: str
    title: str
    source: str
    chunks: int
    topics: list[str] = Field(default_factory=list)
