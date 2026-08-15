"""Multi-agent orchestration for the LSD diagnostic pipeline.

Workflow
--------

    [1] triage  ──(proceed?)──┐
                              ├─ parallel ─┬─ [2] cnn_vision  (deterministic)
                              │            └─ [3] lesion_analyst (multimodal)
                              │
                              └─→ [4] knowledge_retrieval  (RAG, query built
                                        from the evidence above)
                                    │
                                    └─→ [5] synthesis   (verdict + citations)
                                            │
                                            └─→ [6] verification (grounding,
                                                    agreement, safety)

Agents 2 and 3 run concurrently because they are independent evidence streams —
one is a frozen CNN, the other a vision-language model that is deliberately not
told what the CNN said. Agent 4 needs both, so it is the join point.

Every stage is individually fault-tolerant: if an agent fails the step is marked
``degraded`` and the pipeline continues with whatever evidence remains, rather
than returning nothing.
"""

from __future__ import annotations

import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Callable, Optional, Sequence

from .agents import (
    knowledge_qa,
    lesion,
    reference_gallery,
    retrieval,
    synthesis,
    triage,
    verification,
)
from .agents.cnn_vision import CnnModelUnavailable, get_cnn_agent
from .config import get_settings
from .llm import ImagePayload, LLMUnavailable, get_llm
from .rag.store import get_store
from .schemas import (
    AgentName,
    AgentStatus,
    AgentStep,
    CnnResult,
    DiagnosisReport,
    ImageRef,
    LesionResult,
    RetrievalResult,
    SynthesisResult,
    TriageResult,
    Verdict,
    VerificationResult,
    VisualEvidence,
)

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[AgentStep], None]

#: Ceiling on the confidence a CNN-only (LLM-unavailable) report may claim. The
#: classifier is uncalibrated and two of the three evidence streams are missing,
#: so a screening-only result must never read as near-certain.
_DEGRADED_CONFIDENCE_CAP = 0.6

_STEP_LABELS = {
    AgentName.TRIAGE: "Intake triage",
    AgentName.CNN_VISION: "CNN screening",
    AgentName.LESION_ANALYST: "Lesion analysis",
    AgentName.KNOWLEDGE_RETRIEVAL: "Knowledge retrieval",
    AgentName.SYNTHESIS: "Diagnostic synthesis",
    AgentName.VERIFICATION: "Grounding & safety review",
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


class _Trace:
    """Collects agent steps and forwards them to an optional progress callback."""

    def __init__(self, on_progress: Optional[ProgressCallback] = None) -> None:
        self.steps: list[AgentStep] = []
        self._on_progress = on_progress
        self._timers: dict[AgentName, float] = {}

    def start(self, agent: AgentName) -> AgentStep:
        step = AgentStep(
            agent=agent,
            label=_STEP_LABELS[agent],
            status=AgentStatus.RUNNING,
            started_at=_now(),
        )
        self.steps.append(step)
        self._timers[agent] = time.perf_counter()
        self._emit(step)
        return step

    def finish(
        self,
        step: AgentStep,
        status: AgentStatus,
        summary: str = "",
        error: Optional[str] = None,
        output: Optional[dict] = None,
    ) -> AgentStep:
        step.status = status
        step.summary = summary
        step.error = error
        step.output = output
        step.finished_at = _now()
        started = self._timers.pop(step.agent, None)
        if started is not None:
            step.duration_ms = round((time.perf_counter() - started) * 1000, 1)
        self._emit(step)
        return step

    def skip(self, agent: AgentName, reason: str) -> AgentStep:
        step = AgentStep(
            agent=agent,
            label=_STEP_LABELS[agent],
            status=AgentStatus.SKIPPED,
            started_at=_now(),
            finished_at=_now(),
            duration_ms=0.0,
            summary=reason,
        )
        self.steps.append(step)
        self._emit(step)
        return step

    def _emit(self, step: AgentStep) -> None:
        if self._on_progress is None:
            return
        try:
            self._on_progress(step.model_copy(deep=True))
        except Exception:  # pragma: no cover - progress must never break the run
            logger.exception("Progress callback raised; continuing")


class DiagnosticOrchestrator:
    """Runs the six-agent workflow over a set of uploaded images."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.llm = get_llm()
        self.cnn = get_cnn_agent()
        self.store = get_store()

    # -- public entry point ----------------------------------------------
    def run(
        self,
        images: Sequence[ImagePayload],
        user_message: str = "",
        session_id: Optional[str] = None,
        on_progress: Optional[ProgressCallback] = None,
    ) -> DiagnosisReport:
        started = time.perf_counter()
        trace = _Trace(on_progress)
        report = DiagnosisReport(
            request_id=uuid.uuid4().hex,
            session_id=session_id,
            user_message=user_message,
            images=[
                ImageRef(
                    id=image.id,
                    filename=image.filename,
                    media_type=image.media_type,
                    bytes=image.size_bytes,
                )
                for image in images
            ],
        )

        llm_available = self.llm.available
        if not llm_available:
            report.degraded = True
            report.degradation_notes.append(
                f"Language model unavailable: {self.llm.unavailable_reason}"
            )

        # --- 1. triage ---------------------------------------------------
        triage_result = self._run_triage(trace, images, user_message, llm_available)
        report.triage = triage_result
        if triage_result is not None and not triage_result.proceed:
            return self._finalise_rejected(report, trace, triage_result, started)

        # --- 2 & 3. parallel evidence gathering ---------------------------
        cnn_result, visual_result, lesion_result = self._run_parallel_evidence(
            trace, images, user_message, llm_available
        )
        report.cnn = cnn_result
        report.visual = visual_result
        report.lesions = lesion_result
        if cnn_result is not None and not cnn_result.available:
            report.degraded = True
            report.degradation_notes.append(f"CNN unavailable: {cnn_result.note}")

        # --- 4. retrieval -------------------------------------------------
        retrieval_result = self._run_retrieval(
            trace, user_message, lesion_result, cnn_result
        )
        report.retrieval = retrieval_result

        # --- 5. synthesis -------------------------------------------------
        synthesis_result = self._run_synthesis(
            trace,
            user_message=user_message,
            triage_result=triage_result,
            cnn_result=cnn_result,
            visual_result=visual_result,
            lesion_result=lesion_result,
            retrieval_result=retrieval_result,
            llm_available=llm_available,
        )
        report.synthesis = synthesis_result

        # --- 6. verification ----------------------------------------------
        if synthesis_result is not None and llm_available:
            report.verification = self._run_verification(
                trace,
                synthesis=synthesis_result,
                retrieval=retrieval_result,
                cnn=cnn_result,
                lesions=lesion_result,
            )
        else:
            trace.skip(
                AgentName.VERIFICATION,
                "Skipped — no LLM synthesis to review.",
            )

        return self._finalise(report, trace, started)

    # -- individual stages -------------------------------------------------
    def _run_triage(
        self,
        trace: _Trace,
        images: Sequence[ImagePayload],
        user_message: str,
        llm_available: bool,
    ) -> Optional[TriageResult]:
        if not llm_available:
            trace.skip(AgentName.TRIAGE, "Skipped — language model unavailable.")
            return None
        step = trace.start(AgentName.TRIAGE)
        try:
            result = triage.run_triage(self.llm, images, user_message)
        except LLMUnavailable as exc:
            trace.finish(
                step,
                AgentStatus.DEGRADED,
                summary="Triage unavailable; continuing without it.",
                error=str(exc),
            )
            return None
        trace.finish(
            step,
            AgentStatus.COMPLETED,
            summary=(
                f"{result.subject_description} "
                f"(quality: {result.image_quality})"
            ),
            output=result.model_dump(mode="json"),
        )
        return result

    def _run_parallel_evidence(
        self,
        trace: _Trace,
        images: Sequence[ImagePayload],
        user_message: str,
        llm_available: bool,
    ) -> tuple[Optional[CnnResult], Optional[VisualEvidence], Optional[LesionResult]]:
        cnn_step = trace.start(AgentName.CNN_VISION)
        lesion_step = (
            trace.start(AgentName.LESION_ANALYST)
            if llm_available
            else trace.skip(
                AgentName.LESION_ANALYST, "Skipped — language model unavailable."
            )
        )

        visual_result: Optional[VisualEvidence] = None
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="evidence") as pool:
            cnn_future = pool.submit(self.cnn.run_with_evidence, images)
            lesion_future = (
                pool.submit(lesion.run_lesion_analysis, self.llm, images, user_message)
                if llm_available
                else None
            )

            # CNN — the screening probability plus its own visual explanation
            # (saliency + nearest labelled precedents), both from the same model.
            try:
                cnn_result: Optional[CnnResult]
                cnn_result, visual_result = cnn_future.result()
                trace.finish(
                    cnn_step,
                    AgentStatus.COMPLETED,
                    summary=(
                        f"P(LSD-like)={cnn_result.aggregate_probability_lsd} "
                        f"-> {cnn_result.aggregate_label} "
                        f"(threshold {cnn_result.threshold})"
                        + self._visual_summary(visual_result)
                    ),
                    output=cnn_result.model_dump(mode="json"),
                )
            except CnnModelUnavailable as exc:
                cnn_result = self.cnn.unavailable_result(str(exc))
                trace.finish(
                    cnn_step,
                    AgentStatus.DEGRADED,
                    summary="CNN model unavailable; relying on visual + literature evidence.",
                    error=str(exc),
                    output=cnn_result.model_dump(mode="json"),
                )
            except Exception as exc:  # noqa: BLE001 - agent must not kill the run
                logger.exception("CNN agent failed")
                cnn_result = self.cnn.unavailable_result(str(exc))
                trace.finish(
                    cnn_step,
                    AgentStatus.FAILED,
                    summary="CNN inference failed.",
                    error=str(exc),
                )

            # Lesion analyst
            lesion_result: Optional[LesionResult] = None
            if lesion_future is not None:
                try:
                    lesion_result = lesion_future.result()
                    trace.finish(
                        lesion_step,
                        AgentStatus.COMPLETED,
                        summary=(
                            f"{len(lesion_result.lesions)} finding(s), "
                            f"severity {lesion_result.severity.value}, "
                            f"observer confidence {lesion_result.confidence}"
                        ),
                        output=lesion_result.model_dump(mode="json"),
                    )
                except LLMUnavailable as exc:
                    trace.finish(
                        lesion_step,
                        AgentStatus.DEGRADED,
                        summary="Visual analysis unavailable.",
                        error=str(exc),
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Lesion agent failed")
                    trace.finish(
                        lesion_step,
                        AgentStatus.FAILED,
                        summary="Visual analysis failed.",
                        error=str(exc),
                    )

        return cnn_result, visual_result, lesion_result

    @staticmethod
    def _visual_summary(visual: Optional[VisualEvidence]) -> str:
        """Append the explanation counts to the CNN step's trace line."""
        if visual is None or not visual.available:
            return ""
        parts = []
        if visual.saliency:
            parts.append(f"{len(visual.saliency)} saliency map(s)")
        if visual.neighbours:
            parts.append(
                f"{len(visual.neighbours)} precedent(s) {visual.neighbour_agreement}"
            )
        return f" · {', '.join(parts)}" if parts else ""

    def _run_retrieval(
        self,
        trace: _Trace,
        user_message: str,
        lesion_result: Optional[LesionResult],
        cnn_result: Optional[CnnResult],
    ) -> RetrievalResult:
        step = trace.start(AgentName.KNOWLEDGE_RETRIEVAL)
        try:
            result = retrieval.run_retrieval(
                user_message=user_message,
                lesions=lesion_result,
                cnn=cnn_result,
                store=self.store,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Retrieval agent failed")
            trace.finish(
                step,
                AgentStatus.FAILED,
                summary="Retrieval failed; synthesis will be ungrounded.",
                error=str(exc),
            )
            return RetrievalResult(query=user_message, passages=[])

        trace.finish(
            step,
            AgentStatus.COMPLETED,
            summary=(
                f"{len(result.passages)} passage(s) from "
                f"{len({p.document_id for p in result.passages})} document(s) "
                f"via {result.strategy}"
            ),
            output={
                "query": result.query,
                "strategy": result.strategy,
                "passages": [
                    {
                        "chunk_id": passage.chunk_id,
                        "document_title": passage.document_title,
                        "section": passage.section,
                        "score": passage.score,
                    }
                    for passage in result.passages
                ],
            },
        )
        return result

    def _run_synthesis(
        self,
        trace: _Trace,
        *,
        user_message: str,
        triage_result: Optional[TriageResult],
        cnn_result: Optional[CnnResult],
        visual_result: Optional[VisualEvidence],
        lesion_result: Optional[LesionResult],
        retrieval_result: RetrievalResult,
        llm_available: bool,
    ) -> Optional[SynthesisResult]:
        if not llm_available:
            trace.skip(AgentName.SYNTHESIS, "Skipped — language model unavailable.")
            return None
        step = trace.start(AgentName.SYNTHESIS)
        try:
            result = synthesis.run_synthesis(
                self.llm,
                user_message=user_message,
                triage=triage_result,
                cnn=cnn_result,
                visual=visual_result,
                lesions=lesion_result,
                retrieval=retrieval_result,
            )
        except LLMUnavailable as exc:
            trace.finish(
                step,
                AgentStatus.DEGRADED,
                summary="Synthesis unavailable; falling back to a deterministic report.",
                error=str(exc),
            )
            return None
        except Exception as exc:  # noqa: BLE001
            logger.exception("Synthesis agent failed")
            trace.finish(
                step, AgentStatus.FAILED, summary="Synthesis failed.", error=str(exc)
            )
            return None

        trace.finish(
            step,
            AgentStatus.COMPLETED,
            summary=(
                f"{result.verdict.value} at confidence {result.confidence} "
                f"with {len(result.citations)} citation(s)"
            ),
            output=result.model_dump(mode="json"),
        )
        return result

    def _run_verification(
        self,
        trace: _Trace,
        *,
        synthesis: SynthesisResult,
        retrieval: RetrievalResult,
        cnn: Optional[CnnResult],
        lesions: Optional[LesionResult],
    ) -> Optional[VerificationResult]:
        step = trace.start(AgentName.VERIFICATION)
        try:
            result = verification.run_verification(
                self.llm,
                synthesis=synthesis,
                retrieval=retrieval,
                cnn=cnn,
                lesions=lesions,
            )
        except LLMUnavailable as exc:
            trace.finish(
                step,
                AgentStatus.DEGRADED,
                summary="Verification unavailable; report returned unreviewed.",
                error=str(exc),
            )
            return None
        except Exception as exc:  # noqa: BLE001
            logger.exception("Verification agent failed")
            trace.finish(
                step, AgentStatus.FAILED, summary="Verification failed.", error=str(exc)
            )
            return None

        flags = len(result.safety_flags) + len(result.unsupported_claims)
        trace.finish(
            step,
            AgentStatus.COMPLETED,
            summary=(
                f"grounded={result.grounded}, "
                f"cnn/vlm {result.cnn_llm_agreement}, "
                f"{flags} flag(s)"
            ),
            output=result.model_dump(mode="json"),
        )
        return result

    # -- assembly ---------------------------------------------------------
    def _finalise_rejected(
        self,
        report: DiagnosisReport,
        trace: _Trace,
        triage_result: TriageResult,
        started: float,
    ) -> DiagnosisReport:
        for agent in (
            AgentName.CNN_VISION,
            AgentName.LESION_ANALYST,
            AgentName.KNOWLEDGE_RETRIEVAL,
            AgentName.SYNTHESIS,
            AgentName.VERIFICATION,
        ):
            trace.skip(agent, "Skipped — triage rejected the submission.")

        report.verdict = (
            Verdict.NOT_CATTLE if not triage_result.is_cattle else Verdict.INCONCLUSIVE
        )
        report.confidence = 0.0
        report.headline = triage_result.reason or triage_result.subject_description
        report.answer_markdown = (
            f"I could not run the diagnostic pipeline on these images.\n\n"
            f"**What I see:** {triage_result.subject_description}\n\n"
            f"**Why I stopped:** {triage_result.reason or 'The images are not suitable for assessment.'}"
            + (
                f"\n\n**Image quality:** {triage_result.image_quality}"
                f" — {triage_result.quality_notes}"
                if triage_result.quality_notes
                else ""
            )
            + "\n\nPlease upload clear, well-lit photographs of the affected animal "
            "showing the whole body and close-ups of any skin lesions."
        )
        # The standing safety framing has to survive this path too: a rejected
        # submission still reaches a farmer, and "I could not assess this" must
        # not read as "there is nothing wrong".
        report.answer_markdown += self._disclaimer()
        report.trace = trace.steps
        report.total_ms = round((time.perf_counter() - started) * 1000, 1)
        return report

    def _finalise(
        self, report: DiagnosisReport, trace: _Trace, started: float
    ) -> DiagnosisReport:
        report.trace = trace.steps
        report.total_ms = round((time.perf_counter() - started) * 1000, 1)

        if report.synthesis is not None:
            report.verdict = report.synthesis.verdict
            report.confidence = report.synthesis.confidence
            report.headline = report.synthesis.headline
            report.answer_markdown = report.synthesis.summary_markdown

            verification_result = report.verification
            if verification_result is not None:
                if verification_result.adjusted_confidence is not None:
                    report.confidence = verification_result.adjusted_confidence
                notes = self._render_review_notes(verification_result)
                if notes:
                    report.answer_markdown += notes
        else:
            report.degraded = True
            report.answer_markdown = self._render_fallback_markdown(report)
            report.headline = (
                "Automated screening result only — the reasoning agents were "
                "unavailable."
            )
            if report.cnn is not None and report.cnn.available:
                probability = report.cnn.aggregate_probability_lsd or 0.0
                margin = abs(probability - report.cnn.threshold)
                report.verdict = (
                    Verdict.LSD_LIKELY
                    if probability >= report.cnn.threshold
                    else Verdict.LSD_UNLIKELY
                )
                if margin < self.settings.uncertainty_band:
                    report.verdict = Verdict.INCONCLUSIVE
                # Distance from the threshold is the only confidence signal
                # available here, and it is a weak one — the classifier is not
                # calibrated (see scripts/evaluate_model.py) and two of the
                # three evidence streams are missing. Cap it so a screening-only
                # result can never present itself as near-certain.
                report.confidence = round(min(margin * 2, 1.0) * _DEGRADED_CONFIDENCE_CAP, 2)
            else:
                report.verdict = Verdict.INCONCLUSIVE
                report.confidence = 0.0

        report.answer_markdown += self._disclaimer()
        return report

    @staticmethod
    def _render_review_notes(result: VerificationResult) -> str:
        lines: list[str] = []
        if result.unsupported_claims:
            lines.append(
                "- Claims not traceable to the knowledge base: "
                + "; ".join(result.unsupported_claims)
            )
        if result.cnn_llm_agreement == "disagree":
            lines.append(
                "- The image classifier and the visual assessment disagree. "
                + (result.disagreement_note or "Treat this result with extra caution.")
            )
        if result.safety_flags:
            lines.append("- Safety review: " + "; ".join(result.safety_flags))
        if result.adjusted_confidence is not None:
            lines.append(
                f"- Confidence adjusted to {result.adjusted_confidence} by the "
                "verification agent."
            )
        if not lines:
            return ""
        return "\n\n### Review notes\n\n" + "\n".join(lines)

    def _render_fallback_markdown(self, report: DiagnosisReport) -> str:
        sections = [
            "The reasoning agents are unavailable, so this is a **screening-only "
            "result** from the trained image classifier and the local knowledge base."
        ]
        cnn = report.cnn
        if cnn is not None and cnn.available:
            sections.append(
                f"**Image classifier:** P(LSD-like) = "
                f"`{cnn.aggregate_probability_lsd}` against a decision threshold of "
                f"`{cnn.threshold}` → `{cnn.aggregate_label}`.\n\n"
                "This is a binary classifier trained only on healthy versus "
                "LSD-affected cattle images. A high score means the image resembles "
                "the LSD training set more than the healthy training set — it is not "
                "a diagnosis, and the model has no class for other skin conditions."
            )
        else:
            sections.append(
                "**Image classifier:** unavailable. "
                + (cnn.note if cnn else "The model did not run.")
            )

        if report.retrieval and report.retrieval.passages:
            sections.append("**Relevant reference material:**")
            for passage in report.retrieval.passages[:3]:
                sections.append(
                    f"*{passage.document_title} — {passage.section}* "
                    f"[{passage.chunk_id}]\n\n{passage.text}"
                )
        return "\n\n".join(sections)

    @staticmethod
    def _disclaimer() -> str:
        return (
            "\n\n---\n\n"
            "*This is automated decision support, not a diagnosis. Lumpy skin "
            "disease can only be confirmed by laboratory testing, and it is a "
            "notifiable disease in most countries. Contact your veterinarian and "
            "your local veterinary authority before acting on this result.*"
        )


# --------------------------------------------------------------------------
# Text-only path
# --------------------------------------------------------------------------


def answer_text_question(question: str):
    """Answer a question with no images attached, via RAG.

    A question about appearance ("what does LSD look like?", "give me an
    example") also gets labelled reference photographs attached. That happens
    here, after the answer is produced, so both the language-model and the
    retrieval-only paths gain it and neither can have its claims influenced by
    the pictures.
    """
    llm = get_llm()
    answer = None
    if llm.available:
        try:
            answer = knowledge_qa.answer_question(llm, question)
        except LLMUnavailable as exc:
            logger.warning("Knowledge QA degraded: %s", exc)
    if answer is None:
        answer = knowledge_qa.fallback_answer(question)

    try:
        reference_gallery.attach(answer, question)
    except Exception:  # noqa: BLE001 - illustrations must never break an answer
        logger.exception("Reference gallery failed; returning the answer without it")
    return answer


_orchestrator: Optional[DiagnosticOrchestrator] = None


def get_orchestrator() -> DiagnosticOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = DiagnosticOrchestrator()
    return _orchestrator
