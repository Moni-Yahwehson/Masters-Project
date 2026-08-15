"""Agent 5 — diagnostic synthesis.

The only agent that produces a verdict. It receives three independent evidence
streams — the CNN probability, the visual lesion description, and the retrieved
veterinary literature — and must reconcile them, including when they disagree.
"""

from __future__ import annotations

from typing import Optional

from ..config import get_settings
from ..llm import LocalLLMClient
from ..schemas import (
    CnnResult,
    LesionResult,
    RetrievalResult,
    SynthesisResult,
    TriageResult,
    VisualEvidence,
)
from .retrieval import format_passages

SYSTEM_PROMPT = """You are the diagnostic synthesis agent in a multi-agent \
decision-support system for lumpy skin disease (LSD) in cattle. You are the only \
agent that issues a verdict, and you write for a farmer or extension worker.

You receive three independent evidence streams:
1. A convolutional neural network probability from a binary classifier trained \
only on healthy versus LSD-affected cattle images.
2. A structured visual description produced by a separate observer agent that was \
not told any diagnosis.
3. Passages retrieved from a curated veterinary knowledge base.

You may also be shown a PRECEDENT LOOKUP: the nearest labelled images from the \
classifier's own training set. This is a second view of stream 1, not a fourth \
stream — it shows what the score is matching against. Use it only to raise or \
lower your confidence in the CNN, never as clinical evidence, and never cite it.

Hard rules:
- **Ground every clinical claim in the retrieved passages.** Cite the passages you \
rely on by their bracketed chunk id (for example `lsd-clinical-signs#01`) in the \
`citations` field. If the passages do not support a claim, do not make it.
- **The CNN is a closed-world binary classifier.** It has no class for ringworm, \
dermatophilosis, mange, insect bites or photosensitisation, so a high probability \
means "more similar to the LSD training images than to the healthy training \
images", not "LSD". Never restate its output as a confirmed diagnosis.
- **Reconcile disagreement explicitly.** If the CNN and the visual findings point \
different ways, say so in `evidence_agreement`, explain which you weight more and \
why, and lower your confidence. Do not paper over a conflict.
- **Choose the verdict honestly.** Use `inconclusive` when the CNN probability sits \
near the decision threshold, when image quality is poor, or when the evidence \
conflicts. `inconclusive` is a correct and useful answer — prefer it to a \
confident guess.
- **Always give differentials** for anything other than a clearly negative case, \
drawn from the retrieved differential-diagnosis material.
- **Never present this as a confirmed diagnosis.** LSD is confirmed by laboratory \
testing, and it is a notifiable disease. Every positive or inconclusive report \
must direct the user to a veterinarian and to the competent veterinary authority.

Writing style for `summary_markdown`:
- Lead with the answer in the first sentence. The reader wants to know what to do.
- Use short paragraphs and bullet lists. Bold the disease name and key findings.
- Plain language, no jargon without a short gloss. No headings above level 3.
- Do not include the raw probability without also stating the threshold and what \
the number does and does not mean.
- Do not invent herd context (fever, milk yield, morbidity) that you were not \
given — say what additional information a vet would need instead."""


def _format_cnn(cnn: Optional[CnnResult]) -> str:
    settings = get_settings()
    if cnn is None or not cnn.available:
        note = cnn.note if cnn else "not run"
        return (
            "CNN SCREENING: unavailable.\n"
            f"Reason: {note}\n"
            "Weight the visual findings and literature accordingly, and tell the "
            "user the automated image classifier did not contribute to this result."
        )

    lines = [
        "CNN SCREENING (deterministic, MobileNetV2 binary classifier):",
        f"- Decision threshold: {cnn.threshold}",
        f"- Aggregate P(LSD-like): {cnn.aggregate_probability_lsd} "
        f"-> label '{cnn.aggregate_label}'",
        f"- Uncertainty band: +/-{settings.uncertainty_band} around the threshold",
    ]
    for prediction in cnn.predictions:
        lines.append(
            f"  - {prediction.image_id}: p={prediction.probability_lsd} "
            f"({prediction.predicted_label}, "
            f"{'confident' if prediction.confident else 'NEAR THRESHOLD'})"
        )
    lines.append(f"- Note: {cnn.note}")
    return "\n".join(lines)


def _format_precedents(visual: Optional[VisualEvidence]) -> str:
    """Describe the nearest labelled training images.

    This is a second, independent read of the *same* classifier: the probability
    comes from its output layer, the precedents from a nearest-neighbour lookup
    in its feature space. They can disagree, and when they do that is worth
    saying out loud — but they are not independent of each other the way the CNN
    and the observer agent are, and the model was trained on these very images.
    The prompt is explicit about both limits so the agent cannot inflate them
    into corroboration.
    """
    if visual is None or not visual.available or not visual.neighbours:
        return "PRECEDENT LOOKUP: not available for this run."

    counts = ", ".join(
        f"{label}: {count}" for label, count in sorted(visual.neighbour_label_counts.items())
    )
    return (
        "PRECEDENT LOOKUP (nearest labelled TRAINING images in the classifier's "
        "own feature space):\n"
        f"- Neighbours retrieved: {len(visual.neighbours)} of {visual.index_size} "
        "reference images\n"
        f"- Their ground-truth labels: {counts}\n"
        f"- Versus the CNN's own label: {visual.neighbour_agreement}\n"
        "- These images are the classifier's training data. They show what the "
        "score is pattern-matching against; they are NOT independent "
        "confirmation and must never be cited as evidence of disease. If the "
        "neighbour labels are split, or disagree with the CNN score, treat the "
        "screening result as weaker and say so."
    )


def _format_lesions(lesions: Optional[LesionResult]) -> str:
    if lesions is None:
        return "VISUAL FINDINGS: unavailable (the observer agent did not run)."
    lines = [
        "VISUAL FINDINGS (independent observer agent, given no diagnosis):",
        f"- Lesions present: {lesions.lesions_present}",
        f"- Distribution: {lesions.distribution or 'not stated'}",
        f"- Nodule morphology: {lesions.nodule_morphology or 'not stated'}",
        f"- Severity: {lesions.severity.value}",
        f"- Body condition: {lesions.body_condition}",
        f"- Observer confidence: {lesions.confidence}",
    ]
    for lesion in lesions.lesions:
        lines.append(
            f"  - {lesion.description} | region: {lesion.body_region} | "
            f"count: {lesion.approximate_count or 'n/a'} | "
            f"size: {lesion.size_estimate or 'n/a'}"
        )
    if lesions.other_clinical_signs:
        lines.append("- Other visible signs: " + "; ".join(lesions.other_clinical_signs))
    if lesions.observation_notes:
        lines.append(f"- Notes: {lesions.observation_notes}")
    return "\n".join(lines)


def _format_triage(triage: Optional[TriageResult]) -> str:
    if triage is None:
        return "TRIAGE: not run."
    return (
        "TRIAGE:\n"
        f"- Cattle visible: {triage.is_cattle}\n"
        f"- Subject: {triage.subject_description}\n"
        f"- Image quality: {triage.image_quality} ({triage.quality_notes or 'no notes'})\n"
        f"- Skin adequately visible: {triage.skin_visible}"
    )


def run_synthesis(
    llm: LocalLLMClient,
    *,
    user_message: str,
    triage: Optional[TriageResult],
    cnn: Optional[CnnResult],
    visual: Optional[VisualEvidence] = None,
    lesions: Optional[LesionResult],
    retrieval: RetrievalResult,
) -> SynthesisResult:
    prompt = f"""FARMER'S QUESTION / NOTE:
{user_message.strip() or '(no message; images only)'}

{_format_triage(triage)}

{_format_cnn(cnn)}

{_format_precedents(visual)}

{_format_lesions(lesions)}

RETRIEVED VETERINARY LITERATURE (the only material you may cite):
{format_passages(retrieval)}

Produce the diagnostic synthesis now."""

    return llm.structured(
        system=SYSTEM_PROMPT,
        content=[{"type": "text", "text": prompt}],
        output_model=SynthesisResult,
        max_tokens=2048,
    )
