"""Agent 6 — grounding and safety verification.

An adversarial reviewer that reads the synthesis agent's output against the
retrieved passages and the raw evidence. Its job is to catch three failure modes
before anything reaches the farmer:

* claims that are not supported by the retrieved literature (RAG hallucination);
* a confident verdict built on conflicting or weak evidence;
* missing safety framing — no vet referral, no notification advice, or an
  image-derived result presented as a confirmed diagnosis.
"""

from __future__ import annotations

from typing import Optional

from ..llm import LocalLLMClient
from ..schemas import (
    CnnResult,
    LesionResult,
    RetrievalResult,
    SynthesisResult,
    VerificationResult,
)
from .retrieval import format_passages

SYSTEM_PROMPT = """You are the verification agent — an adversarial reviewer in a \
veterinary decision-support pipeline. Another agent has produced a diagnostic \
report. Your job is to try to find what is wrong with it before a farmer sees it.

Check four things:

1. **Grounding.** Does every clinical claim in the report trace back to the \
retrieved passages you were given? List any claim that does not in \
`unsupported_claims`, quoting the claim. Generic safety advice and restatements \
of the CNN output do not need a citation; clinical facts about the disease do. \
Set `grounded` to false if any clinical claim is unsupported.

2. **Evidence agreement.** Compare the CNN probability with the visual findings. \
Set `cnn_llm_agreement` to `agree`, `disagree`, or `not_applicable` when the CNN \
did not run. If they disagree, explain the conflict in `disagreement_note` and \
check that the report acknowledged it rather than silently siding with one.

3. **Confidence calibration.** If the stated confidence is higher than the \
evidence supports — near-threshold CNN score, poor image quality, low observer \
confidence, conflicting evidence, thin retrieval — set `adjusted_confidence` to a \
value you can defend. Leave it null if the stated confidence is reasonable.

4. **Safety.** Add a `safety_flags` entry for each of these that is missing or \
wrong: a clear statement that this is not a confirmed diagnosis; a referral to a \
veterinarian; for suspected LSD, advice to notify the competent veterinary \
authority (it is a notifiable disease); any recommendation that could cause harm, \
such as prescribing a specific drug or dose, advising culling, or telling the user \
to move or sell affected animals.

Be strict but not pedantic. Report real problems, not stylistic preferences. If \
the report is sound, say so plainly and return no flags."""


def run_verification(
    llm: LocalLLMClient,
    *,
    synthesis: SynthesisResult,
    retrieval: RetrievalResult,
    cnn: Optional[CnnResult],
    lesions: Optional[LesionResult],
) -> VerificationResult:
    cnn_line = (
        f"aggregate P(LSD-like)={cnn.aggregate_probability_lsd} at threshold "
        f"{cnn.threshold} -> {cnn.aggregate_label}"
        if cnn and cnn.available
        else "CNN did not run"
    )
    lesion_line = (
        f"lesions_present={lesions.lesions_present}, "
        f"severity={lesions.severity.value}, "
        f"distribution={lesions.distribution or 'not stated'}, "
        f"observer_confidence={lesions.confidence}"
        if lesions
        else "visual observer did not run"
    )

    prompt = f"""REPORT UNDER REVIEW
Verdict: {synthesis.verdict.value}
Stated confidence: {synthesis.confidence}
Urgency: {synthesis.urgency}
Headline: {synthesis.headline}
Citations claimed: {', '.join(synthesis.citations) or '(none)'}
Evidence agreement as written: {synthesis.evidence_agreement or '(not stated)'}

Summary shown to the farmer:
{synthesis.summary_markdown}

Clinical reasoning:
{synthesis.clinical_reasoning}

Differentials: {
        '; '.join(
            f"{d.condition} ({d.likelihood})" for d in synthesis.differentials
        ) or '(none given)'
    }

Recommended actions:
{chr(10).join(f'- {action}' for action in synthesis.recommended_actions) or '(none given)'}

Notifiable-disease notice: {synthesis.notifiable_disease_notice or '(none)'}

RAW EVIDENCE
- CNN: {cnn_line}
- Visual findings: {lesion_line}

RETRIEVED PASSAGES (the full set of material the report was allowed to cite):
{format_passages(retrieval)}

Review the report now."""

    return llm.structured(
        system=SYSTEM_PROMPT,
        content=[{"type": "text", "text": prompt}],
        output_model=VerificationResult,
        max_tokens=1536,
    )
