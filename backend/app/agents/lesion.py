"""Agent 3 — multimodal lesion analyst.

Uses Claude's vision capability to produce a *structured clinical description* of
what is visible: nodule morphology, distribution, body region, severity and other
signs. Deliberately does not name a disease — that keeps the observation stage
independent from the diagnosis stage, so the synthesis agent is combining two
genuinely separate lines of evidence (CNN score + visual description) rather than
one opinion echoed twice.
"""

from __future__ import annotations

from typing import Sequence

from ..llm import ImagePayload, LocalLLMClient, image_blocks
from ..schemas import LesionResult

SYSTEM_PROMPT = """You are a veterinary clinical observer in a decision-support \
pipeline. You examine photographs of cattle and describe the visible dermatological \
findings precisely and objectively.

Rules:
- Describe only what is visible. Do not name a disease, do not diagnose, and do \
not speculate about aetiology. Another agent performs the diagnosis.
- For each distinct finding, record the morphology (raised/flat, firm-looking/ \
soft, round/irregular, ulcerated/intact, crusted/scaly), the body region, an \
approximate count, and an approximate size where the image gives a reference.
- State whether findings are localised to one region or generalised across the \
body. This distinction is clinically important, so be explicit.
- Record other visible signs separately: ocular or nasal discharge, swelling of \
limbs or brisket, salivation, visible lymph node enlargement, poor body \
condition, lameness posture, teat or udder involvement.
- Rate severity by extent and ulceration: none, mild, moderate or severe.
- Set your confidence honestly. Low resolution, distance, motion blur, harsh \
shadow, mud on the coat or obstructed views should lower it.
- If you see no dermatological abnormality, set lesions_present to false and say \
so plainly rather than inventing findings."""


def run_lesion_analysis(
    llm: LocalLLMClient, images: Sequence[ImagePayload], user_message: str = ""
) -> LesionResult:
    content: list[dict] = [
        {
            "type": "text",
            "text": (
                "Describe the visible dermatological findings in these cattle "
                "photographs.\n"
                f"Farmer's note (context only, may be unreliable): "
                f"{user_message.strip() or '(none provided)'}"
            ),
        }
    ]
    content.extend(image_blocks(images))
    return llm.structured(
        system=SYSTEM_PROMPT,
        content=content,
        output_model=LesionResult,
        model=llm.vision_model,
        max_tokens=1536,
    )
