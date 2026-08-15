"""Agent 1 — intake triage.

Runs a fast multimodal check before any expensive work: are these actually
cattle, is the skin visible, is the image usable? Catches the common failure
mode where a binary classifier is handed a photo of a goat, a dog or a landscape
and confidently returns one of its two known labels.
"""

from __future__ import annotations

from typing import Sequence

from ..llm import ImagePayload, LocalLLMClient, image_blocks
from ..schemas import TriageResult

SYSTEM_PROMPT = """You are the intake triage agent in a veterinary decision-support \
pipeline for lumpy skin disease (LSD) in cattle. You run before the diagnostic \
agents and your only job is to decide whether the submitted images are suitable \
for assessment.

Judge four things:
1. Are cattle or water buffalo visible? Other animals, humans, plants, documents \
and unrelated scenes must be rejected — the downstream classifier only knows two \
cattle classes and will produce a misleading label on anything else.
2. Is enough of the animal's skin and body surface visible to look for lesions?
3. Is the image quality usable — focus, lighting, resolution, framing, occlusion?
4. Should the pipeline proceed?

Set proceed=false only when the images are unusable or contain no cattle. Poor \
but interpretable images should proceed with a quality note. Do not attempt any \
diagnosis; that is another agent's job. Describe only what you can actually see."""


def run_triage(
    llm: LocalLLMClient, images: Sequence[ImagePayload], user_message: str = ""
) -> TriageResult:
    content: list[dict] = [
        {
            "type": "text",
            "text": (
                "Assess the following image(s) for suitability.\n"
                f"Farmer's note: {user_message.strip() or '(none provided)'}"
            ),
        }
    ]
    content.extend(image_blocks(images))
    return llm.structured(
        system=SYSTEM_PROMPT,
        content=content,
        output_model=TriageResult,
        # Perceptual check — needs the multimodal model.
        model=llm.vision_model,
        max_tokens=768,
    )
