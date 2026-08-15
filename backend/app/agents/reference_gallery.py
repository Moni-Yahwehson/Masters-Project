"""Reference photographs for text-only answers.

"What does lumpy skin disease look like?" is a question about appearance, and
answering it in prose alone is a poor answer — the whole point of the condition
is that it is recognised by eye. The dataset already contains 1000+ labelled
photographs, so the knowledge path can show them.

Deliberate boundaries:

* these are **training images**, presented as illustrations of the condition,
  never as a match for anything the user uploaded (that is the precedent strip
  in the diagnosis panel, which is a different thing and says so);
* the gallery is attached to the *answer*, not fed to the language model. The
  model still answers only from the retrieved text passages, so adding pictures
  cannot change what it claims;
* nothing here is a diagnosis, and the caption on every image says which class
  it came from.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from .. import reference_index
from ..config import get_settings
from ..schemas import ExampleImage
from .visual_explain import jpeg_bytes_to_data_url

logger = logging.getLogger(__name__)

#: Phrases that mean "show me", in the words farmers and examiners actually use.
#: Deliberately generous: a spurious thumbnail strip costs the reader a moment,
#: whereas asking "give me an example" and getting pure prose reads as broken.
_VISUAL_INTENT = re.compile(
    r"""
    \b(
        example s? | sample s? | illustrat \w* | demonstrat \w*
      | picture s? | photo (?: graph )? s? | image s? | pic s?
      | visual \w* | gallery | diagram
      | look s? \s+ like | looks? | appear ance? | appears?
      | show \s+ me | show \b | see \s+ (?: what|one|some )
      | what \s+ does .{0,30} look
      | how \s+ (?: do (?:es)? | can ) \s+ i \s+ (?: tell | spot | recognise | recognize | identify )
      | recognise | recognize | identify | spot \s+ it
      | sign s? | symptom s? | lesion s? | nodule s? | bump s?
      | difference \s+ between | compare d? | versus | \bvs\b
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

#: Questions that are purely administrative get no gallery — pictures of sick
#: cattle are not an answer to "is LSD notifiable?".
_NON_VISUAL = re.compile(
    r"\b(notifiable|report(ing)?\s+requirement|legislation|law|regulation|"
    r"trade|export|import|quarantine\s+period|vaccine\s+schedule|cost|price)\b",
    re.IGNORECASE,
)

_HEALTHY_HINT = re.compile(r"\bhealthy|normal|unaffected|clean\s+skin\b", re.IGNORECASE)

_CAPTIONS = {
    "lumpycows": "LSD-affected (training set)",
    "healthycows": "Healthy (training set)",
}


def wants_examples(question: str) -> bool:
    """Whether a text question is asking, however indirectly, to be shown."""
    if not question or not question.strip():
        return False
    if _NON_VISUAL.search(question) and not _VISUAL_INTENT.search(question):
        return False
    return bool(_VISUAL_INTENT.search(question))


def build_gallery(question: str) -> tuple[list[ExampleImage], str]:
    """Return `(examples, note)` for a question, or `([], reason)`.

    Never raises: a missing or unreadable index means the text answer is
    returned exactly as it would have been before.
    """
    settings = get_settings()
    if not settings.enable_visual_evidence:
        return [], ""

    try:
        index = reference_index.get_index()
    except Exception as exc:  # noqa: BLE001 - defensive; get_index already swallows
        logger.warning("Reference gallery unavailable: %s", exc)
        return [], ""

    if index is None:
        return [], ""

    # A question naming healthy animals, or asking for a comparison, gets both
    # classes side by side; anything else gets the disease it asked about.
    wants_healthy = bool(_HEALTHY_HINT.search(question))
    wants_comparison = bool(
        re.search(r"\b(difference|compare|versus|vs|between)\b", question, re.IGNORECASE)
    )

    total = max(2, settings.visual_neighbours)
    examples: list[ExampleImage] = []

    if wants_comparison:
        half = total // 2
        examples += _pick(index, settings.positive_class, half, settings)
        examples += _pick(index, settings.negative_class, total - half, settings)
    elif wants_healthy:
        examples += _pick(index, settings.negative_class, total, settings)
    else:
        examples += _pick(index, settings.positive_class, total, settings)

    if not examples:
        return [], ""

    note = (
        f"Reference photographs from the {len(index)}-image labelled dataset this "
        "system was trained on, chosen as typical examples of their class. They "
        "illustrate the condition — they are not a diagnosis, and not a "
        "comparison against any animal of yours."
    )
    return examples, note


def _pick(index, label: str, count: int, settings) -> list[ExampleImage]:
    found = index.representatives(
        label, count, positive_class=settings.positive_class
    )
    return [
        ExampleImage(
            reference_id=item.reference_id,
            label=item.label,
            caption=_CAPTIONS.get(item.label, item.label),
            thumbnail_data_url=jpeg_bytes_to_data_url(item.thumbnail_jpeg),
        )
        for item in found
    ]


def attach(answer, question: Optional[str] = None):
    """Attach a gallery to a `KnowledgeAnswer` in place, when one is wanted."""
    text = question if question is not None else answer.question
    if not wants_examples(text):
        return answer
    examples, note = build_gallery(text)
    if examples:
        answer.examples = examples
        answer.examples_note = note
    return answer
