"""Reference gallery — example photographs attached to text-only answers.

The intent tests are the important ones. The failure that prompted this feature
was silent: asking "tell me about LSD and give me an example" returned prose and
nothing else, which reads as broken rather than as a missing feature.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import reference_index  # noqa: E402
from app.agents.reference_gallery import (  # noqa: E402
    build_gallery,
    wants_examples,
)
from app.agents.visual_explain import normalise_rows  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.schemas import KnowledgeAnswer  # noqa: E402

SETTINGS = get_settings()

needs_index = pytest.mark.skipif(
    not SETTINGS.visual_index_path.is_file(),
    reason="precedent index not built (run `make index`)",
)


# -- intent ------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "tell me about lsd and also give me an example",   # the original report
        "what does lumpy skin disease look like?",
        "show me a picture of an affected cow",
        "can you show examples of the nodules",
        "how do I recognise LSD?",
        "what are the signs and symptoms",
        "difference between healthy and lumpy cattle",
        "what does a healthy cow look like",
        "I want to see some photos",
    ],
)
def test_visual_questions_get_examples(question: str) -> None:
    assert wants_examples(question)


@pytest.mark.parametrize(
    "question",
    [
        "how does LSD spread?",
        "is lumpy skin disease notifiable?",
        "can I vaccinate a pregnant cow?",
        "what is the incubation period",
        "how long does quarantine last",
        "",
        "   ",
    ],
)
def test_non_visual_questions_get_nothing(question: str) -> None:
    assert not wants_examples(question)


# -- selection ---------------------------------------------------------------


def _fake_index(probabilities=None) -> reference_index.ReferenceIndex:
    """Four lumpy images — two of them near-duplicates — and two healthy."""
    import numpy as np

    vectors = np.array(
        [
            [1.00, 0.00, 0.0],   # lumpy, typical
            [0.99, 0.14, 0.0],   # lumpy, near-duplicate of the first
            [0.80, 0.60, 0.0],   # lumpy, distinct
            [0.60, 0.80, 0.0],   # lumpy, distinct
            [0.00, 0.00, 1.0],   # healthy
            [0.10, 0.00, 0.99],  # healthy
        ],
        dtype="float32",
    )
    return reference_index.ReferenceIndex(
        embeddings=normalise_rows(vectors),
        labels=["lumpycows"] * 4 + ["healthycows"] * 2,
        names=[f"img{i}.jpg" for i in range(6)],
        thumbnails=[b"x"] * 6,
        embedding_layer="dense",
        probabilities=probabilities,
    )


def test_representatives_only_returns_the_requested_class() -> None:
    picked = _fake_index().representatives("healthycows", 2)
    assert {item.label for item in picked} == {"healthycows"}


def test_representatives_skip_near_duplicates() -> None:
    """Six frames of the same animal illustrate nothing."""
    picked = _fake_index().representatives("lumpycows", 3, max_similarity=0.9)
    names = [item.reference_id for item in picked]
    assert len(names) == len(set(names))
    # img0 and img1 are ~0.99 similar, so both must not appear.
    assert not {"img0.jpg", "img1.jpg"} <= set(names)


def test_representatives_top_up_rather_than_return_too_few() -> None:
    """Diversity filtering must not starve the gallery."""
    picked = _fake_index().representatives("lumpycows", 4, max_similarity=0.0)
    assert len(picked) == 4
    assert len({item.reference_id for item in picked}) == 4


def test_representatives_prefer_confidently_scored_images() -> None:
    import numpy as np

    # Mark the otherwise-most-typical lumpy image as one the model is unsure of.
    probabilities = np.array([0.30, 0.97, 0.98, 0.99, 0.02, 0.01], dtype="float32")
    picked = _fake_index(probabilities).representatives("lumpycows", 3)
    assert "img0.jpg" not in {item.reference_id for item in picked}


def test_representatives_work_without_probabilities() -> None:
    """A v1-shaped index has no scores; selection must still function."""
    assert len(_fake_index(None).representatives("lumpycows", 2)) == 2


def test_representatives_handle_an_absent_class() -> None:
    assert _fake_index().representatives("goats", 3) == []


def test_representatives_handle_a_zero_count() -> None:
    assert _fake_index().representatives("lumpycows", 0) == []


# -- integration -------------------------------------------------------------


@needs_index
def test_gallery_is_built_for_the_question_that_prompted_this() -> None:
    reference_index.reset_cache()
    examples, note = build_gallery("tell me about lsd and also give me an example")

    assert examples, "an explicit request for an example must return images"
    assert all(item.label == SETTINGS.positive_class for item in examples)
    assert all(
        item.thumbnail_data_url.startswith("data:image/jpeg;base64,") for item in examples
    )
    assert "not a diagnosis" in note
    assert "trained on" in note, "the note must say these are training images"


@needs_index
def test_comparison_questions_show_both_classes() -> None:
    reference_index.reset_cache()
    examples, _ = build_gallery("what is the difference between healthy and affected cattle")
    labels = {item.label for item in examples}
    assert labels == {SETTINGS.positive_class, SETTINGS.negative_class}


@needs_index
def test_healthy_questions_show_healthy_animals() -> None:
    reference_index.reset_cache()
    examples, _ = build_gallery("what does a healthy cow look like")
    assert {item.label for item in examples} == {SETTINGS.negative_class}


@needs_index
def test_gallery_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    reference_index.reset_cache()
    monkeypatch.setattr(SETTINGS, "enable_visual_evidence", False)
    examples, _ = build_gallery("show me an example")
    assert examples == []


def test_missing_index_returns_no_gallery(monkeypatch: pytest.MonkeyPatch) -> None:
    """No index must mean a plain text answer, never an error."""
    monkeypatch.setattr(reference_index, "get_index", lambda: None)
    examples, note = build_gallery("show me an example")
    assert examples == []
    assert note == ""


def test_attach_leaves_non_visual_answers_untouched() -> None:
    from app.agents import reference_gallery

    answer = KnowledgeAnswer(question="how does LSD spread?", markdown="It spreads.")
    reference_gallery.attach(answer)
    assert answer.examples == []
    assert answer.examples_note == ""
