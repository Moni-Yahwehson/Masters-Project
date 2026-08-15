"""CNN agent tests.

Skipped automatically when TensorFlow/Keras or the trained model are absent, so
the suite still runs in a lightweight environment.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.cnn_vision import get_cnn_agent  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.llm import ImagePayload  # noqa: E402

pytestmark = pytest.mark.skipif(
    not get_cnn_agent().available,
    reason=f"CNN unavailable: {get_cnn_agent().unavailable_reason}",
)

DATASET = get_settings().model_path.parent / "cow dataset"

# The training dataset is not distributed with this repository (262 MB, and it
# is training data rather than something the application needs to run). Every
# test below feeds the model real images from it, so they skip rather than fail
# when it is absent — a skipped test says "not checked here", a failing one
# wrongly says "this build is broken".
needs_dataset = pytest.mark.skipif(
    not DATASET.is_dir(), reason=f"dataset not present at {DATASET}"
)


def _payloads(pattern: str, limit: int, tag: str) -> list[ImagePayload]:
    paths = sorted(glob.glob(str(DATASET / pattern)))[:limit]
    return [
        ImagePayload.from_bytes(
            Path(path).read_bytes(), f"{tag}-{index}", "image/jpeg", Path(path).name
        )
        for index, path in enumerate(paths)
    ]


def test_model_loads() -> None:
    assert get_cnn_agent().available


@needs_dataset
def test_probabilities_are_bounded() -> None:
    images = _payloads("lumpycows/*.jpg", 3, "lumpy")
    result = get_cnn_agent().run(images)
    assert result.available
    assert len(result.predictions) == len(images)
    for prediction in result.predictions:
        assert 0.0 <= prediction.probability_lsd <= 1.0


@needs_dataset
def test_aggregate_is_max_of_predictions() -> None:
    images = _payloads("lumpycows/*.jpg", 4, "lumpy")
    result = get_cnn_agent().run(images)
    assert result.aggregate_probability_lsd == max(
        prediction.probability_lsd for prediction in result.predictions
    )


@needs_dataset
def test_separates_the_two_training_classes() -> None:
    """Sanity check on preprocessing, not a performance claim.

    These images are from the folders the model was trained on, so this only
    proves the inference path (resize, channel order, no double-rescaling)
    matches training. Held-out numbers come from scripts/evaluate_model.py.
    """
    lumpy = get_cnn_agent().run(_payloads("lumpycows/*.jpg", 6, "lumpy"))
    healthy = get_cnn_agent().run(_payloads("healthycows/*.jpg", 6, "healthy"))

    lumpy_mean = sum(p.probability_lsd for p in lumpy.predictions) / len(lumpy.predictions)
    healthy_mean = sum(p.probability_lsd for p in healthy.predictions) / len(
        healthy.predictions
    )
    assert lumpy_mean > 0.6, f"lumpy mean too low ({lumpy_mean:.3f}) — check preprocessing"
    assert healthy_mean < 0.4, (
        f"healthy mean too high ({healthy_mean:.3f}) — check preprocessing"
    )


def test_empty_input_is_marked_unavailable() -> None:
    result = get_cnn_agent().run([])
    assert not result.available
    assert result.predictions == []
