"""Visual evidence tests — Grad-CAM saliency and the precedent index.

Split into three groups by what they need:

* pure logic (rendering, index round-trip) always runs;
* anything needing the trained model skips when it is absent;
* anything needing the built index skips when it has not been generated,
  since `make index` is optional.
"""

from __future__ import annotations

import base64
import glob
import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import reference_index  # noqa: E402
from app.agents.cnn_vision import get_cnn_agent  # noqa: E402
from app.agents.visual_explain import (  # noqa: E402
    jpeg_bytes_to_data_url,
    make_thumbnail,
    normalise_rows,
    render_saliency,
    split_stack,
)
from app.config import get_settings  # noqa: E402
from app.llm import ImagePayload  # noqa: E402

SETTINGS = get_settings()
DATASET = SETTINGS.dataset_dir

needs_model = pytest.mark.skipif(
    not get_cnn_agent().available,
    reason=f"CNN unavailable: {get_cnn_agent().unavailable_reason}",
)
needs_index = pytest.mark.skipif(
    not SETTINGS.visual_index_path.is_file(),
    reason="precedent index not built (run `make index`)",
)
needs_dataset = pytest.mark.skipif(not DATASET.is_dir(), reason="dataset not present")


def _solid_jpeg(width: int = 96, height: int = 64, colour=(130, 100, 80)) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buffer, format="JPEG")
    return buffer.getvalue()


def _decode(data_url: str) -> bytes:
    assert data_url.startswith("data:image/jpeg;base64,")
    return base64.b64decode(data_url.split(",", 1)[1])


def _dataset_payload(pattern: str, offset: int = 0) -> ImagePayload:
    paths = sorted(glob.glob(str(DATASET / pattern)))
    path = Path(paths[offset])
    return ImagePayload.from_bytes(path.read_bytes(), "img-01", "image/jpeg", path.name)


# -- rendering, no model needed ---------------------------------------------


def test_render_saliency_returns_two_distinct_jpegs() -> None:
    import numpy as np

    cam = np.zeros((7, 7), dtype="float32")
    cam[3, 3] = 1.0
    original, overlay = render_saliency(_solid_jpeg(600, 400), cam, render_size=128)

    assert original != overlay, "a non-flat map must visibly change the image"
    from PIL import Image

    with Image.open(io.BytesIO(_decode(overlay))) as frame:
        assert max(frame.size) == 128, "a large photo must be capped to render_size"


def test_render_does_not_upscale_a_small_photo() -> None:
    """Blowing a thumbnail up to 384 px would fake detail that is not there."""
    import numpy as np
    from PIL import Image

    cam = np.random.default_rng(1).random((7, 7)).astype("float32")
    _, overlay = render_saliency(_solid_jpeg(96, 64), cam, render_size=384)
    with Image.open(io.BytesIO(_decode(overlay))) as frame:
        assert frame.size == (96, 64)


def test_flat_saliency_map_is_not_painted() -> None:
    """A gradient that says nothing must not be dressed up as a heat map."""
    import numpy as np

    original, overlay = render_saliency(
        _solid_jpeg(), np.zeros((7, 7), dtype="float32"), render_size=128
    )
    assert original == overlay


def test_render_preserves_aspect_ratio() -> None:
    import numpy as np
    from PIL import Image

    cam = np.random.default_rng(0).random((7, 7)).astype("float32")
    _, overlay = render_saliency(_solid_jpeg(200, 100), cam, render_size=128)
    with Image.open(io.BytesIO(_decode(overlay))) as frame:
        assert frame.size == (128, 64)


def test_thumbnail_is_bounded_and_decodable() -> None:
    from PIL import Image

    raw = make_thumbnail(_solid_jpeg(500, 300), 64)
    with Image.open(io.BytesIO(raw)) as frame:
        assert max(frame.size) == 64
    assert jpeg_bytes_to_data_url(raw).startswith("data:image/jpeg;base64,")


def test_normalise_rows_produces_unit_vectors() -> None:
    import numpy as np

    rows = normalise_rows(np.array([[3.0, 4.0], [0.0, 0.0]], dtype="float32"))
    assert np.isclose(np.linalg.norm(rows[0]), 1.0)
    # A zero row must not divide by zero.
    assert np.all(np.isfinite(rows[1]))


# -- index round-trip --------------------------------------------------------


def test_index_round_trip(tmp_path: Path) -> None:
    """Write and re-read an index without pickle, preserving every field."""
    import numpy as np

    thumbnails = [_solid_jpeg(16, 16), _solid_jpeg(16, 16, (10, 20, 30))]
    blob = b"".join(thumbnails)
    offsets = np.array([0, len(thumbnails[0]), len(blob)], dtype="int64")
    path = tmp_path / "index.npz"
    np.savez_compressed(
        path,
        format_version=np.asarray([reference_index.INDEX_FORMAT_VERSION], dtype="int32"),
        embeddings=normalise_rows(np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")),
        labels=np.asarray(["healthycows", "lumpycows"]),
        names=np.asarray(["a.jpg", "b.jpg"]),
        thumbnail_blob=np.frombuffer(blob, dtype="uint8"),
        thumbnail_offsets=offsets,
        embedding_layer=np.asarray(["dense"]),
    )

    index = reference_index.load_index(path)
    assert len(index) == 2
    assert index.embedding_layer == "dense"
    assert index.thumbnails[0] == thumbnails[0]
    assert index.thumbnails[1] == thumbnails[1]

    hits = index.query(np.array([1.0, 0.0], dtype="float32"), top_k=2)
    assert [hit.reference_id for hit in hits] == ["a.jpg", "b.jpg"]
    assert hits[0].similarity == pytest.approx(1.0, abs=1e-4)
    assert reference_index.summarise_labels(hits) == {"healthycows": 1, "lumpycows": 1}


def test_index_rejects_a_future_format(tmp_path: Path) -> None:
    import numpy as np

    path = tmp_path / "index.npz"
    np.savez_compressed(
        path,
        format_version=np.asarray([reference_index.INDEX_FORMAT_VERSION + 1], dtype="int32"),
        embeddings=np.zeros((1, 2), dtype="float32"),
        labels=np.asarray(["x"]),
        names=np.asarray(["x.jpg"]),
        thumbnail_blob=np.zeros(0, dtype="uint8"),
        thumbnail_offsets=np.array([0, 0], dtype="int64"),
        embedding_layer=np.asarray(["dense"]),
    )
    with pytest.raises(reference_index.IndexUnavailable, match="format v"):
        reference_index.load_index(path)


def test_missing_index_is_reported_not_raised_to_callers(tmp_path: Path) -> None:
    with pytest.raises(reference_index.IndexUnavailable, match="make index"):
        reference_index.load_index(tmp_path / "absent.npz")


def test_query_more_neighbours_than_exist(tmp_path: Path) -> None:
    import numpy as np

    index = reference_index.ReferenceIndex(
        embeddings=normalise_rows(np.array([[1.0, 0.0]], dtype="float32")),
        labels=["lumpycows"],
        names=["only.jpg"],
        thumbnails=[b""],
        embedding_layer="dense",
    )
    assert len(index.query(np.array([1.0, 0.0], dtype="float32"), top_k=25)) == 1


# -- model-backed ------------------------------------------------------------


@needs_model
def test_split_stack_finds_the_backbone_and_embedding_layer() -> None:
    stack = split_stack(get_cnn_agent()._load_model())
    assert stack.pre, "the rescaling layers must be separated out"
    assert "mobilenet" in stack.backbone_name.lower()
    assert stack.head, "the classifier head must be separated out"
    assert stack.embedding_layer


@needs_model
@needs_dataset
def test_evidence_matches_the_screening_probability() -> None:
    """The saliency card must never show a number the CNN card contradicts."""
    payload = _dataset_payload("lumpycows/*.jpg")
    result, visual = get_cnn_agent().run_with_evidence([payload])

    assert visual is not None and visual.available, visual.note if visual else "no evidence"
    assert len(visual.saliency) == 1
    saliency = visual.saliency[0]
    assert saliency.image_id == payload.id
    assert saliency.probability_lsd == pytest.approx(
        result.predictions[0].probability_lsd, abs=5e-4
    )
    assert saliency.explains == (
        "lsd_like" if saliency.probability_lsd >= result.threshold else "healthy_like"
    )
    assert _decode(saliency.original_data_url)
    assert _decode(saliency.overlay_data_url)


@needs_model
@needs_dataset
def test_saliency_is_not_a_uniform_wash() -> None:
    """A blank overlay would mean the sigmoid saturated and gradients vanished.

    This is the failure the linear-activation swap in visual_explain exists to
    prevent, and it is invisible without an assertion: the pipeline still
    returns a perfectly well-formed but meaningless image.
    """
    import numpy as np
    from PIL import Image

    _, visual = get_cnn_agent().run_with_evidence([_dataset_payload("lumpycows/*.jpg")])
    saliency = visual.saliency[0]

    with Image.open(io.BytesIO(_decode(saliency.original_data_url))) as frame:
        before = np.asarray(frame.convert("RGB"), dtype="int16")
    with Image.open(io.BytesIO(_decode(saliency.overlay_data_url))) as frame:
        after = np.asarray(frame.convert("RGB"), dtype="int16")

    difference = np.abs(after - before).mean(axis=2)
    changed = float((difference > 12).mean())
    assert 0.01 < changed < 0.95, (
        f"{changed:.1%} of pixels changed — the heat map should mark a region, "
        "not the whole frame or nothing at all"
    )


@needs_model
def test_evidence_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = get_cnn_agent()
    monkeypatch.setattr(agent._settings, "enable_visual_evidence", False)
    result, visual = agent.run_with_evidence(
        [ImagePayload.from_bytes(_solid_jpeg(), "img-01", "image/jpeg", "flat.jpg")]
    )
    assert result.available
    assert visual is None


@needs_model
def test_screening_survives_a_broken_explanation(monkeypatch: pytest.MonkeyPatch) -> None:
    """The probability must reach the farmer even if the explanation fails."""
    agent = get_cnn_agent()

    def explode(*args, **kwargs):
        raise RuntimeError("synthetic explanation failure")

    monkeypatch.setattr(agent, "_explain", explode)
    result, visual = agent.run_with_evidence(
        [ImagePayload.from_bytes(_solid_jpeg(), "img-01", "image/jpeg", "flat.jpg")]
    )
    assert result.available and result.aggregate_probability_lsd is not None
    assert visual is not None and not visual.available
    assert "synthetic explanation failure" in visual.note


# -- index-backed ------------------------------------------------------------


@needs_model
@needs_index
@needs_dataset
def test_precedents_exclude_the_image_itself() -> None:
    """Re-uploading a training image must be flagged, not silently rewarded."""
    reference_index.reset_cache()
    payload = _dataset_payload("lumpycows/*.jpg", offset=5)
    _, visual = get_cnn_agent().run_with_evidence([payload])

    assert visual.query_in_index, "an indexed image must be recognised as seen"
    assert payload.filename not in {n.reference_id for n in visual.neighbours}
    assert "already in the training set" in visual.note
    assert len(visual.neighbours) == get_settings().visual_neighbours
    assert all(n.similarity < 0.999 for n in visual.neighbours)


@needs_model
@needs_index
def test_a_novel_image_is_not_flagged_as_seen() -> None:
    reference_index.reset_cache()
    _, visual = get_cnn_agent().run_with_evidence(
        [ImagePayload.from_bytes(_solid_jpeg(), "img-01", "image/jpeg", "flat.jpg")]
    )
    assert not visual.query_in_index


@needs_model
@needs_index
@needs_dataset
def test_precedent_vote_is_consistent_with_its_counts() -> None:
    reference_index.reset_cache()
    settings = get_settings()
    result, visual = get_cnn_agent().run_with_evidence(
        [_dataset_payload("lumpycows/*.jpg", offset=7)]
    )

    counts = visual.neighbour_label_counts
    assert sum(counts.values()) == len(visual.neighbours)
    positive = counts.get(settings.positive_class, 0)
    negative = counts.get(settings.negative_class, 0)
    if positive == negative:
        assert visual.neighbour_agreement == "split"
    else:
        winner = settings.positive_class if positive > negative else settings.negative_class
        expected = "agree" if winner == result.aggregate_label else "disagree"
        assert visual.neighbour_agreement == expected


@needs_index
def test_built_index_covers_both_classes() -> None:
    reference_index.reset_cache()
    index = reference_index.get_index()
    assert index is not None
    labels = set(index.labels)
    assert {"healthycows", "lumpycows"} <= labels
    assert len(index) == index.embeddings.shape[0] == len(index.thumbnails)
