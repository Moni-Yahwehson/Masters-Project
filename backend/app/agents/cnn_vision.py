"""Agent 2 — deterministic CNN screening using the trained Keras model.

Wraps `vision/cattle_lsd_detector.keras` (MobileNetV2 backbone with a
frozen ImageNet base and a 128-unit dense head ending in a single sigmoid unit).

Preprocessing must mirror training exactly:

* training used `keras.utils.image_dataset_from_directory(..., image_size=(224, 224))`,
  which performs a plain bilinear resize without preserving aspect ratio;
* the `Rescaling(1/127.5, offset=-1)` layer lives *inside* the saved model, so we
  feed raw 0-255 float pixels and must **not** normalise here;
* `RandomRotation` and `RandomFlip` are also inside the model but are inert at
  inference time (`training=False`), so they need no special handling.

Class order comes from the alphabetically sorted directory listing of
`cow dataset/`: index 0 = healthycows, index 1 = lumpycows. The sigmoid output is
therefore P(lumpycows).
"""

from __future__ import annotations

import io
import logging
import re
import threading
import time
from contextlib import contextmanager
from typing import Optional, Sequence

from ..config import get_settings
from ..llm import ImagePayload
from ..schemas import (
    CnnPrediction,
    CnnResult,
    SaliencyMap,
    VisualEvidence,
    VisualNeighbour,
)

logger = logging.getLogger(__name__)


class CnnModelUnavailable(RuntimeError):
    """Raised when the Keras model cannot be loaded or run."""


# Layers whose serialised config gains new keys between Keras releases. The
# saved model was trained on a newer Keras than may be installed here (Keras
# pins to the Python version — 3.10 is the ceiling on Python 3.9, and it does
# not know BatchNormalization's `renorm*` keys), so deserialisation fails with
# "Unrecognized keyword arguments". Rather than pin everyone to one Keras
# build, wrap the affected layers so unknown config keys are dropped.
_COMPAT_LAYERS = (
    "BatchNormalization",
    "Dense",
    "Conv2D",
    "DepthwiseConv2D",
    "SeparableConv2D",
    "Dropout",
    "ReLU",
    "Rescaling",
    "RandomRotation",
    "RandomFlip",
    "GlobalAveragePooling2D",
    "InputLayer",
    "ZeroPadding2D",
    "Add",
    "Multiply",
    "Reshape",
    "Activation",
)

# Keras raises "Unrecognized keyword arguments passed to Dense: {'foo': None}".
# Scope the key harvest to that trailing dict — the surrounding message also
# echoes the *full* config, and matching against that strips every key and
# leaves an empty config behind.
_UNRECOGNISED_BLOCK_RE = re.compile(
    r"[Uu]nrecognized keyword arguments passed to \w+:\s*\{(?P<args>[^}]*)\}"
)
_KEY_RE = re.compile(r"'([A-Za-z_][A-Za-z0-9_]*)'\s*:")
_MAX_STRIP_ROUNDS = 12


def _unrecognised_keys(message: str) -> set[str]:
    keys: set[str] = set()
    for block in _UNRECOGNISED_BLOCK_RE.finditer(message):
        keys.update(_KEY_RE.findall(block.group("args")))
    return keys


def _forgiving_from_config(original):
    """Wrap a `from_config` so it retries after dropping rejected config keys."""

    def from_config(cls, config):
        attempt = dict(config)
        # Each failure names the offending keys, so drop exactly those and retry
        # rather than trying to guess the full cross-version delta up front.
        for _ in range(_MAX_STRIP_ROUNDS):
            try:
                return original(attempt)
            except (TypeError, ValueError) as exc:
                offenders = {
                    key for key in _unrecognised_keys(str(exc)) if key in attempt
                }
                if not offenders:
                    raise
                logger.debug("Dropping unsupported config keys: %s", sorted(offenders))
                for key in offenders:
                    attempt.pop(key, None)
        return original(attempt)

    return from_config


@contextmanager
def _relaxed_layer_configs(keras_module):
    """Temporarily make the listed layers tolerant of unknown config keys.

    Keras resolves built-in layers by module path, and only consults
    `custom_objects` via the config's `registered_name` — which is null for
    built-ins. So a custom_objects subclass never fires; patching `from_config`
    on the concrete classes for the duration of the load is what actually works.
    """
    patched: list[tuple[type, object, bool]] = []
    for name in _COMPAT_LAYERS:
        base = getattr(keras_module.layers, name, None)
        if base is None:
            continue
        owned = "from_config" in vars(base)
        original = base.from_config
        patched.append((base, vars(base).get("from_config"), owned))
        base.from_config = classmethod(_forgiving_from_config(original))
    try:
        yield
    finally:
        for base, saved, owned in patched:
            if owned:
                setattr(base, "from_config", saved)
            else:
                # It was inherited; remove the override to restore inheritance.
                try:
                    delattr(base, "from_config")
                except AttributeError:  # pragma: no cover - defensive
                    pass


class CnnScreeningAgent:
    """Loads the Keras classifier once and scores uploaded images."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._model = None
        self._load_error: Optional[str] = None
        self._attempted = False
        self._lock = threading.Lock()

    # -- model loading ----------------------------------------------------
    def _load_model(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            if self._attempted and self._load_error:
                raise CnnModelUnavailable(self._load_error)
            self._attempted = True

            path = self._settings.model_path
            if not path.is_file():
                self._load_error = (
                    f"Keras model not found at {path}. Set LSD_MODEL_PATH to the "
                    "location of cattle_lsd_detector.keras."
                )
                raise CnnModelUnavailable(self._load_error)

            try:
                import keras  # Keras 3 ships standalone and with TensorFlow
            except ImportError:
                try:
                    from tensorflow import keras  # type: ignore
                except ImportError as exc:
                    self._load_error = (
                        "Neither `keras` nor `tensorflow` is installed, so the CNN "
                        "agent cannot run. Install them with: "
                        "pip install -r backend/requirements.txt"
                    )
                    raise CnnModelUnavailable(self._load_error) from exc

            try:
                started = time.perf_counter()
                # compile=False: we only need the forward pass, and skipping
                # compilation avoids optimizer-state deserialisation issues
                # across Keras versions.
                with _relaxed_layer_configs(keras):
                    self._model = keras.models.load_model(str(path), compile=False)
                logger.info(
                    "Loaded CNN from %s in %.0f ms",
                    path,
                    (time.perf_counter() - started) * 1000,
                )
            except Exception as exc:
                self._load_error = f"Failed to load {path.name}: {exc}"
                raise CnnModelUnavailable(self._load_error) from exc
            return self._model

    @property
    def available(self) -> bool:
        try:
            self._load_model()
            return True
        except CnnModelUnavailable:
            return False

    @property
    def unavailable_reason(self) -> str:
        return self._load_error or ""

    # -- preprocessing ----------------------------------------------------
    def _preprocess(self, images: Sequence[ImagePayload]):
        import numpy as np
        from PIL import Image

        size = self._settings.model_image_size
        batch = []
        for image in images:
            with Image.open(io.BytesIO(image.to_bytes())) as handle:
                # Bilinear resize without preserving aspect ratio, matching
                # image_dataset_from_directory's default behaviour.
                frame = handle.convert("RGB").resize((size, size), Image.BILINEAR)
                batch.append(np.asarray(frame, dtype="float32"))
        return np.stack(batch, axis=0)

    # -- inference --------------------------------------------------------
    def run(self, images: Sequence[ImagePayload]) -> CnnResult:
        if not images:
            return CnnResult(
                available=False,
                predictions=[],
                threshold=self._settings.decision_threshold,
                note="No images supplied.",
            )

        import numpy as np

        model = self._load_model()
        started = time.perf_counter()
        batch = self._preprocess(images)
        # Flatten to one scalar per image: the head is a single sigmoid unit, so
        # predict() returns shape (n, 1).
        raw = np.asarray(model.predict(batch, verbose=0)).reshape(len(images), -1)
        elapsed_ms = (time.perf_counter() - started) * 1000

        threshold = self._settings.decision_threshold
        band = self._settings.uncertainty_band
        predictions: list[CnnPrediction] = []
        for image, row in zip(images, raw):
            probability = min(max(float(row[0]), 0.0), 1.0)
            label = (
                self._settings.positive_class
                if probability >= threshold
                else self._settings.negative_class
            )
            predictions.append(
                CnnPrediction(
                    image_id=image.id,
                    probability_lsd=round(probability, 4),
                    predicted_label=label,
                    confident=abs(probability - threshold) >= band,
                )
            )

        # Screening-conservative aggregation: take the highest per-image score,
        # so a single clearly affected photo is not diluted by other frames of
        # the same animal that happen to show unaffected body regions.
        aggregate = max(pred.probability_lsd for pred in predictions)
        aggregate_label = (
            self._settings.positive_class
            if aggregate >= threshold
            else self._settings.negative_class
        )

        return CnnResult(
            available=True,
            predictions=predictions,
            aggregate_probability_lsd=round(aggregate, 4),
            aggregate_label=aggregate_label,
            threshold=threshold,
            inference_ms=round(elapsed_ms, 1),
            note=(
                "Aggregate is the maximum per-image probability (screening-"
                "conservative). Sigmoid output is P(lumpycows) under the training "
                "class order [healthycows=0, lumpycows=1]."
            ),
        )

    # -- explanation ------------------------------------------------------
    def run_with_evidence(
        self, images: Sequence[ImagePayload]
    ) -> tuple[CnnResult, Optional[VisualEvidence]]:
        """Score the images and, if possible, explain the score visually.

        The explanation is strictly additive: `run()` remains the authority on
        the probability, and any failure below degrades to "no evidence"
        instead of losing the screening result. It costs one extra
        forward+backward pass (~0.3 s for a handful of images) — cheap against a
        pipeline dominated by language-model inference, and worth not having two
        code paths that could disagree about the number shown to a farmer.
        """
        result = self.run(images)
        if not result.available or not self._settings.enable_visual_evidence:
            return result, None
        try:
            return result, self._explain(images, result)
        except Exception as exc:  # noqa: BLE001 - never fail the screening step
            logger.warning("Visual evidence unavailable: %s", exc, exc_info=True)
            return result, VisualEvidence(
                available=False, note=f"Visual explanation unavailable: {exc}"
            )

    def _explain(
        self, images: Sequence[ImagePayload], result: CnnResult
    ) -> VisualEvidence:
        from .. import reference_index
        from .visual_explain import (
            embed_and_explain,
            jpeg_bytes_to_data_url,
            render_saliency,
            split_stack,
        )

        started = time.perf_counter()
        model = self._load_model()
        batch = self._preprocess(images)
        probabilities, embeddings, cams = embed_and_explain(
            model, batch, threshold=self._settings.decision_threshold
        )

        saliency: list[SaliencyMap] = []
        for index, image in enumerate(images):
            probability = float(probabilities[index])
            original_url, overlay_url = render_saliency(
                image.to_bytes(),
                cams[index],
                render_size=self._settings.saliency_render_size,
            )
            saliency.append(
                SaliencyMap(
                    image_id=image.id,
                    original_data_url=original_url,
                    overlay_data_url=overlay_url,
                    probability_lsd=round(min(max(probability, 0.0), 1.0), 4),
                    explains=(
                        "lsd_like"
                        if probability >= self._settings.decision_threshold
                        else "healthy_like"
                    ),
                )
            )

        precedents = self._precedents(
            images, embeddings, probabilities, result, reference_index
        )
        neighbours = precedents["neighbours"]
        if neighbours:
            neighbour_models = [
                VisualNeighbour(
                    query_image_id=query_id,
                    reference_id=neighbour.reference_id,
                    label=neighbour.label,
                    similarity=neighbour.similarity,
                    thumbnail_data_url=jpeg_bytes_to_data_url(neighbour.thumbnail_jpeg),
                )
                for query_id, neighbour in neighbours
            ]
        else:
            neighbour_models = []

        return VisualEvidence(
            available=True,
            saliency=saliency,
            neighbours=neighbour_models,
            neighbour_label_counts=precedents["counts"],
            neighbour_agreement=precedents["agreement"],
            index_size=precedents["index_size"],
            query_in_index=precedents["query_in_index"],
            embedding_layer=precedents["layer"] or split_stack(model).embedding_layer,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
            note=precedents["note"],
        )

    #: Cosine similarity above which a "neighbour" is the query image itself.
    #: Distinct images in this dataset top out around 0.78, so 0.999 only ever
    #: catches a re-upload of an indexed file.
    _SELF_MATCH_SIMILARITY = 0.999

    def _precedents(self, images, embeddings, probabilities, result, reference_index):
        """Look up nearest labelled training images for the decisive upload.

        Only one image is queried, not all of them: the reported probability is
        the per-image maximum (screening-conservative), so the precedents that
        explain the verdict are the ones for that same frame. Querying every
        upload would return up to 36 thumbnails and bury the signal.
        """
        empty = {
            "neighbours": [],
            "counts": {},
            "agreement": None,
            "index_size": 0,
            "query_in_index": False,
            "layer": "",
            "note": "",
        }

        index = reference_index.get_index()
        if index is None:
            return {
                **empty,
                "note": "Precedent lookup unavailable: "
                + (reference_index.unavailable_reason() or "no index built."),
            }

        decisive = max(range(len(images)), key=lambda i: float(probabilities[i]))
        # Ask for one extra so that dropping a self-match still fills the strip.
        found = index.query(embeddings[decisive], self._settings.visual_neighbours + 1)
        if not found:
            return {
                **empty,
                "index_size": len(index),
                "layer": index.embedding_layer,
                "note": "No precedent images matched.",
            }

        # Uploading a file that is already in the training set makes the model
        # look better than it is, and it is exactly what happens when someone
        # demos with a dataset image. Drop the self-match from the vote and say
        # so, rather than showing the image as its own strongest precedent.
        self_match = found[0] if found[0].similarity >= self._SELF_MATCH_SIMILARITY else None
        if self_match is not None:
            found = found[1:]
        found = found[: self._settings.visual_neighbours]

        counts = reference_index.summarise_labels(found)
        positive = counts.get(self._settings.positive_class, 0)
        negative = counts.get(self._settings.negative_class, 0)
        if positive == negative:
            agreement = "split"
        else:
            vote = (
                self._settings.positive_class
                if positive > negative
                else self._settings.negative_class
            )
            agreement = "agree" if vote == result.aggregate_label else "disagree"

        # The Vue client uploads data URLs without a filename, so fall back to
        # neutral wording rather than leaking an internal id into a farmer's
        # report.
        subject = (
            f"'{images[decisive].filename}'"
            if images[decisive].filename
            else ("the uploaded image" if len(images) == 1 else "the decisive image")
        )
        note = (
            f"Nearest {len(found)} of {len(index)} labelled training images to "
            f"{subject}, by cosine similarity in the classifier's trained "
            f"'{index.embedding_layer}' layer. These are the images the score is "
            "pattern-matching against — they are training data, not independent "
            "confirmation."
        )
        if self_match is not None:
            note = (
                f"This image is already in the training set as "
                f"'{self_match.reference_id}' ({self_match.label}), so the model "
                "has seen it and its score here is not a fair test. It has been "
                "excluded from the precedents below. " + note
            )

        return {
            "neighbours": [(images[decisive].id, neighbour) for neighbour in found],
            "counts": counts,
            "agreement": agreement,
            "index_size": len(index),
            "query_in_index": self_match is not None,
            "layer": index.embedding_layer,
            "note": note,
        }

    def unavailable_result(self, reason: str = "") -> CnnResult:
        return CnnResult(
            available=False,
            predictions=[],
            threshold=self._settings.decision_threshold,
            note=reason or self.unavailable_reason or "CNN model unavailable.",
        )


_agent: Optional[CnnScreeningAgent] = None


def get_cnn_agent() -> CnnScreeningAgent:
    global _agent
    if _agent is None:
        _agent = CnnScreeningAgent()
    return _agent
