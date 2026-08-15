"""Visual explanation of the CNN's call — Grad-CAM saliency and embeddings.

Both outputs come out of the classifier that agent 2 already runs, so this adds
no model, no download and no network. It exists because a bare probability is
not reviewable: `0.87` tells a farmer nothing about *why*, and tells an examiner
nothing about whether the model is looking at lesions or at the background.

Architecture of `cattle_lsd_detector.keras` (a Sequential stack)::

    random_rotation → random_flip → rescaling    ← inert / scaling at inference
    mobilenetv2_1.00_224   (Functional)          ← (7, 7, 1280) feature map
    global_average_pooling2d                     ← (1280,)
    dense (128) → dropout → dense_1 (1, sigmoid) ← (128,) is the embedding

Two details matter and are easy to get wrong:

* **The stack is Sequential, not Functional.** Its layers were never called as
  a graph, so `layer.output` raises and the usual
  ``keras.Model(model.input, [conv.output, model.output])`` Grad-CAM recipe
  fails outright. We therefore split the layer list around the backbone and
  re-apply the head by hand, which also gives us the feature map and the
  embedding from a single pass.

* **Gradients must be taken on the logit, not the sigmoid.** This model
  saturates hard — a confident image scores 0.9999, where the sigmoid's
  derivative underflows to zero and the heat map comes out uniformly blank.
  Swapping the head to a linear activation for the duration of the tape
  recovers a usable gradient. Verified: the raw gradient span rises from
  ~0.001 to 0.13–0.86 on the same images.

The heat map is *not* a lesion segmentation. It is a 7×7 grid smoothed up to
image size, so it localises to roughly a third of the frame at best.
"""

from __future__ import annotations

import base64
import io
import logging
import threading
from dataclasses import dataclass
from typing import Any, Sequence

logger = logging.getLogger(__name__)

#: Grad-CAM mutates the head's activation in place, which is process-global
#: state on a shared model object. Serialise every explanation pass so two
#: concurrent requests cannot observe each other's swapped activation.
_EXPLAIN_LOCK = threading.Lock()

#: Only the top of the activation range is painted. Without a floor the whole
#: frame gets tinted and the photograph underneath stops being readable.
_HEAT_FLOOR = 0.55
_PEAK_ALPHA = 0.66

#: A warm ramp with no dark stops, so the overlay adds light rather than
#: muddying the image. Columns: position, R, G, B.
_HEAT_STOPS = [
    (0.00, 255, 224, 130),
    (0.45, 255, 160, 50),
    (0.75, 235, 70, 45),
    (1.00, 175, 20, 60),
]


class ExplanationUnavailable(RuntimeError):
    """Raised when the model's shape does not support Grad-CAM."""


@dataclass
class _Stack:
    """The classifier split around its convolutional backbone."""

    pre: list           # augmentation + rescaling, inert or scaling at inference
    backbone: Any       # the MobileNetV2 sub-model
    head: list          # pooling → dense → dropout → dense_1
    backbone_name: str
    embedding_layer: str


def split_stack(model) -> _Stack:
    """Split the Sequential classifier around its nested backbone sub-model."""
    import keras

    layers = list(model.layers)
    index = next(
        (i for i, layer in enumerate(layers) if isinstance(layer, keras.Model)), None
    )
    if index is None:
        raise ExplanationUnavailable(
            "No nested convolutional sub-model found, so there is no feature map "
            "to attribute the score to."
        )
    head = layers[index + 1 :]
    if not head:
        raise ExplanationUnavailable("The backbone has no classifier head above it.")
    return _Stack(
        pre=layers[:index],
        backbone=layers[index],
        head=head,
        backbone_name=layers[index].name,
        # The pooled backbone output is generic ImageNet feature space; the
        # trained dense layer below it is the one that separates these classes.
        # Measured leave-one-out kNN over the dataset: 0.955 for dense-128
        # against 0.846 for the 1280-d pooled features.
        embedding_layer=head[1].name if len(head) > 1 else head[0].name,
    )


def embed_and_explain(
    model,
    batch,
    *,
    threshold: float,
    want_saliency: bool = True,
):
    """One pass returning ``(probabilities, embeddings, cams)``.

    ``cams`` is ``None`` when saliency was not requested; it is a
    ``(n, h, w)`` float array of un-normalised Grad-CAM weights otherwise.
    """
    import keras
    import numpy as np
    import tensorflow as tf

    stack = split_stack(model)
    head_layer = stack.head[-1]

    with _EXPLAIN_LOCK:
        saved_activation = getattr(head_layer, "activation", None)
        swap = want_saliency and saved_activation is not None
        if swap:
            head_layer.activation = keras.activations.linear
        try:
            x = tf.convert_to_tensor(batch)
            for layer in stack.pre:
                x = layer(x, training=False)

            if not want_saliency:
                # No activation swap here, so the head's own sigmoid already
                # turned the output into a probability.
                features = stack.backbone(x, training=False)
                embeddings, outputs = _apply_head(stack, features)
                return (
                    np.asarray(outputs).ravel(),
                    np.asarray(embeddings),
                    None,
                )

            with tf.GradientTape() as tape:
                features = stack.backbone(x, training=False)
                tape.watch(features)
                embeddings, logits = _apply_head(stack, features)
                logits = tf.reshape(logits, [-1])
                probabilities = tf.sigmoid(logits)
                # Explain the call the model actually made: for a negative
                # prediction the evidence is whatever pushes the logit *down*,
                # so flip the sign rather than always explaining "LSD-like".
                signs = tf.where(probabilities >= threshold, 1.0, -1.0)
                target = tf.reduce_sum(logits * signs)
            gradients = tape.gradient(target, features)
        finally:
            if swap:
                head_layer.activation = saved_activation

    if gradients is None:  # pragma: no cover - defensive
        raise ExplanationUnavailable(
            "The tape returned no gradient for the feature map."
        )

    # Grad-CAM proper: channel importance is the spatially averaged gradient,
    # and the map is the ReLU of the importance-weighted feature map.
    weights = tf.reduce_mean(gradients, axis=(1, 2))
    cams = tf.nn.relu(tf.einsum("nhwc,nc->nhw", features, weights))

    return (
        np.asarray(probabilities).ravel(),
        np.asarray(embeddings),
        np.asarray(cams),
    )


def _apply_head(stack: _Stack, features):
    """Run the classifier head over a feature map, capturing the embedding."""
    activation = features
    embedding = None
    for position, layer in enumerate(stack.head):
        activation = layer(activation, training=False)
        # head[0] pools to 1280-d, head[1] is the trained 128-d dense layer.
        if position == min(1, len(stack.head) - 1):
            embedding = activation
    return embedding, activation


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def render_saliency(original_bytes: bytes, cam, *, render_size: int) -> tuple[str, str]:
    """Composite a Grad-CAM map over the original photo.

    Returns ``(original_data_url, overlay_data_url)``, both JPEG. The overlay is
    drawn on the image at its own aspect ratio — the model saw a squashed 224×224
    square, but the 7×7 grid maps back linearly either way, and showing the
    farmer their own uncropped photo is worth more than matching the model's
    letterboxing.
    """
    import numpy as np
    from PIL import Image

    with Image.open(io.BytesIO(original_bytes)) as handle:
        base = handle.convert("RGB")
        base.thumbnail((render_size, render_size), Image.LANCZOS)
        base = base.copy()

    original_url = _to_jpeg_data_url(base, quality=82)

    cam = np.asarray(cam, dtype="float32")
    span = float(cam.max() - cam.min())
    if span < 1e-8:
        # A flat map means the gradient told us nothing; showing a uniform wash
        # would imply structure that is not there.
        return original_url, original_url

    normalised = (cam - cam.min()) / span
    # Upsample through PIL so the 7×7 grid is smoothed rather than blocky.
    # Mode is inferred from the array's dtype and rank; passing it explicitly is
    # deprecated in Pillow 13.
    heat = Image.fromarray((normalised * 255).astype("uint8")).resize(
        base.size, Image.BICUBIC
    )
    intensity = np.asarray(heat, dtype="float32") / 255.0

    positions = [stop[0] for stop in _HEAT_STOPS]
    channels = [
        np.interp(intensity, positions, [stop[i] for stop in _HEAT_STOPS])
        for i in (1, 2, 3)
    ]
    rgb = np.stack(channels, axis=-1).astype("uint8")

    strength = np.clip((intensity - _HEAT_FLOOR) / (1.0 - _HEAT_FLOOR), 0.0, 1.0)
    alpha = (strength**0.7 * _PEAK_ALPHA * 255).astype("uint8")

    layer = Image.fromarray(np.concatenate([rgb, alpha[..., None]], axis=-1))
    composited = Image.alpha_composite(base.convert("RGBA"), layer).convert("RGB")
    return original_url, _to_jpeg_data_url(composited, quality=82)


def make_thumbnail(image_bytes: bytes, size: int) -> bytes:
    """Downscale to a square-bounded JPEG for storage in the precedent index."""
    from PIL import Image

    with Image.open(io.BytesIO(image_bytes)) as handle:
        frame = handle.convert("RGB")
        frame.thumbnail((size, size), Image.LANCZOS)
        buffer = io.BytesIO()
        frame.save(buffer, format="JPEG", quality=78, optimize=True)
        return buffer.getvalue()


def jpeg_bytes_to_data_url(raw: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")


def _to_jpeg_data_url(image, quality: int) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return jpeg_bytes_to_data_url(buffer.getvalue())


def preprocess_paths(paths: Sequence, size: int):
    """Load images from disk exactly as the CNN agent preprocesses uploads."""
    import numpy as np
    from PIL import Image

    batch = []
    for path in paths:
        with Image.open(path) as handle:
            frame = handle.convert("RGB").resize((size, size), Image.BILINEAR)
            batch.append(np.asarray(frame, dtype="float32"))
    return np.stack(batch, axis=0)


def normalise_rows(matrix, epsilon: float = 1e-9):
    """L2-normalise each row so a dot product is a cosine similarity."""
    import numpy as np

    matrix = np.asarray(matrix, dtype="float32")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / (norms + epsilon)


__all__ = [
    "ExplanationUnavailable",
    "embed_and_explain",
    "jpeg_bytes_to_data_url",
    "make_thumbnail",
    "normalise_rows",
    "preprocess_paths",
    "render_saliency",
    "split_stack",
]
