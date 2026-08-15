"""The precedent index — labelled training images in the CNN's feature space.

The classifier is a closed-world binary model: it must answer "healthy" or
"lumpy" for anything it is shown, including ringworm, mange or a photograph of a
fence. A bare probability hides that. Showing the nearest labelled images the
score is actually pattern-matching against does not.

The index is a single `.npz` holding, for every image in `cow dataset/`:

* its 128-d embedding from the classifier's trained penultimate layer;
* its ground-truth folder label;
* its filename;
* a small JPEG thumbnail, so the API can return precedents without serving files
  off disk — no static route, no path traversal, and the payload stays offline.

Thumbnails are stored as one concatenated blob plus offsets rather than an
object array, which keeps `np.load(allow_pickle=False)` safe to use.

**On the numbers this produces.** Leave-one-out kNN over the index scores 0.955
(k=5) in this embedding space against 0.846 for the raw pooled backbone
features, which is why the trained dense layer is used. That figure is measured
on the images the network was trained on, so it is a statement about retrieval
behaviour, not a held-out accuracy claim — `scripts/evaluate_model.py` remains
the only source for that.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from .config import get_settings

logger = logging.getLogger(__name__)

#: Bumped whenever the stored layout changes so a stale file is rejected rather
#: than silently misread.
#: v2 added `probabilities`, used to pick reference images the classifier itself
#: scores confidently.
INDEX_FORMAT_VERSION = 2

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class IndexUnavailable(RuntimeError):
    """Raised when the precedent index cannot be loaded."""


@dataclass
class Neighbour:
    reference_id: str
    label: str
    similarity: float
    thumbnail_jpeg: bytes


@dataclass
class ReferenceIndex:
    embeddings: "object"          # (n, d) float32, L2-normalised
    labels: list[str]
    names: list[str]
    thumbnails: list[bytes]
    embedding_layer: str
    probabilities: "object" = None  # (n,) float32 P(lumpycows), or None on a v1 index

    def __len__(self) -> int:
        return len(self.names)

    def representatives(
        self,
        label: str,
        count: int,
        *,
        positive_class: str = "lumpycows",
        min_confidence: float = 0.9,
        max_similarity: float = 0.88,
    ) -> list[Neighbour]:
        """Pick images that best *typify* a class, for use as reference photos.

        Not the highest-scoring images, and not a random sample. Both are bad
        illustrations: top-scored images are the model's extremes, and a random
        draw regularly turns up a blurry animal at the back of a field.

        Instead, take the images the classifier is confident about, then rank by
        closeness to that class's centroid in embedding space — the most typical
        member of the cluster first. A greedy diversity filter then drops any
        candidate too similar to one already chosen, because consecutive frames
        of the same animal are common in this dataset and six near-duplicates
        teach the reader nothing.
        """
        import numpy as np

        if len(self) == 0 or count <= 0:
            return []

        labels = np.asarray(self.labels)
        eligible = np.flatnonzero(labels == label)
        if eligible.size == 0:
            return []

        # Keep only images the model itself calls confidently, when we know.
        if self.probabilities is not None:
            scores = np.asarray(self.probabilities, dtype="float32")[eligible]
            confident = (
                scores >= min_confidence
                if label == positive_class
                else scores <= (1.0 - min_confidence)
            )
            if confident.any():
                eligible = eligible[confident]

        vectors = np.asarray(self.embeddings, dtype="float32")[eligible]
        centroid = vectors.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm == 0.0:
            return []
        typicality = vectors @ (centroid / norm)
        ranking = np.argsort(-typicality)          # positions within `eligible`

        all_vectors = np.asarray(self.embeddings, dtype="float32")
        chosen: list[int] = []                     # positions within `eligible`
        for position in ranking:
            if len(chosen) >= count:
                break
            candidate = all_vectors[eligible[position]]
            if any(
                float(candidate @ all_vectors[eligible[picked]]) > max_similarity
                for picked in chosen
            ):
                continue
            chosen.append(int(position))

        # If diversity filtering starved the gallery — a small or very uniform
        # class — top it up rather than returning two images where six were
        # asked for. Take the *least* similar candidates first, so the images
        # the filter just rejected are the last ones added back.
        if len(chosen) < count:
            remaining = [int(p) for p in ranking if int(p) not in chosen]
            remaining.sort(
                key=lambda position: max(
                    (
                        float(
                            all_vectors[eligible[position]]
                            @ all_vectors[eligible[picked]]
                        )
                        for picked in chosen
                    ),
                    default=0.0,
                )
            )
            chosen.extend(remaining[: count - len(chosen)])

        return [
            Neighbour(
                reference_id=self.names[eligible[position]],
                label=self.labels[eligible[position]],
                similarity=round(float(typicality[position]), 4),
                thumbnail_jpeg=self.thumbnails[eligible[position]],
            )
            for position in chosen
        ]

    def query(self, embedding, top_k: int) -> list[Neighbour]:
        """Return the `top_k` most cosine-similar reference images."""
        import numpy as np

        if len(self) == 0:
            return []
        vector = np.asarray(embedding, dtype="float32").ravel()
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return []
        similarities = self.embeddings @ (vector / norm)

        top_k = max(0, min(top_k, len(self)))
        if top_k == 0:
            return []
        # argpartition then sort the slice: O(n) instead of sorting 1000+ rows
        # for the six we display.
        candidates = np.argpartition(-similarities, top_k - 1)[:top_k]
        candidates = candidates[np.argsort(-similarities[candidates])]
        return [
            Neighbour(
                reference_id=self.names[i],
                label=self.labels[i],
                similarity=round(float(similarities[i]), 4),
                thumbnail_jpeg=self.thumbnails[i],
            )
            for i in candidates
        ]


# --------------------------------------------------------------------------
# Building
# --------------------------------------------------------------------------


def discover_images(dataset_dir: Path) -> list[tuple[Path, str]]:
    """List `(path, label)` pairs, label being the immediate folder name.

    Sorted so a rebuild on another machine produces the same index.
    """
    if not dataset_dir.is_dir():
        raise IndexUnavailable(f"Dataset directory not found: {dataset_dir}")
    pairs: list[tuple[Path, str]] = []
    for class_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                pairs.append((path, class_dir.name))
    if not pairs:
        raise IndexUnavailable(f"No images found under {dataset_dir}")
    return pairs


def build_index(
    *,
    dataset_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
    batch_size: int = 32,
    on_progress=None,
) -> Path:
    """Embed every dataset image and write the index to disk.

    Roughly 20 s for ~1000 images on a CPU-only laptop. Run once at setup, and
    again only if the model or the dataset changes.
    """
    import numpy as np

    from .agents.cnn_vision import get_cnn_agent
    from .agents.visual_explain import (
        embed_and_explain,
        make_thumbnail,
        normalise_rows,
        preprocess_paths,
        split_stack,
    )

    settings = get_settings()
    dataset_dir = Path(dataset_dir or settings.dataset_dir)
    output_path = Path(output_path or settings.visual_index_path)

    agent = get_cnn_agent()
    if not agent.available:
        raise IndexUnavailable(
            f"The CNN is unavailable, so images cannot be embedded: "
            f"{agent.unavailable_reason}"
        )
    model = agent._load_model()  # noqa: SLF001 - same package, one shared model
    layer_name = split_stack(model).embedding_layer

    pairs = discover_images(dataset_dir)
    size = settings.model_image_size
    started = time.perf_counter()

    vectors: list = []
    scores: list = []
    thumbnails: list[bytes] = []
    labels: list[str] = []
    names: list[str] = []

    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        batch = preprocess_paths([path for path, _ in chunk], size)
        probabilities, embeddings, _ = embed_and_explain(
            model, batch, threshold=settings.decision_threshold, want_saliency=False
        )
        vectors.append(np.asarray(embeddings, dtype="float32"))
        scores.append(np.asarray(probabilities, dtype="float32"))
        for path, label in chunk:
            thumbnails.append(
                make_thumbnail(path.read_bytes(), settings.thumbnail_size)
            )
            labels.append(label)
            names.append(path.name)
        if on_progress is not None:
            on_progress(min(start + batch_size, len(pairs)), len(pairs))

    matrix = normalise_rows(np.concatenate(vectors, axis=0))

    blob = b"".join(thumbnails)
    offsets = np.zeros(len(thumbnails) + 1, dtype="int64")
    for i, thumbnail in enumerate(thumbnails):
        offsets[i + 1] = offsets[i] + len(thumbnail)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        format_version=np.asarray([INDEX_FORMAT_VERSION], dtype="int32"),
        embeddings=matrix,
        probabilities=np.concatenate(scores, axis=0),
        labels=np.asarray(labels),
        names=np.asarray(names),
        thumbnail_blob=np.frombuffer(blob, dtype="uint8"),
        thumbnail_offsets=offsets,
        embedding_layer=np.asarray([layer_name]),
    )
    logger.info(
        "Built precedent index: %d images -> %s in %.1f s",
        len(pairs),
        output_path,
        time.perf_counter() - started,
    )
    return output_path


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

_index: Optional[ReferenceIndex] = None
_load_error: Optional[str] = None
_attempted = False
_lock = threading.Lock()


def load_index(path: Optional[Path] = None) -> ReferenceIndex:
    import numpy as np

    settings = get_settings()
    path = Path(path or settings.visual_index_path)
    if not path.is_file():
        raise IndexUnavailable(
            f"No precedent index at {path}. Build it with `make index`."
        )

    with np.load(path, allow_pickle=False) as data:
        version = int(data["format_version"][0])
        if version != INDEX_FORMAT_VERSION:
            raise IndexUnavailable(
                f"Precedent index at {path} is format v{version}, but this build "
                f"expects v{INDEX_FORMAT_VERSION}. Rebuild it with `make index`."
            )
        embeddings = np.asarray(data["embeddings"], dtype="float32")
        probabilities = (
            np.asarray(data["probabilities"], dtype="float32")
            if "probabilities" in data.files
            else None
        )
        labels = [str(value) for value in data["labels"]]
        names = [str(value) for value in data["names"]]
        blob = np.asarray(data["thumbnail_blob"], dtype="uint8").tobytes()
        offsets = np.asarray(data["thumbnail_offsets"], dtype="int64")
        layer = str(data["embedding_layer"][0])

    thumbnails = [
        blob[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)
    ]
    if not (len(labels) == len(names) == len(thumbnails) == embeddings.shape[0]):
        raise IndexUnavailable(
            f"Precedent index at {path} is inconsistent; rebuild it with `make index`."
        )
    return ReferenceIndex(
        embeddings=embeddings,
        labels=labels,
        names=names,
        thumbnails=thumbnails,
        embedding_layer=layer,
        probabilities=probabilities,
    )


def get_index() -> Optional[ReferenceIndex]:
    """Load the index once, caching both success and failure.

    Returns `None` when unavailable: precedents are an enhancement, and a
    missing index must never stop a diagnosis.
    """
    global _index, _load_error, _attempted
    if _index is not None or _attempted:
        return _index
    with _lock:
        if _index is not None or _attempted:
            return _index
        _attempted = True
        try:
            _index = load_index()
            logger.info("Loaded precedent index: %d reference images", len(_index))
        except Exception as exc:  # noqa: BLE001 - any failure means "no precedents"
            _load_error = str(exc)
            logger.info("Precedent index unavailable: %s", exc)
        return _index


def unavailable_reason() -> str:
    return _load_error or ""


def reset_cache() -> None:
    """Drop the cached index. Used by tests and after a rebuild."""
    global _index, _load_error, _attempted
    with _lock:
        _index = None
        _load_error = None
        _attempted = False


def summarise_labels(neighbours: Sequence[Neighbour]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for neighbour in neighbours:
        counts[neighbour.label] = counts.get(neighbour.label, 0) + 1
    return counts
