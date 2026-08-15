"""Build the precedent index used by the visual-evidence panel.

Embeds every image in `cow dataset/` through the trained classifier and writes
the vectors, labels and thumbnails to a single `.npz`. Run once after setup:

    make index                       # from the project root
    python scripts/build_visual_index.py --help

Takes roughly 20 s for ~1000 images on a CPU-only laptop. Re-run it if the
trained model or the dataset changes — the index stores embeddings from a
specific set of weights, so a retrained model needs a rebuilt index.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Quieten TensorFlow's startup banner before it is imported anywhere.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.reference_index import (  # noqa: E402
    IndexUnavailable,
    build_index,
    discover_images,
    load_index,
)


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset",
        type=Path,
        default=settings.dataset_dir,
        help=f"Directory of class folders (default: {settings.dataset_dir})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.visual_index_path,
        help=f"Where to write the index (default: {settings.visual_index_path})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Images per forward pass"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if an index already exists at the output path",
    )
    args = parser.parse_args()

    if args.output.is_file() and not args.force:
        try:
            existing = load_index(args.output)
        except IndexUnavailable as exc:
            print(f"Existing index is unusable ({exc}); rebuilding.")
        else:
            print(
                f"Index already present: {args.output} "
                f"({len(existing)} images). Use --force to rebuild."
            )
            return 0

    try:
        pairs = discover_images(args.dataset)
    except IndexUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    classes = sorted({label for _, label in pairs})
    print(f"Embedding {len(pairs)} images from {len(classes)} classes ({', '.join(classes)})")

    started = time.perf_counter()

    def progress(done: int, total: int) -> None:
        width = 32
        filled = int(width * done / total)
        bar = "#" * filled + "." * (width - filled)
        print(f"\r  [{bar}] {done}/{total}", end="", flush=True)

    try:
        path = build_index(
            dataset_dir=args.dataset,
            output_path=args.output,
            batch_size=args.batch_size,
            on_progress=progress,
        )
    except IndexUnavailable as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1

    print()
    size_mb = path.stat().st_size / (1024 * 1024)
    index = load_index(path)
    counts: dict[str, int] = {}
    for label in index.labels:
        counts[label] = counts.get(label, 0) + 1

    print(f"Wrote {path} ({size_mb:.1f} MB) in {time.perf_counter() - started:.1f} s")
    print(f"  embedding layer : {index.embedding_layer}")
    print(f"  reference images: {len(index)}")
    for label in sorted(counts):
        print(f"    {label:<16} {counts[label]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
