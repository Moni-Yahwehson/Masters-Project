"""Held-out evaluation of the trained LSD classifier.

Why this script exists
----------------------
`Processing.ipynb` builds its validation set with ``shuffle=False``:

    val_generator = utils.image_dataset_from_directory(
        DATA_PATH, validation_split=0.2, subset="validation",
        seed=42, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode='binary', shuffle=False)          # <-- the bug

With ``shuffle=False`` the split is taken from the end of the *directory-ordered*
file list. Because the classes sort as ``healthycows`` then ``lumpycows``, the
last 20% of that list is entirely ``lumpycows`` — which is exactly what the
notebook's confusion matrix shows:

    [[  0   0]
     [ 11 197]]      0 healthy samples

Every metric printed from that split is therefore uninterpretable: the reported
"95% accuracy" is just recall on a single class, and precision/recall/F1 for
`Healthy Cows` are undefined (hence the `UndefinedMetricWarning`s). The training
set was built with ``shuffle=True``, so this validation set also *overlaps the
training data*.

This script rebuilds the split the way training did (``shuffle=True``, same
seed), which yields the genuinely held-out 20%, and reports metrics on it via
the same preprocessing path the API uses.

Usage
-----
    .venv/bin/python scripts/evaluate_model.py
    .venv/bin/python scripts/evaluate_model.py --data "../vision/cow dataset"
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    default_data = settings.model_path.parent / "cow dataset"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=default_data)
    parser.add_argument("--model", type=Path, default=settings.model_path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.data.is_dir():
        print(f"Dataset directory not found: {args.data}", file=sys.stderr)
        return 1
    if not args.model.is_file():
        print(f"Model not found: {args.model}", file=sys.stderr)
        return 1

    import keras
    import numpy as np
    from keras import utils
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        roc_auc_score,
    )

    from app.agents.cnn_vision import _relaxed_layer_configs

    settings = get_settings()
    size = settings.model_image_size

    print("=" * 72)
    print("HELD-OUT EVALUATION — cattle_lsd_detector.keras")
    print("=" * 72)

    # shuffle=True mirrors how the *training* subset was drawn, so this really
    # is the complement of the training data.
    validation = utils.image_dataset_from_directory(
        str(args.data),
        validation_split=args.validation_split,
        subset="validation",
        seed=args.seed,
        image_size=(size, size),
        batch_size=args.batch_size,
        label_mode="binary",
        shuffle=True,
    )
    class_names = list(validation.class_names)
    print(f"\nClass order: {dict(enumerate(class_names))}")

    with _relaxed_layer_configs(keras):
        model = keras.models.load_model(str(args.model), compile=False)

    probabilities, truths = [], []
    for batch_images, batch_labels in validation:
        probabilities.append(model.predict(batch_images, verbose=0).flatten())
        truths.append(np.asarray(batch_labels).flatten())

    y_score = np.concatenate(probabilities)
    y_true = np.concatenate(truths).astype(int)

    counts = np.bincount(y_true, minlength=2)
    print(f"Held-out samples: {len(y_true)}")
    print(f"  {class_names[0]}: {counts[0]}")
    print(f"  {class_names[1]}: {counts[1]}")
    if counts.min() == 0:
        print(
            "\nWARNING: the split still contains a single class — metrics below "
            "are not interpretable.",
            file=sys.stderr,
        )

    threshold = settings.decision_threshold
    y_pred = (y_score >= threshold).astype(int)

    print(f"\n--- CONFUSION MATRIX (threshold {threshold}) ---")
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    label_width = max(len(name) for name in class_names) + 6
    cell_width = max(len(name) for name in class_names) + 7
    print(" " * label_width + "".join(f"{'pred ' + name:>{cell_width}}" for name in class_names))
    for index, row in enumerate(matrix):
        label = f"true {class_names[index]}"
        print(f"{label:<{label_width}}" + "".join(f"{value:>{cell_width}}" for value in row))

    print("\n--- CLASSIFICATION REPORT ---")
    print(
        classification_report(
            y_true, y_pred, target_names=class_names, digits=4, zero_division=0
        )
    )
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    if counts.min() > 0:
        print(f"ROC-AUC : {roc_auc_score(y_true, y_score):.4f}")

    print("\n--- THRESHOLD SWEEP ---")
    print(f"{'thresh':>7} {'acc':>8} {'precision':>10} {'recall':>8} {'f1':>8}")
    for candidate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        predictions = (y_score >= candidate).astype(int)
        true_pos = int(((predictions == 1) & (y_true == 1)).sum())
        false_pos = int(((predictions == 1) & (y_true == 0)).sum())
        false_neg = int(((predictions == 0) & (y_true == 1)).sum())
        precision = true_pos / (true_pos + false_pos) if true_pos + false_pos else 0.0
        recall = true_pos / (true_pos + false_neg) if true_pos + false_neg else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        accuracy = float((predictions == y_true).mean())
        print(
            f"{candidate:>7.1f} {accuracy:>8.4f} {precision:>10.4f} "
            f"{recall:>8.4f} {f1:>8.4f}"
        )

    band = settings.uncertainty_band
    uncertain = int(((y_score > threshold - band) & (y_score < threshold + band)).sum())
    print(
        f"\nPredictions inside the ±{band} uncertainty band: "
        f"{uncertain} / {len(y_score)} ({uncertain / len(y_score):.1%}) — "
        "these are reported as 'inconclusive' by the API."
    )

    print("\n--- CONFIDENCE DISTRIBUTION ---")
    edges = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0001]
    for low, high in zip(edges, edges[1:]):
        mask = (y_score >= low) & (y_score < high)
        total = int(mask.sum())
        if not total:
            continue
        actual_positive = float(y_true[mask].mean())
        print(
            f"  p in [{low:.2f}, {min(high, 1.0):.2f}): n={total:<4} "
            f"actual {class_names[1]} rate = {actual_positive:.2%}"
        )
    print(
        "\nA well-calibrated model's actual positive rate should track the "
        "probability band. Large gaps mean the sigmoid output should not be "
        "read as a probability of disease without calibration."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
