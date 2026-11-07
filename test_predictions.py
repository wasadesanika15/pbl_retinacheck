import argparse
import csv
import sys
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf

from class_mapping import CLASS_NAMES
from train_model import make_dataset, read_dataset, split_dataset

BASE_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the trained model on the training script's held-out split."
    )
    parser.add_argument("dataset_zip", type=Path, help="Labeled dataset ZIP used for training.")
    parser.add_argument(
        "--model",
        type=Path,
        default=BASE_DIR / "models" / "retinopathy.keras",
        help="Trained Keras model (default: models/retinopathy.keras).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    archive_path = args.dataset_zip.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    if not archive_path.is_file():
        parser.error(f"Dataset ZIP not found: {archive_path}")
    if not model_path.is_file():
        parser.error(f"Model not found: {model_path}")
    if args.batch_size < 1:
        parser.error("--batch-size must be a positive integer.")

    model = tf.keras.models.load_model(model_path, compile=False)
    input_shape = model.input_shape
    if (
        len(input_shape) != 4
        or input_shape[-1] != 3
        or input_shape[1] is None
        or input_shape[2] is None
        or model.output_shape != (None, len(CLASS_NAMES))
    ):
        parser.error("Model must accept fixed-size RGB images and return five class scores.")

    with tempfile.TemporaryDirectory(prefix="retinacheck-evaluation-") as temp_dir:
        samples = read_dataset(archive_path, Path(temp_dir) / "images")
        _, _, test = split_dataset(samples, args.seed)
        test_dataset = make_dataset(
            test,
            args.batch_size,
            training=False,
            image_size=input_shape[1],
        )
        scores = model.predict(test_dataset, verbose=0)

    if scores.shape != (len(test), len(CLASS_NAMES)):
        raise ValueError(f"Expected (N, 5) model scores, got {scores.shape}.")
    if (
        not np.isfinite(scores).all()
        or np.any(scores < 0)
        or np.any(scores > 1)
        or not np.allclose(scores.sum(axis=1), 1, atol=0.02)
    ):
        raise ValueError("Model outputs are not valid five-class softmax scores.")

    actual = np.asarray([label for _, label in test])
    predicted = scores.argmax(axis=1)
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    np.add.at(confusion, (actual, predicted), 1)
    precision_scores = []
    recall_scores = []
    f1_scores = []

    writer = csv.writer(sys.stdout)
    writer.writerow(("filename", "actual_class", "predicted_class", "model_score", "correct"))
    for (image_path, actual_index), predicted_index, row_scores in zip(
        test, predicted, scores
    ):
        writer.writerow(
            (
                Path(image_path).name,
                CLASS_NAMES[actual_index],
                CLASS_NAMES[predicted_index],
                f"{row_scores[predicted_index] * 100:.1f}%",
                str(actual_index == predicted_index).lower(),
            )
        )

    print(f"\nHeld-out test accuracy: {(predicted == actual).mean():.3f}")
    print("Per-class metrics (class, precision, recall, F1, support):")
    for index, name in enumerate(CLASS_NAMES):
        true_positive = confusion[index, index]
        precision = true_positive / max(int(confusion[:, index].sum()), 1)
        recall = true_positive / max(int(confusion[index, :].sum()), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        print(
            f"{name}, {precision:.3f}, {recall:.3f}, {f1:.3f}, "
            f"{int(confusion[index, :].sum())}"
        )
    print(
        "Macro precision={:.3f}, recall={:.3f}, F1={:.3f}".format(
            np.mean(precision_scores),
            np.mean(recall_scores),
            np.mean(f1_scores),
        )
    )
    print("Confusion matrix (rows=actual, columns=predicted):")
    print("actual\\pred," + ",".join(CLASS_NAMES))
    for name, row in zip(CLASS_NAMES, confusion):
        print(name + "," + ",".join(str(int(value)) for value in row))
    print(f"Severe predicted as No DR: {int(confusion[3, 0])}")
    print(f"PDR predicted as No DR: {int(confusion[4, 0])}")
    print(f"Moderate predicted as No DR: {int(confusion[2, 0])}")
    print(
        "\nUse the same --seed as training to reproduce its held-out split. "
        "A different seed may include training images."
    )


if __name__ == "__main__":
    main()
