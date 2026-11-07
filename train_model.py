import argparse
import csv
import io
import re
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import tensorflow as tf

from class_mapping import (
    CLASS_NAMES,
    DATASET_LABEL_TO_CLASS_INDEX,
    PREPROCESSING,
)

BASE_DIR = Path(__file__).resolve().parent
IMAGE_SIZE = 224
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def read_dataset(archive_path, image_dir):
    samples = []

    with zipfile.ZipFile(archive_path) as archive:
        csv_files = [
            name for name in archive.namelist() if Path(name).name == "train.csv"
        ]
        if len(csv_files) != 1:
            raise ValueError("Dataset ZIP must contain exactly one train.csv file.")

        with archive.open(csv_files[0]) as csv_file:
            rows = csv.DictReader(io.TextIOWrapper(csv_file, encoding="utf-8-sig"))
            for row in rows:
                image_id = row.get("id_code", "").strip()
                try:
                    label = int(row["diagnosis"])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"Invalid diagnosis row: {row}") from error

                if not image_id or re.fullmatch(r"[A-Za-z0-9_-]+", image_id) is None:
                    raise ValueError(f"Invalid image id in dataset CSV: {image_id!r}")
                if label not in DATASET_LABEL_TO_CLASS_INDEX:
                    raise ValueError(f"Diagnosis for {image_id} must be between 0 and 4.")
                samples.append(
                    (image_id, DATASET_LABEL_TO_CLASS_INDEX[label])
                )

        if not samples:
            raise ValueError("Dataset CSV contains no labeled images.")
        if len({image_id for image_id, _ in samples}) != len(samples):
            raise ValueError("Dataset CSV contains duplicate image ids.")

        members_by_stem = {
            Path(name).stem: name
            for name in archive.namelist()
            if name.lower().endswith(".png")
        }
        image_dir.mkdir(parents=True, exist_ok=True)
        labeled_paths = []
        for image_id, label in samples:
            member_name = members_by_stem.get(image_id)
            if member_name is None:
                raise ValueError(f"Image {image_id}.png from train.csv is missing.")
            image_path = image_dir / f"{image_id}.png"
            with archive.open(member_name) as source, image_path.open("wb") as target:
                target.write(source.read())
            labeled_paths.append((str(image_path), label))

    return labeled_paths


def split_dataset(
    samples,
    seed,
    validation_fraction=VALIDATION_FRACTION,
    test_fraction=TEST_FRACTION,
):
    if validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError("Validation and test fractions must both be positive.")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must sum to less than 1.")

    rng = np.random.default_rng(seed)
    training = []
    validation = []
    test = []

    for label in range(len(CLASS_NAMES)):
        class_samples = [sample for sample in samples if sample[1] == label]
        if len(class_samples) < 3:
            raise ValueError(
                f"At least three images per class are needed; class {label} has "
                f"{len(class_samples)}."
            )
        order = rng.permutation(len(class_samples))
        test_count = max(1, round(len(class_samples) * test_fraction))
        validation_count = max(1, round(len(class_samples) * validation_fraction))
        if test_count + validation_count >= len(class_samples):
            raise ValueError(
                f"Not enough images in class {label} to create train, validation, "
                "and test splits."
            )
        test.extend(class_samples[index] for index in order[:test_count])
        validation_start = test_count
        training_start = validation_start + validation_count
        validation.extend(
            class_samples[index]
            for index in order[validation_start:training_start]
        )
        training.extend(class_samples[index] for index in order[training_start:])

    rng.shuffle(training)
    rng.shuffle(validation)
    rng.shuffle(test)
    return training, validation, test


def make_dataset(samples, batch_size, training, image_size=IMAGE_SIZE):
    paths = [path for path, _ in samples]
    labels = [label for _, label in samples]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load_image(path, label):
        encoded = tf.io.read_file(path)
        image = tf.io.decode_png(encoded, channels=3)
        image = tf.image.resize(image, (image_size, image_size))
        return tf.cast(image, tf.float32), label

    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        dataset = dataset.shuffle(len(samples), reshuffle_each_iteration=True)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model(initial_model=None, image_size=IMAGE_SIZE):
    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=None if initial_model is not None else "imagenet",
        input_shape=(image_size, image_size, 3),
    )
    if initial_model is not None:
        backbone.set_weights(initial_model.get_layer("efficientnetb0").get_weights())

    backbone.trainable = False

    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    augmented = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.04),
            tf.keras.layers.RandomZoom(0.08),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="image_augmentation",
    )(inputs)
    features = backbone(augmented, training=False)
    pooled = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling2d")(
        features
    )
    dropout = tf.keras.layers.Dropout(0.25, name="dropout")(pooled)
    outputs = tf.keras.layers.Dense(
        len(CLASS_NAMES), activation="softmax", name="dense"
    )(dropout)
    model = tf.keras.Model(inputs, outputs)
    if initial_model is not None:
        model.get_layer("dense").set_weights(initial_model.get_layer("dense").get_weights())
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def evaluate_model(model, dataset, samples, split_name):
    labels = np.asarray([label for _, label in samples])
    probabilities = model.predict(dataset, verbose=0)
    if probabilities.shape != (len(samples), len(CLASS_NAMES)):
        raise ValueError(
            f"Expected {len(CLASS_NAMES)} output scores per image, got "
            f"{probabilities.shape}."
        )
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or not np.allclose(probabilities.sum(axis=1), 1, atol=0.02)
    ):
        raise ValueError("Model outputs are not valid five-class softmax scores.")
    predictions = probabilities.argmax(axis=1)
    confusion = tf.math.confusion_matrix(
        labels, predictions, num_classes=len(CLASS_NAMES)
    ).numpy()
    f1_scores = []
    precision_scores = []
    recall_scores = []

    print(f"{split_name} accuracy: {(predictions == labels).mean():.3f}")
    for index, name in enumerate(CLASS_NAMES):
        true_positive = confusion[index, index]
        precision = true_positive / max(int(confusion[:, index].sum()), 1)
        recall = true_positive / max(int(confusion[index, :].sum()), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        print(
            f"{name}: precision={precision:.3f}, recall={recall:.3f}, "
            f"F1={f1:.3f} (n={int(confusion[index, :].sum())})"
        )
    print(
        f"{split_name} macro precision={np.mean(precision_scores):.3f}, "
        f"recall={np.mean(recall_scores):.3f}, F1={np.mean(f1_scores):.3f}"
    )
    print("Confusion matrix (rows=actual, columns=predicted):")
    print("actual\\pred," + ",".join(CLASS_NAMES))
    for name, row in zip(CLASS_NAMES, confusion):
        print(name + "," + ",".join(str(int(value)) for value in row))
    print(f"Severe predicted as No DR: {int(confusion[3, 0])}")
    print(f"PDR predicted as No DR: {int(confusion[4, 0])}")
    print(f"Moderate predicted as No DR: {int(confusion[2, 0])}")

    true_referable = labels >= 2
    predicted_referable = predictions >= 2
    true_positive = int(np.sum(true_referable & predicted_referable))
    false_negative = int(np.sum(true_referable & ~predicted_referable))
    true_negative = int(np.sum(~true_referable & ~predicted_referable))
    false_positive = int(np.sum(~true_referable & predicted_referable))
    sensitivity = true_positive / max(true_positive + false_negative, 1)
    specificity = true_negative / max(true_negative + false_positive, 1)
    print(
        f"{split_name} exploratory referable-DR metrics (grades 2-4 vs 0-1; "
        "not a clinical threshold): "
        f"sensitivity={sensitivity:.3f} ({true_positive}/"
        f"{true_positive + false_negative}), specificity={specificity:.3f} "
        f"({true_negative}/{true_negative + false_positive})"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train a diabetic-retinopathy classifier from the labeled image ZIP."
    )
    parser.add_argument("dataset_zip", type=Path, help="Path to the dataset ZIP file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "models" / "retinopathy.keras",
        help="Where to save the trained model (default: models/retinopathy.keras).",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Continue from an existing classifier (not suitable for an untouched test set).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        choices=(224, 384),
        default=IMAGE_SIZE,
        help="Square model input resolution (default: 224).",
    )
    parser.add_argument("--epochs", type=positive_int, default=8)
    parser.add_argument("--warmup-epochs", type=positive_int, default=3)
    parser.add_argument("--fine-tune-layers", type=positive_int, default=50)
    parser.add_argument("--batch-size", type=positive_int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    archive_path = args.dataset_zip.expanduser().resolve()
    if not archive_path.is_file():
        parser.error(f"Dataset ZIP not found: {archive_path}")
    resume_path = args.resume.expanduser().resolve() if args.resume else None
    if resume_path is not None and not resume_path.is_file():
        parser.error(f"Resume model not found: {resume_path}")
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="retinacheck-training-") as temp_dir, tempfile.TemporaryDirectory(
        prefix=f".{output_path.stem}-candidate-", dir=output_path.parent
    ) as candidate_dir:
        candidate_path = Path(candidate_dir) / output_path.name
        samples = read_dataset(archive_path, Path(temp_dir) / "images")
        training, validation, test = split_dataset(samples, args.seed)
        counts = np.bincount([label for _, label in training], minlength=len(CLASS_NAMES))
        class_weights = {
            label: (len(training) / (len(CLASS_NAMES) * count)) ** 0.5
            for label, count in enumerate(counts)
        }
        tf.keras.utils.set_random_seed(args.seed)
        print(
            f"Loaded {len(samples)} images at {args.image_size}px; training on "
            f"{len(training)}, validating on {len(validation)}, and reserving "
            f"{len(test)} for final testing."
        )
        for label, name in enumerate(CLASS_NAMES):
            print(
                f"{name}: total={sum(sample[1] == label for sample in samples)}, "
                f"train={int(counts[label])}, "
                f"validation={sum(sample[1] == label for sample in validation)}, "
                f"test={sum(sample[1] == label for sample in test)}, "
                f"class_weight={class_weights[label]:.3f}"
            )

        training_dataset = make_dataset(
            training, args.batch_size, training=True, image_size=args.image_size
        )
        validation_dataset = make_dataset(
            validation, args.batch_size, training=False, image_size=args.image_size
        )
        test_dataset = make_dataset(
            test, args.batch_size, training=False, image_size=args.image_size
        )

        initial_model = (
            tf.keras.models.load_model(resume_path, compile=False)
            if resume_path is not None
            else None
        )
        model = build_model(initial_model, image_size=args.image_size)
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                candidate_path, monitor="val_loss", save_best_only=True
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=2, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.3, patience=1, min_lr=1e-7
            ),
        ]

        if initial_model is None:
            model.fit(
                training_dataset,
                validation_data=validation_dataset,
                epochs=args.warmup_epochs,
                class_weight=class_weights,
                callbacks=callbacks,
            )

        backbone = model.get_layer("efficientnetb0")
        backbone.trainable = True
        fine_tune_from = max(0, len(backbone.layers) - args.fine_tune_layers)
        for layer in backbone.layers[:fine_tune_from]:
            layer.trainable = False
        for layer in backbone.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            training_dataset,
            validation_data=validation_dataset,
            epochs=args.epochs,
            class_weight=class_weights,
            callbacks=callbacks,
        )

        best_model = tf.keras.models.load_model(candidate_path, compile=False)
        evaluate_model(best_model, test_dataset, test, "Test")
        candidate_path.replace(output_path)

    print(f"Saved best fine-tuned model to {output_path}")
    print(f"Output labels: {', '.join(CLASS_NAMES)}")
    print(f"Inference preprocessing: {PREPROCESSING}")


if __name__ == "__main__":
    main()
