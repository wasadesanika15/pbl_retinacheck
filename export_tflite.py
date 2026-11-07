import argparse
import json
from pathlib import Path

import tensorflow as tf

from class_mapping import CLASS_NAMES, PREPROCESSING

BASE_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(
        description="Export the trained Keras model for browser inference."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=BASE_DIR / "models" / "retinopathy.keras",
        help="Trained Keras model (default: models/retinopathy.keras).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "static" / "models" / "retinopathy.tflite",
        help="Output TFLite file (default: static/models/retinopathy.tflite).",
    )
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        parser.error(f"Trained model not found: {input_path}")

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = tf.keras.models.load_model(input_path)
    if (
        len(model.input_shape) != 4
        or model.input_shape[-1] != 3
        or len(model.output_shape) != 2
        or model.output_shape[-1] != len(CLASS_NAMES)
    ):
        parser.error("Model must accept RGB batches and return exactly five class scores.")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    output_path.write_bytes(converter.convert())
    input_shape = model.input_shape
    metadata = {
        "inputSize": int(input_shape[1]),
        "classNames": CLASS_NAMES,
        "preprocessing": PREPROCESSING,
        "outputActivation": "softmax",
    }
    metadata_path = output_path.with_name("metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Exported browser model to {output_path}")
    print(f"Exported model metadata to {metadata_path}")


if __name__ == "__main__":
    main()
