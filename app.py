import os
from pathlib import Path
from threading import Lock

import numpy as np
from PIL import Image, UnidentifiedImageError
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge

from class_mapping import CLASS_NAMES

BASE_DIR = Path(__file__).resolve().parent
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_IMAGE_SIZE = 10 * 1024 * 1024
MODEL_PATH = Path(
    os.environ.get("MODEL_PATH", BASE_DIR / "models" / "retinopathy.keras")
)

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_SIZE

_model = None
_model_lock = Lock()


@app.get("/")
def home():
    return render_template("index.html")


@app.errorhandler(RequestEntityTooLarge)
def handle_large_upload(_error):
    return jsonify({"success": False, "error": "Image must be 10 MB or smaller."}), 413


@app.post("/api/predict")
def predict():
    global _model

    image = request.files.get("image")
    if image is None or not image.filename:
        return jsonify({"success": False, "error": "Choose an image to continue."}), 400

    if Path(image.filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        return jsonify(
            {"success": False, "error": "Upload a JPG, JPEG, or PNG image."}
        ), 415

    if not MODEL_PATH.is_file():
        return jsonify(
            {
                "success": False,
                "error": (
                    "The local model is not trained yet. Run train_model.py first "
                    "and try again."
                ),
            }
        ), 503

    try:
        import tensorflow as tf
    except ModuleNotFoundError:
        app.logger.exception("TensorFlow is required for local model inference.")
        return jsonify(
            {
                "success": False,
                "error": "TensorFlow is not installed. Install requirements.txt and restart the app.",
            }
        ), 503

    try:
        with Image.open(image.stream) as uploaded_image:
            uploaded_image.load()
            rgb_image = uploaded_image.convert("RGB")
    except (UnidentifiedImageError, OSError):
        return jsonify(
            {"success": False, "error": "The uploaded file is not a readable image."}
        ), 400

    with _model_lock:
        if _model is None:
            try:
                _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            except (OSError, ValueError):
                app.logger.exception("Could not load the local classification model.")
                return jsonify(
                    {
                        "success": False,
                        "error": "The local model could not be loaded. Check MODEL_PATH and the model file.",
                    }
                ), 503
        model = _model

    input_shape = model.input_shape
    output_shape = model.output_shape
    if (
        len(input_shape) != 4
        or input_shape[-1] != 3
        or len(output_shape) != 2
        or output_shape[-1] != len(CLASS_NAMES)
        or input_shape[1] is None
        or input_shape[2] is None
    ):
        app.logger.error("The local model has incompatible input or output dimensions.")
        return jsonify(
            {
                "success": False,
                "error": "The local model must accept RGB images and return five class scores.",
            }
        ), 503

    pixels = tf.image.resize(
        np.asarray(rgb_image), (input_shape[1], input_shape[2])
    )
    scores = np.asarray(
        model.predict(tf.expand_dims(pixels, axis=0), verbose=0)[0]
    )
    if (
        scores.shape != (len(CLASS_NAMES),)
        or not np.isfinite(scores).all()
        or np.any(scores < 0)
        or np.any(scores > 1)
        or not np.isclose(scores.sum(), 1, atol=0.02)
    ):
        app.logger.error("The local model returned invalid five-class softmax scores.")
        return jsonify(
            {"success": False, "error": "The model returned invalid class scores."}
        ), 500
    class_index = int(tf.argmax(scores).numpy())

    return jsonify(
        {
            "success": True,
            "prediction": CLASS_NAMES[class_index],
            "confidence": float(scores[class_index]),
            "scores": {
                name: float(score) for name, score in zip(CLASS_NAMES, scores)
            },
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")))
