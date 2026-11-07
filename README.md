# RetinaCheck

RetinaCheck is an experimental five-class diabetic-retinopathy image classifier. The Flask-served website runs the exported TensorFlow Lite model in the browser; the optional Flask prediction endpoint loads the Keras model.

## Train the model

Use Python 3.9 or newer. From this directory, create an environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train from the labeled APTOS-style ZIP archive (it should contain one `train.csv` with `id_code` and `diagnosis` columns and the matching Gaussian-filtered PNG files):

```bash
python train_model.py "/path/to/archive (2).zip"
```

The script maps dataset grades 0–4 to **No DR, Mild, Moderate, Severe, PDR** and trains a five-output EfficientNetB0 softmax classifier with sparse categorical cross-entropy. Its stratified 70/15/15 split reserves test images from training; validation selects the checkpoint. It applies conservative augmentation and moderated class weights, then fine-tunes the higher-level layers. By default, the model is saved to `models/retinopathy.keras` only after held-out evaluation succeeds; use `--output` to select another destination. The default input is 224 pixels; use `--image-size 384` for a higher-resolution model. The first run downloads ImageNet weights and therefore needs an internet connection. Extracted dataset images are removed after training.

The training loader converts PNGs to RGB, resizes with TensorFlow bilinear resize, and passes float32 pixel values in the 0–255 range to EfficientNetB0. The browser and Flask prediction paths now do the same, without applying the Gaussian high-pass transform a second time. EfficientNetB0 handles its own input rescaling inside the model. A shared mapping in `class_mapping.py` is used by training, export, and Flask; exported metadata supplies the same ordered labels to the browser.

To fine-tune an existing model rather than start with a new classifier head:

```bash
python train_model.py "/path/to/archive (2).zip" --resume models/retinopathy.keras
```

After training, export the updated browser model with:

```bash
python export_tflite.py
```

To report per-image predictions, accuracy, per-class precision/recall/F1, and a confusion matrix on the same held-out split:

```bash
python test_predictions.py "/path/to/dataset.zip" --model models/retinopathy.keras --seed 42
```

Use the training seed; choosing another split may include images the model trained on. A dataset ZIP must be provided to retrain or run this evaluation. Results on the same dataset are not independent external validation.

## Run the app

Build the browser runtime and start the app:

```bash
npm ci
npm run build
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000). The website uses the browser TFLite model in `static/models/retinopathy.tflite`; it does not send images to the server. The separate `/api/predict` endpoint uses the Keras model. To choose another Keras model for that endpoint, set `MODEL_PATH` before starting Flask:

```bash
MODEL_PATH="/path/to/retinopathy.keras" python app.py
```

The model class order is **No DR, Mild, Moderate, Severe, PDR**. The prediction page displays the winning class score and all five model scores; these are not calibrated medical probabilities. Upload retinal fundus photographs only. The browser asks you to confirm the image type, but this is not automated image verification: unrelated images may still receive meaningless grades. A separate labeled retinal/non-retinal dataset would be needed to validate an input detector. The first browser prediction initializes the TFLite runtime; later predictions reuse it. The source Keras models in `models/` are ignored by Git, while the browser model and metadata in `static/models/` are included for deployment.

The previous inference code applied an additional Gaussian high-pass transform to images that were already Gaussian-filtered in the training dataset. That training/inference mismatch could distort retinal features and is now removed. The existing model weights were not retrained or replaced; a labeled dataset is required to determine whether the model's Severe/PDR accuracy improves.

The training report also prints exploratory sensitivity and specificity for a binary grouping of grades 2-4 versus 0-1. This is a research metric only, not a clinically selected threshold or evidence that the app can screen for disease. A qualified clinician must define the intended use and threshold, and an independent external dataset is needed for meaningful validation.

## Deploy to Vercel

The Vercel deployment is a static site: the TFLite model runs in the visitor's browser, so no TensorFlow serverless function or model API is needed. The first analysis downloads the model; uploaded images stay in the browser.

Install the JavaScript dependencies and build the static site:

```bash
npm ci
npm run build
```

Deploy the project root using the Vercel CLI or import this repository in Vercel. The included `vercel.json` runs `npm run build` and serves the generated `dist/` directory. No environment variables are required.

## Deploy to Render

This project includes a Render Blueprint in `render.yaml`. Push the project to a GitHub, GitLab, or Bitbucket repository, then create a new Blueprint in Render and select that repository. Render will run `npm ci && npm run build` and publish `dist/` as a static site. No Python runtime, environment variables, or server-side model are needed for the website; inference and image processing run in each visitor's browser.

The static deployment does not expose Flask's optional `/api/predict` endpoint. Use a separate Render web service if server-side Keras inference is required; that service needs the Python dependencies and a model artifact.

If you retrain the Keras model, export and commit the updated browser model before deploying:

```bash
python export_tflite.py
```

The Flask-served website and static site both use the same in-browser model. Server-side `/api/predict`, training, and TFLite export require TensorFlow and the Keras model.

## Important

This is a research and educational prototype, not a medical device or validated screening system. Predictions and scores are experimental, may be wrong, and are not calibrated probabilities or diagnoses. Do not use them to make health decisions. Consult a qualified eye-care professional for medical concerns.
