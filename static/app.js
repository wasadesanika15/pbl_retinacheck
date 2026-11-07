const fileInput = document.querySelector("#image-input");
const dropZone = document.querySelector("#drop-zone");
const previewCard = document.querySelector("#preview-card");
const imagePreview = document.querySelector("#image-preview");
const fundusConfirmation = document.querySelector("#fundus-confirmation");
const fileName = document.querySelector("#file-name");
const fileSize = document.querySelector("#file-size");
const analyzeButton = document.querySelector("#analyze-button");
const removeButton = document.querySelector("#remove-image");
const errorMessage = document.querySelector("#error-message");
const emptyResult = document.querySelector("#empty-result");
const loadingResult = document.querySelector("#loading-result");
const predictionResult = document.querySelector("#prediction-result");
const classScores = document.querySelector("#class-scores");

let selectedFile = null;
let previewUrl = null;
let inferenceModelPromise = null;

const MODEL_URL = "/static/models/retinopathy.tflite";
const MODEL_METADATA_URL = "/static/models/metadata.json";
const PREPROCESSING = "rgb-resize-bilinear-0-255-v1";

function loadModel() {
  if (!inferenceModelPromise) {
    inferenceModelPromise = (async () => {
      if (!window.tf || !window.tflite) {
        throw new Error("The browser model runtime could not be loaded. Refresh and try again.");
      }

      const metadataResponse = await fetch(MODEL_METADATA_URL);
      if (!metadataResponse.ok) {
        throw new Error("The model settings could not be loaded. Refresh and try again.");
      }
      const metadata = await metadataResponse.json();
      if (
        !Number.isInteger(metadata.inputSize) ||
        metadata.inputSize < 32 ||
        !Array.isArray(metadata.classNames) ||
        metadata.classNames.length !== 5 ||
        new Set(metadata.classNames).size !== metadata.classNames.length ||
        metadata.preprocessing !== PREPROCESSING ||
        metadata.outputActivation !== "softmax"
      ) {
        throw new Error("The deployed model settings are invalid. Please try again later.");
      }

      await window.tf.setBackend("cpu");
      await window.tf.ready();
      window.tflite.setWasmPath("/static/vendor/");
      const model = await window.tflite.loadTFLiteModel(MODEL_URL, {
        numThreads: window.crossOriginIsolated
          ? Math.max(
              1,
              Math.min(4, Math.floor((navigator.hardwareConcurrency || 2) / 2)),
            )
          : 1,
      });
      const outputShape = model.outputs?.[0]?.shape;
      if (
        model.outputs?.length !== 1 ||
        !Array.isArray(outputShape) ||
        outputShape.length !== 2 ||
        outputShape[1] !== metadata.classNames.length
      ) {
        model.dispose();
        throw new Error("The browser model must return exactly five class scores.");
      }
      const warmupInput = window.tf.zeros([
        1,
        metadata.inputSize,
        metadata.inputSize,
        3,
      ]);
      let warmupOutput;
      try {
        warmupOutput = model.predict(warmupInput);
      } finally {
        warmupInput.dispose();
        if (Array.isArray(warmupOutput)) {
          warmupOutput.forEach((tensor) => tensor.dispose());
        } else {
          warmupOutput?.dispose();
        }
      }
      return {
        model,
        inputSize: metadata.inputSize,
        classNames: metadata.classNames,
      };
    })().catch((error) => {
      inferenceModelPromise = null;
      throw error;
    });
  }

  return inferenceModelPromise;
}

async function predictImage() {
  const { model, inputSize, classNames } = await loadModel();
  await imagePreview.decode();

  const sourcePixels = window.tf.browser.fromPixels(imagePreview);
  const pixels = window.tf.image.resizeBilinear(
    sourcePixels,
    [inputSize, inputSize],
    false,
    true,
  );
  const input = window.tf.expandDims(pixels, 0);
  let output;
  try {
    output = model.predict(input);
    if (Array.isArray(output)) output = output[0];
    const scores = Array.from(output.dataSync());
    if (
      scores.length !== classNames.length ||
      scores.some((score) => !Number.isFinite(score) || score < 0 || score > 1) ||
      Math.abs(scores.reduce((sum, score) => sum + score, 0) - 1) > 0.02
    ) {
      throw new Error("The model returned an invalid prediction. Please try again.");
    }
    const classIndex = scores.indexOf(Math.max(...scores));
    return {
      prediction: classNames[classIndex],
      confidence: scores[classIndex],
      scores,
      classNames,
    };
  } finally {
    sourcePixels.dispose();
    pixels.dispose();
    input.dispose();
    output?.dispose();
  }
}

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.hidden = false;
}

function clearError() {
  errorMessage.textContent = "";
  errorMessage.hidden = true;
}

function clearPreviewUrl() {
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }
}

function setFile(file) {
  clearError();
  if (!file) return;

  const allowedTypes = ["image/jpeg", "image/png"];
  if (!allowedTypes.includes(file.type)) {
    clearSelection();
    showError("Please choose a JPG, JPEG, or PNG image.");
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    clearSelection();
    showError("This image is larger than 10 MB. Choose a smaller file.");
    return;
  }

  clearPreviewUrl();
  selectedFile = file;
  previewUrl = URL.createObjectURL(file);
  imagePreview.src = previewUrl;
  fundusConfirmation.checked = false;
  fileName.textContent = file.name;
  fileSize.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB`;
  previewCard.hidden = false;
  dropZone.hidden = true;
  analyzeButton.disabled = true;
  emptyResult.hidden = false;
  predictionResult.hidden = true;
}

function clearSelection() {
  clearPreviewUrl();
  selectedFile = null;
  fileInput.value = "";
  imagePreview.removeAttribute("src");
  fundusConfirmation.checked = false;
  previewCard.hidden = true;
  dropZone.hidden = false;
  analyzeButton.disabled = true;
  clearError();
}

fundusConfirmation.addEventListener("change", () => {
  analyzeButton.disabled = !selectedFile || !fundusConfirmation.checked;
});

fileInput.addEventListener("change", () => setFile(fileInput.files[0]));
removeButton.addEventListener("click", clearSelection);

for (const eventName of ["dragenter", "dragover"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("is-dragging");
  });
}

for (const eventName of ["dragleave", "drop"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("is-dragging");
  });
}

dropZone.addEventListener("drop", (event) => {
  setFile(event.dataTransfer.files[0]);
});

analyzeButton.addEventListener("click", async () => {
  if (!selectedFile || !fundusConfirmation.checked) return;

  clearError();
  emptyResult.hidden = true;
  predictionResult.hidden = true;
  loadingResult.hidden = false;
  analyzeButton.disabled = true;
  analyzeButton.classList.add("is-loading");

  try {
    const result = await predictImage();

    document.querySelector("#prediction-label").textContent = result.prediction;
    const confidencePercent = result.confidence * 100;
    document.querySelector("#confidence-value").textContent =
      `${confidencePercent.toFixed(1)}%`;
    document.querySelector("#confidence-bar").style.width = `${confidencePercent}%`;
    classScores.replaceChildren(
      ...result.scores.map((score, index) => {
        const row = document.createElement("div");
        row.className = "class-score";

        const heading = document.createElement("div");
        heading.className = "class-score-heading";
        const label = document.createElement("span");
        label.textContent = result.classNames[index];
        const value = document.createElement("strong");
        value.textContent = `${(score * 100).toFixed(1)}%`;
        heading.append(label, value);

        const track = document.createElement("div");
        track.className = "class-score-track";
        const bar = document.createElement("span");
        bar.style.width = `${score * 100}%`;
        track.append(bar);
        row.append(heading, track);
        return row;
      }),
    );
    predictionResult.hidden = false;
  } catch (error) {
    emptyResult.hidden = false;
    showError(error instanceof Error ? error.message : "Something went wrong. Please try again.");
  } finally {
    loadingResult.hidden = true;
    analyzeButton.disabled = !selectedFile || !fundusConfirmation.checked;
    analyzeButton.classList.remove("is-loading");
  }
});
