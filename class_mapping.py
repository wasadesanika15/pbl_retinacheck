CLASS_TO_INDEX = {
    "No DR": 0,
    "Mild": 1,
    "Moderate": 2,
    "Severe": 3,
    "PDR": 4,
}

CLASS_NAMES = tuple(
    name for name, _ in sorted(CLASS_TO_INDEX.items(), key=lambda item: item[1])
)

DATASET_LABEL_TO_CLASS_INDEX = {
    0: CLASS_TO_INDEX["No DR"],
    1: CLASS_TO_INDEX["Mild"],
    2: CLASS_TO_INDEX["Moderate"],
    3: CLASS_TO_INDEX["Severe"],
    4: CLASS_TO_INDEX["PDR"],
}

PREPROCESSING = "rgb-resize-bilinear-0-255-v1"
