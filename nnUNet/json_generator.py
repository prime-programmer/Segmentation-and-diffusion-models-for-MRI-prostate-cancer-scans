import os
import json

# Path to dataset folder
dataset_dir = "/space/local/cug/nnUNet_raw/Dataset001_PROSTATE"

imagesTr = os.path.join(dataset_dir, "imagesTr")
imagesTs = os.path.join(dataset_dir, "imagesTs")
labelsTr = os.path.join(dataset_dir, "labelsTr")

# Build dataset.json structure
dataset = {
    "name": "PROSTATE",
    "description": "Prostate MRI dataset",
    "tensorImageSize": "3D",
    "reference": "Custom dataset",
    "licence": "custom",
    "release": "1.0",
    "modality": {
        "0": "MRI-T2"
    },
    "labels": {
        "0": "background",
        "1": "prostate"
    },
    "numTraining": len([f for f in os.listdir(imagesTr) if f.endswith(".nii.gz")]),
    "numTest": len([f for f in os.listdir(imagesTs) if f.endswith(".nii.gz")]),
    "training": [
        {
            "image": f"./imagesTr/{f}",
            "label": f"./labelsTr/{f.replace('.nii.gz', '_seg.nii.gz')}"
        }
        for f in sorted(os.listdir(imagesTr)) if f.endswith(".nii.gz")
    ],
    "test": [
        f"./imagesTs/{f}"
        for f in sorted(os.listdir(imagesTs)) if f.endswith(".nii.gz")
    ]
}

# Save to dataset.json
with open(os.path.join(dataset_dir, "dataset.json"), "w") as f:
    json.dump(dataset, f, indent=4)

print("dataset.json created at:", os.path.join(dataset_dir, "dataset.json"))
