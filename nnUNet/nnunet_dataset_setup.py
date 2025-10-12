import os
import shutil
import random

# Paths
basepart_dir = "/space/local/cug/basePart"
dataset_dir = "/space/local/cug/nnUNet_raw/Dataset001_PROSTATE"
#dataset_dir = "/space/slow/cug/dataset"

imagesTr_dir = os.path.join(dataset_dir, "imagesTr")
imagesTs_dir = os.path.join(dataset_dir, "imagesTs")
labelsTr_dir = os.path.join(dataset_dir, "labelsTr")
labelsTs_dir = os.path.join(dataset_dir, "labelsTs")

# Create directories if they don't exist
for d in [imagesTr_dir, imagesTs_dir, labelsTr_dir, labelsTs_dir]:
    os.makedirs(d, exist_ok=True)

# Collect all images and corresponding prostate labels
data = []

for folder in os.listdir(basepart_dir):
    folder_path = os.path.join(basepart_dir, folder)
    if os.path.isdir(folder_path):
        image_path = os.path.join(folder_path, "MR_StorT2", "image.nii.gz")
        label_path = os.path.join(folder_path, "MR_StorT2", "mask_CTVT_427.nii.gz")  # prostate label
        if os.path.exists(image_path) and os.path.exists(label_path):
            data.append((image_path, label_path))
        else:
            print(f"Skipping {folder}: missing image or label.")

# Shuffle and split 70:30
random.seed(42)
random.shuffle(data)
split_idx = int(0.7 * len(data))
train_data = data[:split_idx]
test_data = data[split_idx:]

# Helper function to copy and rename
def copy_and_rename(data_list, images_dir, labels_dir):
    for idx, (img, lbl) in enumerate(data_list):
        case_name = f"case_{idx:03d}"
        shutil.copy(img, os.path.join(images_dir, f"{case_name}.nii.gz"))
        shutil.copy(lbl, os.path.join(labels_dir, f"{case_name}_seg.nii.gz"))

# Copy training and testing data
copy_and_rename(train_data, imagesTr_dir, labelsTr_dir)
copy_and_rename(test_data, imagesTs_dir, labelsTs_dir)

print(f"Training: {len(train_data)} cases")
print(f"Testing: {len(test_data)} cases")
