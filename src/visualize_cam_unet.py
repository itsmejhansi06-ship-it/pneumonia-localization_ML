import ast
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from models.cnn_unet import PneumoniaCNNWithUNet


# -----------------------------
# Configuration
# -----------------------------
IMAGE_SIZE = 128
THRESHOLD = 0.3

MODEL_PATH = Path("models/cnn_unet.pt")
TEST_CSV = Path("data/processed/test.csv")
IMAGE_DIR = Path("data/raw/images")
SEGMENTED_DIR = Path("data/processed/segmented_images")

OUTPUT_DIR = Path("results/cam_unet_visuals")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Load model
# -----------------------------
device = torch.device("cpu")

model = PneumoniaCNNWithUNet(num_classes=2)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Model loaded.")


# -----------------------------
# Generate CAM
# -----------------------------
def generate_cam(image_tensor, class_idx):

    with torch.no_grad():
        logits, features = model(
            image_tensor,
            return_features=True
        )

    feature_maps = features[0]
    weights = model.fc.weight[class_idx]

    cam = torch.zeros(
        feature_maps.shape[1:],
        dtype=torch.float32
    )

    for i, weight in enumerate(weights):
        cam += weight * feature_maps[i]

    cam = torch.relu(cam)

    cam = cam.detach().numpy()

    if cam.max() > cam.min():
        cam = (
            cam - cam.min()
        ) / (
            cam.max() - cam.min()
        )
    else:
        cam = np.zeros_like(cam)

    cam = cv2.resize(
        cam,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    return cam, logits


# -----------------------------
# CAM bounding box
# -----------------------------
def cam_to_bbox(cam):

    binary = (
        cam >= THRESHOLD
    ).astype(np.uint8)

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    if num_labels <= 1:
        return None

    # Largest connected region
    largest_label = 1 + np.argmax(
        stats[1:, cv2.CC_STAT_AREA]
    )

    x = stats[
        largest_label,
        cv2.CC_STAT_LEFT
    ]

    y = stats[
        largest_label,
        cv2.CC_STAT_TOP
    ]

    w = stats[
        largest_label,
        cv2.CC_STAT_WIDTH
    ]

    h = stats[
        largest_label,
        cv2.CC_STAT_HEIGHT
    ]

    return (
        int(x),
        int(y),
        int(w),
        int(h)
    )


# -----------------------------
# Process images
# -----------------------------
df = pd.read_csv(TEST_CSV)

count = 0

for _, row in df.iterrows():

    # Only pneumonia images
    if int(row["Target"]) != 1:
        continue

    patient_id = row["patientId"]

    image_path = (
        IMAGE_DIR / f"{patient_id}.jpg"
    )

    segmented_path = (
        SEGMENTED_DIR / f"{patient_id}.png"
    )

    if not image_path.exists():
        continue

    if not segmented_path.exists():
        continue

    original = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    segmented = cv2.imread(
        str(segmented_path),
        cv2.IMREAD_GRAYSCALE
    )

    if original is None or segmented is None:
        continue

    # Resize
    original = cv2.resize(
        original,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    segmented = cv2.resize(
        segmented,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    # Normalize
    original_norm = (
        original.astype(np.float32) / 255.0
    )

    segmented_norm = (
        segmented.astype(np.float32) / 255.0
    )

    # Two-channel input
    image = np.stack(
        [
            original_norm,
            segmented_norm
        ],
        axis=0
    )

    image_tensor = torch.tensor(
        image,
        dtype=torch.float32
    ).unsqueeze(0)

    # CAM for pneumonia class
    cam, logits = generate_cam(
        image_tensor,
        class_idx=1
    )

    probabilities = torch.softmax(
        logits,
        dim=1
    )[0]

    prediction = torch.argmax(
        probabilities
    ).item()

    pneumonia_probability = (
        probabilities[1].item()
    )

    predicted_box = cam_to_bbox(cam)

    # -----------------------------
    # Ground truth box
    # -----------------------------
    boxes = ast.literal_eval(
        row["boxes"]
    )

    ground_truth_box = None

    if len(boxes) > 0:

        x, y, w, h = boxes[0]

        ground_truth_box = (
            int(x * IMAGE_SIZE / 1024),
            int(y * IMAGE_SIZE / 1024),
            int(w * IMAGE_SIZE / 1024),
            int(h * IMAGE_SIZE / 1024)
        )

    # -----------------------------
    # Create CAM heatmap
    # -----------------------------
    heatmap = (
        cam * 255
    ).astype(np.uint8)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    original_color = cv2.cvtColor(
        original,
        cv2.COLOR_GRAY2BGR
    )

    overlay = cv2.addWeighted(
        original_color,
        0.6,
        heatmap,
        0.4,
        0
    )

    # -----------------------------
    # Draw predicted box
    # GREEN = predicted CAM box
    # -----------------------------
    if predicted_box is not None:

        x, y, w, h = predicted_box

        cv2.rectangle(
            overlay,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    # -----------------------------
    # Draw ground truth box
    # RED = RSNA box
    # -----------------------------
    if ground_truth_box is not None:

        x, y, w, h = ground_truth_box

        cv2.rectangle(
            overlay,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            2
        )

    # -----------------------------
    # Add information
    # -----------------------------
    text1 = f"Prediction: {prediction}"
    text2 = f"Pneumonia probability: {pneumonia_probability:.3f}"

    cv2.putText(
        overlay,
        text1,
        (5, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1
    )

    cv2.putText(
        overlay,
        text2,
        (5, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1
    )

    # -----------------------------
    # Save
    # -----------------------------
    output_path = (
        OUTPUT_DIR /
        f"{patient_id}.png"
    )

    cv2.imwrite(
        str(output_path),
        overlay
    )

    print(
        f"Saved: {output_path}"
    )

    count += 1

    # Only visualize 10 images
    if count >= 10:
        break


print()
print("==============================")
print("VISUALIZATION COMPLETE")
print("==============================")
print("Images created:", count)
print("Saved in:", OUTPUT_DIR)