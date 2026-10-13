import ast
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from models.cnn_unet import PneumoniaCNNWithUNet


IMAGE_SIZE = 128

THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

MODEL_PATH = Path("models/cnn_unet.pt")
TEST_CSV = Path("data/processed/test.csv")
IMAGE_DIR = Path("data/raw/images")
SEGMENTED_DIR = Path("data/processed/segmented_images")


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

model = PneumoniaCNNWithUNet(num_classes=2)

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("CNN + U-Net model loaded.")
print()


# ------------------------------------------------------------
# Generate CAM
# ------------------------------------------------------------

def generate_cam(image_tensor):

    with torch.no_grad():

        logits, features = model(
            image_tensor,
            return_features=True
        )

    feature_maps = features[0]

    weights = model.fc.weight[1]

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


# ------------------------------------------------------------
# Extract multiple CAM boxes
# ------------------------------------------------------------

def cam_to_bboxes(cam, threshold):

    binary = (
        cam >= threshold
    ).astype(np.uint8)

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    boxes = []

    for label in range(1, num_labels):

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        if area < 4:
            continue

        x = stats[
            label,
            cv2.CC_STAT_LEFT
        ]

        y = stats[
            label,
            cv2.CC_STAT_TOP
        ]

        w = stats[
            label,
            cv2.CC_STAT_WIDTH
        ]

        h = stats[
            label,
            cv2.CC_STAT_HEIGHT
        ]

        boxes.append(
            (
                int(x),
                int(y),
                int(w),
                int(h)
            )
        )

    return boxes


# ------------------------------------------------------------
# IoU
# ------------------------------------------------------------

def calculate_iou(box1, box2):

    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xa = max(x1, x2)
    ya = max(y1, y2)

    xb = min(
        x1 + w1,
        x2 + w2
    )

    yb = min(
        y1 + h1,
        y2 + h2
    )

    intersection = (
        max(0, xb - xa) *
        max(0, yb - ya)
    )

    area1 = w1 * h1
    area2 = w2 * h2

    union = (
        area1 +
        area2 -
        intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


# ------------------------------------------------------------
# Best IoU
# ------------------------------------------------------------

def best_iou(predicted_boxes, ground_truth_boxes):

    if (
        len(predicted_boxes) == 0 or
        len(ground_truth_boxes) == 0
    ):
        return 0.0

    best = 0.0

    for pred in predicted_boxes:

        for gt in ground_truth_boxes:

            value = calculate_iou(
                pred,
                gt
            )

            best = max(
                best,
                value
            )

    return best


# ------------------------------------------------------------
# Load test set
# ------------------------------------------------------------

df = pd.read_csv(TEST_CSV)

# Store IoUs for each threshold
all_results = {
    threshold: []
    for threshold in THRESHOLDS
}

correct_results = {
    threshold: []
    for threshold in THRESHOLDS
}


# ------------------------------------------------------------
# Process pneumonia images
# ------------------------------------------------------------

count = 0

for _, row in df.iterrows():

    if int(row["Target"]) != 1:
        continue

    patient_id = row["patientId"]

    image_path = (
        IMAGE_DIR /
        f"{patient_id}.jpg"
    )

    segmented_path = (
        SEGMENTED_DIR /
        f"{patient_id}.png"
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

    original = cv2.resize(
        original,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    segmented = cv2.resize(
        segmented,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    original = (
        original.astype(
            np.float32
        ) / 255.0
    )

    segmented = (
        segmented.astype(
            np.float32
        ) / 255.0
    )

    image = np.stack(
        [
            original,
            segmented
        ],
        axis=0
    )

    image_tensor = torch.tensor(
        image,
        dtype=torch.float32
    ).unsqueeze(0)

    # Generate CAM
    cam, logits = generate_cam(
        image_tensor
    )

    # Classification
    prediction = torch.argmax(
        logits,
        dim=1
    ).item()

    correctly_classified = (
        prediction == 1
    )

    # Ground-truth boxes
    boxes = ast.literal_eval(
        row["boxes"]
    )

    ground_truth_boxes = []

    for box in boxes:

        x, y, w, h = box

        ground_truth_boxes.append(
            (
                x * IMAGE_SIZE / 1024,
                y * IMAGE_SIZE / 1024,
                w * IMAGE_SIZE / 1024,
                h * IMAGE_SIZE / 1024
            )
        )

    # Test each threshold
    for threshold in THRESHOLDS:

        predicted_boxes = cam_to_bboxes(
            cam,
            threshold
        )

        iou = best_iou(
            predicted_boxes,
            ground_truth_boxes
        )

        all_results[
            threshold
        ].append(iou)

        if correctly_classified:

            correct_results[
                threshold
            ].append(iou)

    count += 1

    if count % 25 == 0:

        print(
            f"Processed {count} "
            f"pneumonia images..."
        )


# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------

print()
print("========================================")
print("CAM THRESHOLD EXPERIMENT")
print("========================================")

print()

print(
    "Threshold | All Mean IoU | Correct Mean IoU"
)

print("----------------------------------------")

for threshold in THRESHOLDS:

    all_mean = np.mean(
        all_results[threshold]
    )

    correct_mean = np.mean(
        correct_results[threshold]
    )

    print(
        f"{threshold:9.2f} | "
        f"{all_mean:.4f}       | "
        f"{correct_mean:.4f}"
    )


# ------------------------------------------------------------
# Find best threshold
# ------------------------------------------------------------

best_threshold = max(
    THRESHOLDS,
    key=lambda t: np.mean(
        correct_results[t]
    )
)

best_score = np.mean(
    correct_results[
        best_threshold
    ]
)

print()

print(
    "BEST THRESHOLD:",
    best_threshold
)

print(
    "BEST MEAN IoU:",
    best_score
)

print()

print("Done.")