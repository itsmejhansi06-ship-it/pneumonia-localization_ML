import ast
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from models.cnn_unet import PneumoniaCNNWithUNet


# ============================================================
# Configuration
# ============================================================

IMAGE_SIZE = 128

# CAM threshold
THRESHOLD = 0.5

# Ignore extremely small CAM regions
MIN_REGION_AREA = 4

MODEL_PATH = Path("models/cnn_unet.pt")
TEST_CSV = Path("data/processed/test.csv")
IMAGE_DIR = Path("data/raw/images")
SEGMENTED_DIR = Path("data/processed/segmented_images")

OUTPUT_DIR = Path("results/cam_unet")
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Load model
# ============================================================

device = torch.device("cpu")

model = PneumoniaCNNWithUNet(
    num_classes=2
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("CNN + U-Net model loaded.")


# ============================================================
# Generate CAM
# ============================================================

def generate_cam(
    model,
    image_tensor,
    class_idx
):

    with torch.no_grad():

        logits, features = model(
            image_tensor,
            return_features=True
        )

    # Final convolutional feature maps
    feature_maps = features[0]

    # FC weights for selected class
    weights = model.fc.weight[
        class_idx
    ]

    # Weighted combination of feature maps
    cam = torch.zeros(
        feature_maps.shape[1:],
        dtype=torch.float32
    )

    for i, weight in enumerate(weights):

        cam += (
            weight *
            feature_maps[i]
        )

    # ReLU
    cam = torch.relu(cam)

    # Convert tensor to NumPy
    cam = cam.detach().numpy()

    # Normalize
    if cam.max() > cam.min():

        cam = (
            cam - cam.min()
        ) / (
            cam.max() - cam.min()
        )

    else:

        cam = np.zeros_like(cam)

    # Resize to original CNN input size
    cam = cv2.resize(
        cam,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    return cam, logits


# ============================================================
# Extract MULTIPLE CAM regions
# ============================================================

def cam_to_bboxes(
    cam,
    threshold=THRESHOLD
):

    # Threshold CAM
    binary = (
        cam >= threshold
    ).astype(np.uint8)

    # Connected components
    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    predicted_boxes = []

    # Ignore background label 0
    for label in range(
        1,
        num_labels
    ):

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        # Remove tiny regions
        if area < MIN_REGION_AREA:

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

        predicted_boxes.append(
            (
                int(x),
                int(y),
                int(w),
                int(h)
            )
        )

    return predicted_boxes, binary


# ============================================================
# IoU between two boxes
# ============================================================

def calculate_iou(
    box1,
    box2
):

    if (
        box1 is None or
        box2 is None
    ):

        return 0.0

    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Intersection
    xa = max(
        x1,
        x2
    )

    ya = max(
        y1,
        y2
    )

    xb = min(
        x1 + w1,
        x2 + w2
    )

    yb = min(
        y1 + h1,
        y2 + h2
    )

    intersection_width = max(
        0,
        xb - xa
    )

    intersection_height = max(
        0,
        yb - ya
    )

    intersection = (
        intersection_width *
        intersection_height
    )

    # Areas
    area1 = (
        w1 *
        h1
    )

    area2 = (
        w2 *
        h2
    )

    union = (
        area1 +
        area2 -
        intersection
    )

    if union <= 0:

        return 0.0

    return (
        intersection /
        union
    )


# ============================================================
# Best IoU across ALL predicted and GT boxes
# ============================================================

def calculate_best_iou(
    predicted_boxes,
    ground_truth_boxes
):

    if (
        len(predicted_boxes) == 0 or
        len(ground_truth_boxes) == 0
    ):

        return 0.0, None, None

    best_iou = 0.0
    best_predicted_box = None
    best_ground_truth_box = None

    for predicted_box in predicted_boxes:

        for ground_truth_box in ground_truth_boxes:

            iou = calculate_iou(
                predicted_box,
                ground_truth_box
            )

            if iou > best_iou:

                best_iou = iou

                best_predicted_box = (
                    predicted_box
                )

                best_ground_truth_box = (
                    ground_truth_box
                )

    return (
        best_iou,
        best_predicted_box,
        best_ground_truth_box
    )


# ============================================================
# Read test dataset
# ============================================================

df = pd.read_csv(
    TEST_CSV
)

results = []

print()
print(
    "Starting CAM localization..."
)

print(
    "Test images:",
    len(df)
)

print()


# ============================================================
# Process test images
# ============================================================

for _, row in df.iterrows():

    patient_id = row[
        "patientId"
    ]

    target = int(
        row["Target"]
    )

    # --------------------------------------------------------
    # Only pneumonia images
    # --------------------------------------------------------

    if target != 1:

        continue

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Read images
    # --------------------------------------------------------

    original = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    segmented = cv2.imread(
        str(segmented_path),
        cv2.IMREAD_GRAYSCALE
    )

    if original is None:

        continue

    if segmented is None:

        continue

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    original = cv2.resize(
        original,
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    )

    segmented = cv2.resize(
        segmented,
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 2-channel input
    #
    # Channel 0 = original
    # Channel 1 = U-Net segmented
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Generate CAM for pneumonia class
    # --------------------------------------------------------

    cam, logits = generate_cam(
        model,
        image_tensor,
        class_idx=1
    )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Extract MULTIPLE CAM boxes
    # --------------------------------------------------------

    predicted_boxes, binary = (
        cam_to_bboxes(
            cam,
            threshold=THRESHOLD
        )
    )

    # --------------------------------------------------------
    # Read ALL ground-truth boxes
    # --------------------------------------------------------

    boxes = ast.literal_eval(
        row["boxes"]
    )

    ground_truth_boxes = []

    for box in boxes:

        x, y, w, h = box

        # Convert from 1024x1024
        # to 128x128

        gt_box = (
            x * IMAGE_SIZE / 1024,
            y * IMAGE_SIZE / 1024,
            w * IMAGE_SIZE / 1024,
            h * IMAGE_SIZE / 1024
        )

        ground_truth_boxes.append(
            gt_box
        )

    # --------------------------------------------------------
    # Best IoU
    # --------------------------------------------------------

    best_iou, best_predicted_box, best_gt_box = (
        calculate_best_iou(
            predicted_boxes,
            ground_truth_boxes
        )
    )

    # --------------------------------------------------------
    # Correct classification?
    # --------------------------------------------------------

    correctly_classified = (
        prediction == 1
    )

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    results.append(
        {
            "patientId":
                patient_id,

            "target":
                target,

            "prediction":
                prediction,

            "pneumonia_probability":
                pneumonia_probability,

            "num_predicted_boxes":
                len(predicted_boxes),

            "num_ground_truth_boxes":
                len(ground_truth_boxes),

            "predicted_boxes":
                str(predicted_boxes),

            "ground_truth_boxes":
                str(ground_truth_boxes),

            "best_predicted_box":
                str(best_predicted_box),

            "best_ground_truth_box":
                str(best_gt_box),

            "best_iou":
                best_iou,

            "correctly_classified":
                correctly_classified
        }
    )

    # Progress
    if len(results) % 25 == 0:

        print(
            f"Processed "
            f"{len(results)} "
            f"pneumonia images..."
        )


# ============================================================
# Create DataFrame
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# Save all results
# ============================================================

all_output = (
    OUTPUT_DIR /
    "cam_unet_results.csv"
)

results_df.to_csv(
    all_output,
    index=False
)


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print(
    "================================"
)

print(
    "CAM LOCALIZATION COMPLETE"
)

print(
    "================================"
)

print()

print(
    "Images evaluated:",
    len(results_df)
)


# ============================================================
# ALL PNEUMONIA IMAGES
# ============================================================

if len(results_df) > 0:

    mean_iou = (
        results_df[
            "best_iou"
        ].mean()
    )

    median_iou = (
        results_df[
            "best_iou"
        ].median()
    )

    print()

    print(
        "ALL PNEUMONIA IMAGES"
    )

    print(
        "Mean IoU:",
        mean_iou
    )

    print(
        "Median IoU:",
        median_iou
    )


# ============================================================
# CORRECTLY CLASSIFIED PNEUMONIA
# ============================================================

correct_df = results_df[
    results_df[
        "correctly_classified"
    ] == True
]

print()

print(
    "Correctly classified "
    "pneumonia images:",
    len(correct_df)
)


if len(correct_df) > 0:

    correct_mean_iou = (
        correct_df[
            "best_iou"
        ].mean()
    )

    correct_median_iou = (
        correct_df[
            "best_iou"
        ].median()
    )

    print()

    print(
        "CORRECTLY CLASSIFIED "
        "PNEUMONIA IMAGES"
    )

    print(
        "Mean IoU:",
        correct_mean_iou
    )

    print(
        "Median IoU:",
        correct_median_iou
    )


# ============================================================
# Save correctly classified results
# ============================================================

correct_output = (
    OUTPUT_DIR /
    "cam_unet_correct_results.csv"
)

correct_df.to_csv(
    correct_output,
    index=False
)


# ============================================================
# Summary
# ============================================================

print()

print(
    "All results saved to:"
)

print(
    all_output
)

print()

print(
    "Correctly classified "
    "results saved to:"
)

print(
    correct_output
)

print()

print("Done.")