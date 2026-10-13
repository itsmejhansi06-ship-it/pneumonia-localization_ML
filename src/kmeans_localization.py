import ast

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans

from .config import (
    IMAGE_DIR,
    MODEL_DIR,
    PROCESSED_ROOT,
    RESULT_DIR,
    IMAGE_SIZE,
)

from .models.cnn import PneumoniaCNN

from .data_utils import (
    read_image,
    preprocess_image,
)

from .cam import (
    generate_cam,
    resize_cam,
)


# ==========================================================
# Parse ground-truth boxes
# ==========================================================

def parse_boxes(value):

    if pd.isna(value):
        return []

    try:

        parsed = ast.literal_eval(
            str(value)
        )

        return [
            tuple(
                float(v)
                for v in box
            )
            for box in parsed
        ]

    except Exception:

        return []


# ==========================================================
# IoU calculation
# ==========================================================

def calculate_iou(
    predicted_box,
    ground_truth_box
):

    if predicted_box is None:
        return 0.0

    px1, py1, px2, py2 = predicted_box

    gx1, gy1, gx2, gy2 = ground_truth_box

    # Intersection
    ix1 = max(
        px1,
        gx1
    )

    iy1 = max(
        py1,
        gy1
    )

    ix2 = min(
        px2,
        gx2
    )

    iy2 = min(
        py2,
        gy2
    )

    intersection_width = max(
        0,
        ix2 - ix1
    )

    intersection_height = max(
        0,
        iy2 - iy1
    )

    intersection_area = (
        intersection_width
        * intersection_height
    )

    # Predicted area
    predicted_area = (
        max(
            0,
            px2 - px1
        )
        *
        max(
            0,
            py2 - py1
        )
    )

    # Ground truth area
    ground_truth_area = (
        max(
            0,
            gx2 - gx1
        )
        *
        max(
            0,
            gy2 - gy1
        )
    )

    # Union
    union_area = (
        predicted_area
        +
        ground_truth_area
        -
        intersection_area
    )

    if union_area <= 0:
        return 0.0

    return (
        intersection_area
        /
        union_area
    )


# ==========================================================
# Best IoU against multiple ground-truth boxes
# ==========================================================

def best_ground_truth_iou(
    predicted_box,
    ground_truth_boxes
):

    if predicted_box is None:
        return 0.0

    if not ground_truth_boxes:
        return 0.0

    best_iou = 0.0

    for box in ground_truth_boxes:

        x, y, w, h = box

        ground_truth_xyxy = (
            x,
            y,
            x + w,
            y + h
        )

        iou = calculate_iou(
            predicted_box,
            ground_truth_xyxy
        )

        best_iou = max(
            best_iou,
            iou
        )

    return best_iou


# ==========================================================
# K-Means localization
# ==========================================================

def kmeans_bounding_box(
    cam,
    n_clusters=2,
    activation_percentile=70
):
    """
    Use K-Means clustering to identify high-activation
    regions in the CAM.

    Steps:

        CAM
         ↓
        Select high activation pixels
         ↓
        K-Means clustering
         ↓
        Select cluster with highest mean activation
         ↓
        Bounding box
    """

    height, width = cam.shape

    # ------------------------------------------------------
    # Select pixels above activation percentile
    # ------------------------------------------------------

    threshold = np.percentile(
        cam,
        activation_percentile
    )

    coordinates = np.argwhere(
        cam >= threshold
    )

    if len(coordinates) < n_clusters:

        return None

    # coordinates are:
    #
    # [row, column]
    #
    # Convert to:
    #
    # [x, y]

    points = np.column_stack(
        (
            coordinates[:, 1],
            coordinates[:, 0]
        )
    )

    # ------------------------------------------------------
    # K-Means
    # ------------------------------------------------------

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(
        points
    )

    # ------------------------------------------------------
    # Find cluster with highest mean CAM activation
    # ------------------------------------------------------

    best_cluster = None
    best_activation = -1

    for cluster_id in range(
        n_clusters
    ):

        cluster_points = points[
            labels == cluster_id
        ]

        if len(cluster_points) == 0:
            continue

        activations = []

        for x, y in cluster_points:

            activations.append(
                cam[
                    int(y),
                    int(x)
                ]
            )

        mean_activation = np.mean(
            activations
        )

        if mean_activation > best_activation:

            best_activation = (
                mean_activation
            )

            best_cluster = (
                cluster_id
            )

    if best_cluster is None:

        return None

    # ------------------------------------------------------
    # Get selected cluster
    # ------------------------------------------------------

    selected_points = points[
        labels == best_cluster
    ]

    if len(selected_points) == 0:

        return None

    # ------------------------------------------------------
    # Bounding box
    # ------------------------------------------------------

    x_coordinates = (
        selected_points[:, 0]
    )

    y_coordinates = (
        selected_points[:, 1]
    )

    x1 = int(
        x_coordinates.min()
    )

    y1 = int(
        y_coordinates.min()
    )

    x2 = int(
        x_coordinates.max()
    )

    y2 = int(
        y_coordinates.max()
    )

    # Ensure valid box

    if x2 <= x1 or y2 <= y1:

        return None

    return (
        x1,
        y1,
        x2,
        y2
    )


# ==========================================================
# Main
# ==========================================================

def main():

    # ======================================================
    # Device
    # ======================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("K-MEANS CAM LOCALIZATION EXPERIMENT")
    print("=" * 60)

    print(
        f"Using device: {device}"
    )

    # ======================================================
    # Load CNN
    # ======================================================

    model = PneumoniaCNN(
        num_classes=2
    ).to(device)

    model_path = (
        MODEL_DIR
        /
        "cnn_cam_baseline.pt"
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model not found:\n{model_path}"
        )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    model.eval()

    print(
        "CNN model loaded successfully."
    )

    # ======================================================
    # Load test CSV
    # ======================================================

    test_path = (
        PROCESSED_ROOT
        /
        "test.csv"
    )

    if not test_path.exists():

        raise FileNotFoundError(
            f"Test CSV not found:\n{test_path}"
        )

    test_df = pd.read_csv(
        test_path
    )

    # Only pneumonia images have
    # localization ground truth.

    pneumonia_df = test_df[
        test_df["Target"] == 1
    ].reset_index(
        drop=True
    )

    print(
        f"Pneumonia test images: "
        f"{len(pneumonia_df)}"
    )

    # ======================================================
    # K-Means settings
    # ======================================================

    cluster_values = [
        2,
        3,
        4
    ]

    percentile = 70

    all_results = []

    # ======================================================
    # Run experiments
    # ======================================================

    for n_clusters in cluster_values:

        print()
        print("-" * 60)

        print(
            f"K-Means clusters: "
            f"{n_clusters}"
        )

        ious = []

        for index, row in pneumonia_df.iterrows():

            patient_id = row[
                "patientId"
            ]

            image_path = (
                IMAGE_DIR
                /
                f"{patient_id}.jpg"
            )

            if not image_path.exists():

                continue

            # --------------------------------------------------
            # Read image
            # --------------------------------------------------

            image = read_image(
                image_path
            )

            original_height, original_width = (
                image.shape
            )

            # --------------------------------------------------
            # Preprocess
            # --------------------------------------------------

            processed = preprocess_image(
                image,
                IMAGE_SIZE
            )

            image_tensor = (
                torch.from_numpy(
                    processed
                )
                .unsqueeze(0)
                .unsqueeze(0)
                .float()
            )

            # --------------------------------------------------
            # Generate CAM
            # --------------------------------------------------

            cam = generate_cam(
                model=model,
                image_tensor=image_tensor,
                class_index=1,
                device=device
            )

            cam = resize_cam(
                cam,
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE
                )
            )

            # --------------------------------------------------
            # K-Means bounding box
            # --------------------------------------------------

            predicted_box = (
                kmeans_bounding_box(
                    cam,
                    n_clusters=n_clusters,
                    activation_percentile=percentile
                )
            )

            # --------------------------------------------------
            # Ground truth
            # --------------------------------------------------

            ground_truth_boxes = parse_boxes(
                row["boxes"]
            )

            # --------------------------------------------------
            # Scale ground truth to 128×128
            # --------------------------------------------------

            scale_x = (
                IMAGE_SIZE
                /
                original_width
            )

            scale_y = (
                IMAGE_SIZE
                /
                original_height
            )

            scaled_ground_truth = []

            for (
                x,
                y,
                w,
                h
            ) in ground_truth_boxes:

                scaled_ground_truth.append(
                    (
                        x * scale_x,
                        y * scale_y,
                        w * scale_x,
                        h * scale_y
                    )
                )

            # --------------------------------------------------
            # Calculate IoU
            # --------------------------------------------------

            iou = best_ground_truth_iou(
                predicted_box,
                scaled_ground_truth
            )

            ious.append(
                iou
            )

            # --------------------------------------------------
            # Progress
            # --------------------------------------------------

            if (
                index + 1
            ) % 25 == 0:

                print(
                    f"Processed "
                    f"{index + 1}/"
                    f"{len(pneumonia_df)}"
                )

        # ==================================================
        # Summary
        # ==================================================

        mean_iou = (
            np.mean(ious)
            if ious
            else 0.0
        )

        median_iou = (
            np.median(ious)
            if ious
            else 0.0
        )

        print(
            f"Mean IoU: "
            f"{mean_iou:.4f}"
        )

        print(
            f"Median IoU: "
            f"{median_iou:.4f}"
        )

        all_results.append(
            {
                "n_clusters": n_clusters,
                "activation_percentile":
                    percentile,
                "mean_iou": mean_iou,
                "median_iou": median_iou,
                "images": len(ious)
            }
        )

    # ======================================================
    # Save results
    # ======================================================

    results_df = pd.DataFrame(
        all_results
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        RESULT_DIR
        /
        "kmeans_localization_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    # ======================================================
    # Print final results
    # ======================================================

    print()
    print("=" * 60)
    print("K-MEANS EXPERIMENT COMPLETE")
    print("=" * 60)

    print()

    print(
        results_df.to_string(
            index=False
        )
    )

    print()
    print(
        f"Results saved to:\n"
        f"{output_path}"
    )


if __name__ == "__main__":
    main()