import ast
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from .config import (
    IMAGE_DIR,
    MODEL_DIR,
    PROCESSED_ROOT,
    RESULT_DIR,
    IMAGE_SIZE
)

from .models.cnn import PneumoniaCNN

from .data_utils import (
    read_image,
    preprocess_image
)

from .cam import (
    generate_cam,
    resize_cam,
    overlay_cam
)

from .localization import (
    extract_bounding_box,
    draw_bounding_box,
    draw_ground_truth_boxes
)


def parse_boxes(value):
    """
    Convert the boxes column from CSV string
    back into a Python list.

    Example:

    "[(100.0, 200.0, 300.0, 150.0)]"

    becomes:

    [(100.0, 200.0, 300.0, 150.0)]
    """

    if pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    try:

        parsed = ast.literal_eval(
            str(value)
        )

        return [
            tuple(float(v) for v in box)
            for box in parsed
        ]

    except Exception:

        return []


def calculate_iou(
    predicted_box,
    ground_truth_box
):
    """
    Calculate Intersection over Union.

    Boxes use:

        x1, y1, x2, y2
    """

    if predicted_box is None:
        return 0.0

    px1, py1, px2, py2 = predicted_box

    gx1, gy1, gx2, gy2 = ground_truth_box

    # Intersection coordinates
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

    # Areas
    predicted_area = max(
        0,
        px2 - px1
    ) * max(
        0,
        py2 - py1
    )

    ground_truth_area = max(
        0,
        gx2 - gx1
    ) * max(
        0,
        gy2 - gy1
    )

    union_area = (
        predicted_area
        + ground_truth_area
        - intersection_area
    )

    if union_area <= 0:
        return 0.0

    return (
        intersection_area
        / union_area
    )


def best_ground_truth_iou(
    predicted_box,
    ground_truth_boxes
):
    """
    If multiple pneumonia boxes exist,
    compare the predicted box against every
    ground-truth box and keep the best IoU.
    """

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


def main():

    # ==================================================
    # Configuration
    # ==================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("CAM-BASED PNEUMONIA LOCALIZATION")
    print("=" * 60)

    print(
        f"Using device: {device}"
    )

    # ==================================================
    # Load model
    # ==================================================

    model = PneumoniaCNN(
        num_classes=2
    ).to(device)

    model_path = (
        MODEL_DIR /
        "cnn_cam_baseline.pt"
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    model.eval()

    print(
        f"Loaded model:\n{model_path}"
    )

    # ==================================================
    # Load test CSV
    # ==================================================

    test_path = (
        PROCESSED_ROOT /
        "test.csv"
    )

    test_df = pd.read_csv(
        test_path
    )

    print(
        f"Test images: {len(test_df)}"
    )

    # ==================================================
    # Output directories
    # ==================================================

    heatmap_dir = (
        RESULT_DIR /
        "heatmaps"
    )

    bbox_dir = (
        RESULT_DIR /
        "bounding_boxes"
    )

    heatmap_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    bbox_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ==================================================
    # Localization results
    # ==================================================

    localization_results = []

    # We only need pneumonia test images
    # for localization evaluation.

    pneumonia_df = test_df[
        test_df["Target"] == 1
    ].reset_index(drop=True)

    print(
        f"Pneumonia test images: "
        f"{len(pneumonia_df)}"
    )

    # ==================================================
    # Process each pneumonia image
    # ==================================================

    for index, row in pneumonia_df.iterrows():

        patient_id = row["patientId"]

        print(
            f"\n[{index + 1}/{len(pneumonia_df)}] "
            f"{patient_id}"
        )

        image_path = (
            IMAGE_DIR /
            f"{patient_id}.jpg"
        )

        if not image_path.exists():

            print(
                "Image not found. Skipping."
            )

            continue

        # ----------------------------------------------
        # Read original image
        # ----------------------------------------------

        original_image = read_image(
            image_path
        )

        original_height, original_width = (
            original_image.shape
        )

        # ----------------------------------------------
        # Preprocess
        # ----------------------------------------------

        processed = preprocess_image(
            original_image,
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

        # ----------------------------------------------
        # Classification
        # ----------------------------------------------

        with torch.no_grad():

            logits = model(
                image_tensor.to(device)
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            predicted_class = (
                logits.argmax(
                    dim=1
                )
                .item()
            )

            pneumonia_probability = (
                probabilities[0, 1]
                .item()
            )

        print(
            f"Predicted class: "
            f"{predicted_class}"
        )

        print(
            f"Pneumonia probability: "
            f"{pneumonia_probability:.4f}"
        )

        # ----------------------------------------------
        # Generate CAM
        # ----------------------------------------------

        cam = generate_cam(
            model=model,
            image_tensor=image_tensor,
            class_index=1,
            device=device
        )

        # CAM is smaller than original image.
        # Resize it to 128 x 128 first.
        cam_resized = resize_cam(
            cam,
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        # ----------------------------------------------
        # Extract bounding box
        # ----------------------------------------------

        predicted_box, binary_mask = (
            extract_bounding_box(
                cam_resized,
                threshold=0.50,
                min_area=10
            )
        )

        # ----------------------------------------------
        # Ground truth
        # ----------------------------------------------

        ground_truth_boxes = parse_boxes(
            row["boxes"]
        )

        # Convert GT boxes to 128x128 coordinates
        scaled_gt_boxes = []

        scale_x = (
            IMAGE_SIZE
            / original_width
        )

        scale_y = (
            IMAGE_SIZE
            / original_height
        )

        for (
            x,
            y,
            w,
            h
        ) in ground_truth_boxes:

            scaled_gt_boxes.append(
                (
                    x * scale_x,
                    y * scale_y,
                    w * scale_x,
                    h * scale_y
                )
            )

        # ----------------------------------------------
        # IoU
        # ----------------------------------------------

        iou = best_ground_truth_iou(
            predicted_box,
            scaled_gt_boxes
        )

        print(
            f"Predicted box: "
            f"{predicted_box}"
        )

        print(
            f"IoU: {iou:.4f}"
        )

        # ----------------------------------------------
        # Create CAM overlay
        # ----------------------------------------------

        overlay = overlay_cam(
            processed,
            cam_resized
        )

        # ----------------------------------------------
        # Draw predicted box
        # ----------------------------------------------

        overlay_with_pred = draw_bounding_box(
            overlay,
            predicted_box,
            color=(0, 255, 0),
            thickness=2
        )

        # ----------------------------------------------
        # Draw ground-truth boxes
        # ----------------------------------------------

        final_visualization = (
            draw_ground_truth_boxes(
                overlay_with_pred,
                scaled_gt_boxes,
                color=(255, 0, 0),
                thickness=2
            )
        )

        # ----------------------------------------------
        # Save CAM heatmap
        # ----------------------------------------------

        heatmap_path = (
            heatmap_dir /
            f"{patient_id}_cam.jpg"
        )

        cv2.imwrite(
            str(heatmap_path),
            overlay
        )

        # ----------------------------------------------
        # Save bounding-box visualization
        # ----------------------------------------------

        bbox_path = (
            bbox_dir /
            f"{patient_id}_localization.jpg"
        )

        cv2.imwrite(
            str(bbox_path),
            final_visualization
        )

        # ----------------------------------------------
        # Save result
        # ----------------------------------------------

        localization_results.append(
            {
                "patientId": patient_id,
                "predicted_class": predicted_class,
                "pneumonia_probability":
                    pneumonia_probability,
                "predicted_x1":
                    None
                    if predicted_box is None
                    else predicted_box[0],
                "predicted_y1":
                    None
                    if predicted_box is None
                    else predicted_box[1],
                "predicted_x2":
                    None
                    if predicted_box is None
                    else predicted_box[2],
                "predicted_y2":
                    None
                    if predicted_box is None
                    else predicted_box[3],
                "iou": iou
            }
        )

    # ==================================================
    # Save localization results
    # ==================================================

    results_df = pd.DataFrame(
        localization_results
    )

    results_path = (
        RESULT_DIR /
        "cam_localization_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    # ==================================================
    # Summary
    # ==================================================

    if len(results_df) > 0:

        mean_iou = (
            results_df["iou"]
            .mean()
        )

        median_iou = (
            results_df["iou"]
            .median()
        )

        print()
        print("=" * 60)
        print("LOCALIZATION SUMMARY")
        print("=" * 60)

        print(
            f"Images evaluated: "
            f"{len(results_df)}"
        )

        print(
            f"Mean IoU: "
            f"{mean_iou:.4f}"
        )

        print(
            f"Median IoU: "
            f"{median_iou:.4f}"
        )

        print(
            f"\nResults saved to:\n"
            f"{results_path}"
        )

    else:

        print(
            "\nNo localization results were generated."
        )


if __name__ == "__main__":
    main()