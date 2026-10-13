import ast

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
    resize_cam
)

from .localization import (
    extract_bounding_box
)


def parse_boxes(value):

    if pd.isna(value):
        return []

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

    if predicted_box is None:
        return 0.0

    px1, py1, px2, py2 = predicted_box

    gx1, gy1, gx2, gy2 = ground_truth_box

    ix1 = max(px1, gx1)
    iy1 = max(py1, gy1)

    ix2 = min(px2, gx2)
    iy2 = min(py2, gy2)

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

    predicted_area = (
        max(0, px2 - px1)
        *
        max(0, py2 - py1)
    )

    ground_truth_area = (
        max(0, gx2 - gx1)
        *
        max(0, gy2 - gy1)
    )

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
        / union_area
    )


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

        gt_box = (
            x,
            y,
            x + w,
            y + h
        )

        iou = calculate_iou(
            predicted_box,
            gt_box
        )

        best_iou = max(
            best_iou,
            iou
        )

    return best_iou


def main():

    # ==================================================
    # Device
    # ==================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("CAM THRESHOLD EXPERIMENT")
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

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    model.eval()

    print(
        "Model loaded successfully."
    )

    # ==================================================
    # Load test data
    # ==================================================

    test_path = (
        PROCESSED_ROOT /
        "test.csv"
    )

    test_df = pd.read_csv(
        test_path
    )

    # Only pneumonia cases have
    # localization ground truth.

    pneumonia_df = test_df[
        test_df["Target"] == 1
    ].reset_index(drop=True)

    print(
        f"Pneumonia images: "
        f"{len(pneumonia_df)}"
    )

    # ==================================================
    # Thresholds
    # ==================================================

    thresholds = [
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80
    ]

    results = []

    # ==================================================
    # Process every image
    # ==================================================

    for threshold in thresholds:

        print()
        print(
            "-" * 60
        )

        print(
            f"Testing threshold: {threshold}"
        )

        ious = []

        for index, row in pneumonia_df.iterrows():

            patient_id = row["patientId"]

            image_path = (
                IMAGE_DIR /
                f"{patient_id}.jpg"
            )

            if not image_path.exists():
                continue

            # ------------------------------------------
            # Read image
            # ------------------------------------------

            image = read_image(
                image_path
            )

            original_height, original_width = (
                image.shape
            )

            # ------------------------------------------
            # Preprocess
            # ------------------------------------------

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

            # ------------------------------------------
            # Generate CAM
            # ------------------------------------------

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

            # ------------------------------------------
            # Bounding box
            # ------------------------------------------

            predicted_box, _ = (
                extract_bounding_box(
                    cam,
                    threshold=threshold,
                    min_area=10
                )
            )

            # ------------------------------------------
            # Ground truth
            # ------------------------------------------

            ground_truth_boxes = parse_boxes(
                row["boxes"]
            )

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

            # ------------------------------------------
            # IoU
            # ------------------------------------------

            iou = best_ground_truth_iou(
                predicted_box,
                scaled_gt_boxes
            )

            ious.append(iou)

        # ----------------------------------------------
        # Calculate mean / median IoU
        # ----------------------------------------------

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
            f"Mean IoU: {mean_iou:.4f}"
        )

        print(
            f"Median IoU: {median_iou:.4f}"
        )

        results.append(
            {
                "threshold": threshold,
                "mean_iou": mean_iou,
                "median_iou": median_iou,
                "images": len(ious)
            }
        )

    # ==================================================
    # Save results
    # ==================================================

    results_df = pd.DataFrame(
        results
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        RESULT_DIR /
        "threshold_experiment.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print()
    print("=" * 60)
    print("THRESHOLD EXPERIMENT COMPLETE")
    print("=" * 60)

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