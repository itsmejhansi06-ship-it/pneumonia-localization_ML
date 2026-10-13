from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import IMAGE_DIR, LABEL_CSV, IMAGE_SIZE


def load_labels(csv_path=LABEL_CSV):
    """Load the RSNA pneumonia labels CSV."""

    df = pd.read_csv(csv_path)

    required = {"patientId", "Target"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    return df


def make_patient_table(df):
    """
    Create one row per patient.

    Target:
        0 = No pneumonia
        1 = Pneumonia

    For pneumonia patients, store all available
    ground-truth bounding boxes.
    """

    rows = []

    for patient_id, group in df.groupby(
        "patientId",
        sort=False
    ):

        target = int(group["Target"].max())

        boxes = []

        if target == 1:

            for _, r in group.iterrows():

                vals = [
                    r.get("x"),
                    r.get("y"),
                    r.get("width"),
                    r.get("height")
                ]

                if all(pd.notna(v) for v in vals):

                    boxes.append(
                        tuple(
                            float(v)
                            for v in vals
                        )
                    )

        rows.append(
            {
                "patientId": patient_id,
                "Target": target,
                "boxes": boxes
            }
        )

    return pd.DataFrame(rows)


def read_image(path):
    """
    Read a JPG chest X-ray as grayscale.
    """

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {path}"
        )

    image = image.astype(np.float32)

    # Normalize to [0, 1]
    image -= image.min()

    max_value = image.max()

    if max_value > 0:
        image /= max_value

    return image


def preprocess_image(
    image,
    size=IMAGE_SIZE
):
    """
    Resize image and standardize pixel values.
    """

    image = cv2.resize(
        image,
        (size, size),
        interpolation=cv2.INTER_AREA
    )

    image = image.astype(np.float32)

    mean = image.mean()
    std = image.std()

    image = (
        image - mean
    ) / (std + 1e-8)

    return image


def resize_box(
    box,
    original_shape,
    new_size=IMAGE_SIZE
):
    """
    Convert original bounding box coordinates
    to resized-image coordinates.

    Input:
        x, y, width, height

    Output:
        x1, y1, x2, y2
    """

    x, y, w, h = box

    old_h, old_w = original_shape[:2]

    scale_x = new_size / old_w
    scale_y = new_size / old_h

    x1 = x * scale_x
    y1 = y * scale_y

    x2 = (x + w) * scale_x
    y2 = (y + h) * scale_y

    return (
        x1,
        y1,
        x2,
        y2
    )


def xywh_to_xyxy(box):

    x, y, w, h = box

    return (
        x,
        y,
        x + w,
        y + h
    )


class RSNAPneumoniaDataset(Dataset):

    def __init__(
        self,
        table,
        image_dir=IMAGE_DIR,
        image_size=IMAGE_SIZE
    ):

        self.table = table.reset_index(
            drop=True
        )

        self.image_dir = Path(
            image_dir
        )

        self.image_size = image_size

    def __len__(self):

        return len(self.table)

    def __getitem__(self, idx):

        row = self.table.iloc[idx]

        # JPEG filename = patientId
        path = (
            self.image_dir
            / f"{row.patientId}.jpg"
        )

        image = read_image(path)

        original_shape = image.shape

        processed = preprocess_image(
            image,
            self.image_size
        )

        # [1, 128, 128]
        tensor = (
            torch.from_numpy(processed)
            .unsqueeze(0)
            .float()
        )

        label = torch.tensor(
            int(row.Target),
            dtype=torch.long
        )

        return (
            tensor,
            label,
            row.patientId,
            original_shape
        )


def scaled_ground_truth_boxes(
    row,
    original_shape,
    size=IMAGE_SIZE
):

    return [
        resize_box(
            box,
            original_shape,
            size
        )
        for box in row.boxes
    ]