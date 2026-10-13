import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset


class OriginalSegmentedDataset(Dataset):

    def __init__(
        self,
        csv_path,
        original_root="data/raw/images",
        segmented_root="data/processed/segmented_images",
        image_size=128
    ):
        self.df = pd.read_csv(csv_path)

        self.original_root = original_root
        self.segmented_root = segmented_root
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        patient_id = str(row["patientId"])
        target = int(row["Target"])

        # ----------------------------------------------------
        # Original X-ray
        # ----------------------------------------------------

        original_path = self.find_original(
            patient_id
        )

        if original_path is None:
            raise FileNotFoundError(
                f"Original image not found: {patient_id}"
            )

        original = Image.open(
            original_path
        ).convert("L")

        # ----------------------------------------------------
        # Segmented X-ray
        # ----------------------------------------------------

        segmented_path = os.path.join(
            self.segmented_root,
            patient_id + ".png"
        )

        if not os.path.exists(segmented_path):
            raise FileNotFoundError(
                f"Segmented image not found: "
                f"{segmented_path}"
            )

        segmented = Image.open(
            segmented_path
        ).convert("L")

        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        original = original.resize(
            (self.image_size, self.image_size),
            Image.Resampling.BILINEAR
        )

        segmented = segmented.resize(
            (self.image_size, self.image_size),
            Image.Resampling.BILINEAR
        )

        # ----------------------------------------------------
        # Convert to NumPy
        # ----------------------------------------------------

        original = np.array(
            original,
            dtype=np.float32
        ) / 255.0

        segmented = np.array(
            segmented,
            dtype=np.float32
        ) / 255.0

        # ----------------------------------------------------
        # Stack as TWO channels
        # ----------------------------------------------------

        image = np.stack(
            [original, segmented],
            axis=0
        )

        image = torch.from_numpy(
            image
        )

        target = torch.tensor(
            target,
            dtype=torch.long
        )

        return image, target

    def find_original(self, patient_id):

        extensions = [
            ".jpg",
            ".jpeg",
            ".png"
        ]

        for ext in extensions:

            path = os.path.join(
                self.original_root,
                patient_id + ext
            )

            if os.path.exists(path):
                return path

        return None