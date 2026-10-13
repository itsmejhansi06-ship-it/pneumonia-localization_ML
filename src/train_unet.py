import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from src.models.unet import UNet


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 64
BATCH_SIZE = 2
EPOCHS = 10
LEARNING_RATE = 1e-3
SEED = 42

CSV_PATH = "data/unet/train.csv"
IMAGE_ROOT = "data/unet/images/images"
MASK_ROOT = "data/unet/masks/masks"

MODEL_PATH = "models/unet_lung.pt"


# ============================================================
# CPU SETTINGS
# ============================================================

torch.set_num_threads(2)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DATASET
# ============================================================

class LungSegmentationDataset(Dataset):

    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_name = str(row["image"])
        mask_name = str(row["label"])

        # Remove .png
        image_id = os.path.splitext(image_name)[0]

        # X-ray:
        # data/unet/images/images/<image-id>/image.png
        image_path = os.path.join(
            IMAGE_ROOT,
            image_id,
            "image.png"
        )

        # Mask:
        # data/unet/masks/masks/Label_X.png
        mask_path = os.path.join(
            MASK_ROOT,
            mask_name
        )

        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"X-ray not found: {image_path}"
            )

        if not os.path.exists(mask_path):
            raise FileNotFoundError(
                f"Mask not found: {mask_path}"
            )

        # Load X-ray
        image = Image.open(image_path).convert("L")

        # Load lung mask
        mask = Image.open(mask_path).convert("L")

        # Resize X-ray
        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BILINEAR
        )

        # Resize mask
        # Nearest-neighbor preserves 0/1 labels
        mask = mask.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.NEAREST
        )

        # Convert X-ray to [0,1]
        image = np.array(
            image,
            dtype=np.float32
        ) / 255.0

        # Convert mask
        mask = np.array(
            mask,
            dtype=np.float32
        )

        # Make binary
        mask = (mask > 0.5).astype(
            np.float32
        )

        # Add channel dimension
        image = torch.from_numpy(
            image
        ).unsqueeze(0)

        mask = torch.from_numpy(
            mask
        ).unsqueeze(0)

        return image, mask


# ============================================================
# DICE LOSS
# ============================================================

class DiceLoss(nn.Module):

    def forward(self, predictions, targets):

        predictions = torch.sigmoid(
            predictions
        )

        predictions = predictions.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (
            predictions * targets
        ).sum()

        dice = (
            2.0 * intersection + 1e-7
        ) / (
            predictions.sum()
            + targets.sum()
            + 1e-7
        )

        return 1.0 - dice


# ============================================================
# IoU
# ============================================================

def calculate_iou(predictions, targets):

    predictions = torch.sigmoid(
        predictions
    )

    predictions = (
        predictions > 0.5
    ).float()

    intersection = (
        predictions * targets
    ).sum()

    union = (
        predictions.sum()
        + targets.sum()
        - intersection
    )

    iou = (
        intersection + 1e-7
    ) / (
        union + 1e-7
    )

    return iou.item()


# ============================================================
# MAIN
# ============================================================

def main():

    # CPU
    device = torch.device("cpu")

    print("================================")
    print("       U-NET TRAINING")
    print("================================")

    print("Device:", device)
    print("Image size:", IMAGE_SIZE)
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = LungSegmentationDataset(
        CSV_PATH
    )

    print(
        "Total image-mask pairs:",
        len(dataset)
    )

    # --------------------------------------------------------
    # Train / validation split
    # --------------------------------------------------------

    train_size = int(
        0.8 * len(dataset)
    )

    val_size = (
        len(dataset) - train_size
    )

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    print(
        "Training samples:",
        len(train_dataset)
    )

    print(
        "Validation samples:",
        len(val_dataset)
    )

    # --------------------------------------------------------
    # Data loaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = UNet().to(device)

    print("Model created.")
    print("Starting training...")
    print()

    # --------------------------------------------------------
    # Loss functions
    # --------------------------------------------------------

    bce_loss = nn.BCEWithLogitsLoss()

    dice_loss = DiceLoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_iou = 0.0

    os.makedirs(
        "models",
        exist_ok=True
    )

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(EPOCHS):

        model.train()

        total_train_loss = 0.0

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] starting...",
            flush=True
        )

        # ----------------------------------------------------
        # Training batches
        # ----------------------------------------------------

        for batch_idx, (images, masks) in enumerate(
            train_loader
        ):

            images = images.to(device)
            masks = masks.to(device)

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # BCE + Dice
            loss_bce = bce_loss(
                outputs,
                masks
            )

            loss_dice = dice_loss(
                outputs,
                masks
            )

            loss = (
                loss_bce
                + loss_dice
            )

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

            total_train_loss += loss.item()

            # Print progress
            if (
                batch_idx == 0
                or (batch_idx + 1) % 20 == 0
            ):

                print(
                    f"  Batch "
                    f"{batch_idx + 1}/"
                    f"{len(train_loader)} "
                    f"Loss: {loss.item():.4f}",
                    flush=True
                )

        # Average training loss
        avg_train_loss = (
            total_train_loss
            / len(train_loader)
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        total_val_loss = 0.0
        total_iou = 0.0

        with torch.no_grad():

            for images, masks in val_loader:

                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)

                loss_bce = bce_loss(
                    outputs,
                    masks
                )

                loss_dice = dice_loss(
                    outputs,
                    masks
                )

                loss = (
                    loss_bce
                    + loss_dice
                )

                total_val_loss += loss.item()

                total_iou += calculate_iou(
                    outputs,
                    masks
                )

        # Average validation loss
        avg_val_loss = (
            total_val_loss
            / len(val_loader)
        )

        # Average IoU
        avg_iou = (
            total_iou
            / len(val_loader)
        )

        # ----------------------------------------------------
        # Print epoch results
        # ----------------------------------------------------

        print()

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] completed"
        )

        print(
            f"Train Loss: {avg_train_loss:.4f}"
        )

        print(
            f"Val Loss: {avg_val_loss:.4f}"
        )

        print(
            f"Val IoU: {avg_iou:.4f}"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if avg_iou > best_iou:

            best_iou = avg_iou

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "image_size":
                        IMAGE_SIZE,

                    "val_iou":
                        best_iou
                },
                MODEL_PATH
            )

            print(
                f"✓ Best model saved "
                f"(IoU = {best_iou:.4f})"
            )

        print()

    # ========================================================
    # COMPLETE
    # ========================================================

    print("================================")
    print("       TRAINING COMPLETE")
    print("================================")

    print(
        f"Best validation IoU: "
        f"{best_iou:.4f}"
    )

    print(
        f"Model saved to: {MODEL_PATH}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()