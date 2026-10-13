import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset_unet import OriginalSegmentedDataset
from src.models.cnn_unet import PneumoniaCNNWithUNet


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4

TRAIN_CSV = "data/processed/train.csv"
VAL_CSV = "data/processed/val.csv"
TEST_CSV = "data/processed/test.csv"

MODEL_PATH = "models/cnn_unet.pt"


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# DATASETS
# ============================================================

train_dataset = OriginalSegmentedDataset(
    TRAIN_CSV,
    image_size=IMAGE_SIZE
)

val_dataset = OriginalSegmentedDataset(
    VAL_CSV,
    image_size=IMAGE_SIZE
)

test_dataset = OriginalSegmentedDataset(
    TEST_CSV,
    image_size=IMAGE_SIZE
)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Test samples:", len(test_dataset))


# ============================================================
# DATALOADERS
# ============================================================

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

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# MODEL
# ============================================================

model = PneumoniaCNNWithUNet(
    num_classes=2
).to(device)


# ============================================================
# LOSS AND OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def evaluate(model, loader):

    model.eval()

    total = 0
    correct = 0

    total_loss = 0.0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            total_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    accuracy = (
        correct / total
    )

    average_loss = (
        total_loss / len(loader)
    )

    return average_loss, accuracy


# ============================================================
# TRAINING
# ============================================================

best_val_accuracy = 0.0

os.makedirs(
    "models",
    exist_ok=True
)

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    correct = 0
    total = 0

    print(
        f"\nEpoch [{epoch + 1}/{EPOCHS}]"
    )

    for batch_idx, (images, labels) in enumerate(
        train_loader
    ):

        images = images.to(device)
        labels = labels.to(device)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Loss
        loss = criterion(
            outputs,
            labels
        )

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += loss.item()

        # Predictions
        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

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

    train_loss = (
        running_loss / len(train_loader)
    )

    train_accuracy = (
        correct / total
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val_loss, val_accuracy = evaluate(
        model,
        val_loader
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Val Loss: {val_loss:.4f}"
    )

    print(
        f"Val Accuracy: "
        f"{val_accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "val_accuracy":
                    best_val_accuracy,

                "image_size":
                    IMAGE_SIZE
            },
            MODEL_PATH
        )

        print(
            f"✓ Best model saved "
            f"(Val Accuracy = "
            f"{best_val_accuracy * 100:.2f}%)"
        )


# ============================================================
# FINAL TEST
# ============================================================

print("\n================================")
print("FINAL TEST")
print("================================")

# Load best model
checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

test_loss, test_accuracy = evaluate(
    model,
    test_loader
)

print(
    f"Test Loss: {test_loss:.4f}"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Model saved to: {MODEL_PATH}"
)