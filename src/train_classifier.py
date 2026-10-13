import argparse

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import (
    BATCH_SIZE,
    DEFAULT_EPOCHS,
    LEARNING_RATE,
    MODEL_DIR,
    PROCESSED_ROOT,
    RESULT_DIR,
)

from .data_utils import RSNAPneumoniaDataset
from .models.cnn import PneumoniaCNN


def load_split(filename):
    """
    Load a train/validation/test split from CSV.
    """

    path = PROCESSED_ROOT / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset split not found: {path}"
        )

    return pd.read_csv(path)


def run_epoch(
    model,
    loader,
    loss_fn,
    optimizer,
    device,
    training=True
):
    """
    Run one complete training or validation epoch.
    """

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    # No tqdm here.
    # This avoids the broken progress-bar display
    # in some Windows terminals.

    for x, y, _, _ in loader:

        x = x.to(device)
        y = y.to(device)

        # Clear gradients only during training
        if training:
            optimizer.zero_grad()

        # Forward pass
        with torch.set_grad_enabled(training):

            logits = model(x)

            loss = loss_fn(
                logits,
                y
            )

            # Backpropagation
            if training:
                loss.backward()
                optimizer.step()

        # Accumulate loss
        total_loss += (
            loss.item()
            * x.size(0)
        )

        # Predictions
        predictions = logits.argmax(
            dim=1
        )

        # Correct predictions
        correct += (
            predictions == y
        ).sum().item()

        total += x.size(0)

    average_loss = total_loss / total

    accuracy = correct / total

    return average_loss, accuracy


def main():

    # ==================================================
    # Command-line arguments
    # ==================================================

    parser = argparse.ArgumentParser(
        description="Train pneumonia classification CNN"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs"
    )

    args = parser.parse_args()

    # ==================================================
    # Select device
    # ==================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("PNEUMONIA CLASSIFICATION TRAINING")
    print("=" * 60)

    print(
        f"Using device: {device}"
    )

    # ==================================================
    # Load training and validation CSV files
    # ==================================================

    train_df = load_split(
        "train.csv"
    )

    val_df = load_split(
        "val.csv"
    )

    print(
        f"Training samples: {len(train_df)}"
    )

    print(
        f"Validation samples: {len(val_df)}"
    )

    # ==================================================
    # Create datasets
    # ==================================================

    train_dataset = RSNAPneumoniaDataset(
        train_df
    )

    val_dataset = RSNAPneumoniaDataset(
        val_df
    )

    # ==================================================
    # Create DataLoaders
    # ==================================================

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

    # ==================================================
    # Create CNN
    # ==================================================

    model = PneumoniaCNN(
        num_classes=2
    ).to(device)

    print(
        "Model created successfully."
    )

    # ==================================================
    # Loss function
    # ==================================================

    loss_fn = nn.CrossEntropyLoss()

    # ==================================================
    # Optimizer
    # ==================================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # ==================================================
    # Create output directories
    # ==================================================

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ==================================================
    # Training history
    # ==================================================

    history = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    best_val_accuracy = 0.0

    # ==================================================
    # Training loop
    # ==================================================

    for epoch_number in range(
        1,
        args.epochs + 1
    ):

        print()
        print(
            "=" * 60
        )

        print(
            f"Epoch {epoch_number}/{args.epochs}"
        )

        print(
            "=" * 60
        )

        # --------------------------------------------------
        # Training
        # --------------------------------------------------

        print(
            "Training..."
        )

        train_loss, train_accuracy = run_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            training=True
        )

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        print(
            "Validation..."
        )

        val_loss, val_accuracy = run_epoch(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            training=False
        )

        # --------------------------------------------------
        # Save history
        # --------------------------------------------------

        history["epoch"].append(
            epoch_number
        )

        history["train_loss"].append(
            train_loss
        )

        history["train_accuracy"].append(
            train_accuracy
        )

        history["val_loss"].append(
            val_loss
        )

        history["val_accuracy"].append(
            val_accuracy
        )

        # --------------------------------------------------
        # Display results
        # --------------------------------------------------

        print()
        print(
            f"Train Loss:       {train_loss:.4f}"
        )

        print(
            f"Train Accuracy:   {train_accuracy:.4f}"
        )

        print(
            f"Validation Loss:  {val_loss:.4f}"
        )

        print(
            f"Validation Accuracy: {val_accuracy:.4f}"
        )

        # --------------------------------------------------
        # Save best model
        # --------------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            model_path = (
                MODEL_DIR
                / "cnn_cam_baseline.pt"
            )

            torch.save(
                model.state_dict(),
                model_path
            )

            print()
            print(
                "Best model saved:"
            )

            print(
                model_path
            )

    # ==================================================
    # Save training history
    # ==================================================

    history_df = pd.DataFrame(
        history
    )

    history_path = (
        RESULT_DIR
        / "training_history.csv"
    )

    history_df.to_csv(
        history_path,
        index=False
    )

    # ==================================================
    # Plot accuracy
    # ==================================================

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        history["epoch"],
        history["train_accuracy"],
        marker="o",
        label="Train Accuracy"
    )

    plt.plot(
        history["epoch"],
        history["val_accuracy"],
        marker="o",
        label="Validation Accuracy"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        "Training and Validation Accuracy"
    )

    plt.legend()

    plt.tight_layout()

    accuracy_path = (
        RESULT_DIR
        / "training_accuracy.png"
    )

    plt.savefig(
        accuracy_path
    )

    plt.close()

    # ==================================================
    # Plot loss
    # ==================================================

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        history["epoch"],
        history["train_loss"],
        marker="o",
        label="Train Loss"
    )

    plt.plot(
        history["epoch"],
        history["val_loss"],
        marker="o",
        label="Validation Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "Training and Validation Loss"
    )

    plt.legend()

    plt.tight_layout()

    loss_path = (
        RESULT_DIR
        / "training_loss.png"
    )

    plt.savefig(
        loss_path
    )

    plt.close()

    # ==================================================
    # Final message
    # ==================================================

    print()
    print("=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.4f}"
    )

    print()
    print(
        f"Model saved at:\n{MODEL_DIR / 'cnn_cam_baseline.pt'}"
    )

    print()
    print(
        f"Training history:\n{history_path}"
    )

    print()
    print(
        f"Accuracy plot:\n{accuracy_path}"
    )

    print()
    print(
        f"Loss plot:\n{loss_path}"
    )


if __name__ == "__main__":
    main()