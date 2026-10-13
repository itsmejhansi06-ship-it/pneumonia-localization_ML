import pandas as pd
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from .config import (
    MODEL_DIR,
    PROCESSED_ROOT,
    RESULT_DIR,
    BATCH_SIZE
)

from .data_utils import RSNAPneumoniaDataset
from .models.cnn import PneumoniaCNN


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
    print("PNEUMONIA CLASSIFIER - TEST EVALUATION")
    print("=" * 60)

    print(
        f"Using device: {device}"
    )

    # ==================================================
    # Load test CSV
    # ==================================================

    test_path = (
        PROCESSED_ROOT / "test.csv"
    )

    if not test_path.exists():
        raise FileNotFoundError(
            f"Test CSV not found: {test_path}"
        )

    test_df = pd.read_csv(
        test_path
    )

    print(
        f"Test samples: {len(test_df)}"
    )

    # ==================================================
    # Create test dataset
    # ==================================================

    test_dataset = RSNAPneumoniaDataset(
        test_df
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # ==================================================
    # Create model
    # ==================================================

    model = PneumoniaCNN(
        num_classes=2
    ).to(device)

    # ==================================================
    # Load trained 20-epoch model
    # ==================================================

    model_path = (
        MODEL_DIR /
        "cnn_cam_baseline.pt"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}"
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
    # Generate predictions
    # ==================================================

    y_true = []
    y_pred = []

    print("\nRunning test evaluation...")

    with torch.no_grad():

        for x, y, _, _ in test_loader:

            x = x.to(device)

            # Forward pass
            logits = model(x)

            # Predicted class
            predictions = (
                logits
                .argmax(dim=1)
                .cpu()
                .numpy()
            )

            y_pred.extend(
                predictions.tolist()
            )

            y_true.extend(
                y.numpy().tolist()
            )

    # ==================================================
    # Calculate metrics
    # ==================================================

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    confusion = confusion_matrix(
        y_true,
        y_pred
    )

    # ==================================================
    # Print results
    # ==================================================

    print()
    print("=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print()
    print("Confusion Matrix:")
    print()

    print(confusion)

    print()
    print("Classification Report:")
    print()

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "No Pneumonia",
                "Pneumonia"
            ],
            zero_division=0
        )
    )

    # ==================================================
    # Save metrics
    # ==================================================

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    metrics_df = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],
        "Value": [
            accuracy,
            precision,
            recall,
            f1
        ]
    })

    metrics_path = (
        RESULT_DIR /
        "classification_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False
    )

    # ==================================================
    # Save confusion matrix
    # ==================================================

    confusion_df = pd.DataFrame(
        confusion,
        index=[
            "Actual No Pneumonia",
            "Actual Pneumonia"
        ],
        columns=[
            "Predicted No Pneumonia",
            "Predicted Pneumonia"
        ]
    )

    confusion_path = (
        RESULT_DIR /
        "confusion_matrix.csv"
    )

    confusion_df.to_csv(
        confusion_path
    )

    # ==================================================
    # Final information
    # ==================================================

    print()
    print("=" * 60)
    print("EVALUATION COMPLETED")
    print("=" * 60)

    print(
        f"Metrics saved to:\n"
        f"{metrics_path}"
    )

    print(
        f"Confusion matrix saved to:\n"
        f"{confusion_path}"
    )


if __name__ == "__main__":
    main()