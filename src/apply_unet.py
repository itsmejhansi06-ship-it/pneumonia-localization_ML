import os
import numpy as np
import pandas as pd
from PIL import Image
import torch

from src.models.unet import UNet


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 64

MODEL_PATH = "models/unet_lung.pt"

CSV_FILES = [
    "data/processed/train.csv",
    "data/processed/val.csv",
    "data/processed/test.csv"
]

IMAGE_ROOT = "data/raw/images"

MASK_OUTPUT = "data/processed/lung_masks"
SEGMENTED_OUTPUT = "data/processed/segmented_images"


# ============================================================
# LOAD U-NET
# ============================================================

def load_model():

    device = torch.device("cpu")

    model = UNet().to(device)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("U-Net loaded.")
    print(
        "Validation IoU:",
        checkpoint.get("val_iou")
    )

    return model, device


# ============================================================
# FIND IMAGE
# ============================================================

def find_image(patient_id):

    extensions = [
        ".jpg",
        ".jpeg",
        ".png"
    ]

    for ext in extensions:

        path = os.path.join(
            IMAGE_ROOT,
            patient_id + ext
        )

        if os.path.exists(path):
            return path

    return None


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(
    model,
    device,
    patient_id
):

    image_path = find_image(
        patient_id
    )

    if image_path is None:

        print(
            "Image not found:",
            patient_id
        )

        return False

    # --------------------------------------------------------
    # Load original X-ray
    # --------------------------------------------------------

    original = Image.open(
        image_path
    ).convert("L")

    original_array = np.array(
        original,
        dtype=np.float32
    )

    original_width, original_height = original.size

    # --------------------------------------------------------
    # Resize for U-Net
    # --------------------------------------------------------

    resized = original.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR
    )

    image = np.array(
        resized,
        dtype=np.float32
    ) / 255.0

    tensor = torch.from_numpy(
        image
    ).unsqueeze(0).unsqueeze(0)

    # --------------------------------------------------------
    # Predict lung mask
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            tensor.to(device)
        )

        probability = torch.sigmoid(
            output
        )

        predicted_mask = (
            probability > 0.5
        ).float()

    mask = (
        predicted_mask
        .squeeze()
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Convert mask to image
    # --------------------------------------------------------

    mask_image = Image.fromarray(
        (mask * 255).astype(np.uint8)
    )

    # Resize mask back to original X-ray size
    mask_image = mask_image.resize(
        (original_width, original_height),
        Image.Resampling.NEAREST
    )

    mask_array = (
        np.array(
            mask_image,
            dtype=np.float32
        ) / 255.0
    )

    # --------------------------------------------------------
    # Create segmented X-ray
    # --------------------------------------------------------

    segmented = (
        original_array
        * mask_array
    )

    segmented = np.clip(
        segmented,
        0,
        255
    ).astype(np.uint8)

    # --------------------------------------------------------
    # Save lung mask
    # --------------------------------------------------------

    mask_path = os.path.join(
        MASK_OUTPUT,
        patient_id + ".png"
    )

    Image.fromarray(
        (mask_array * 255).astype(np.uint8)
    ).save(
        mask_path
    )

    # --------------------------------------------------------
    # Save segmented X-ray
    # --------------------------------------------------------

    segmented_path = os.path.join(
        SEGMENTED_OUTPUT,
        patient_id + ".png"
    )

    Image.fromarray(
        segmented
    ).save(
        segmented_path
    )

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        MASK_OUTPUT,
        exist_ok=True
    )

    os.makedirs(
        SEGMENTED_OUTPUT,
        exist_ok=True
    )

    model, device = load_model()

    # --------------------------------------------------------
    # Collect patient IDs
    # --------------------------------------------------------

    patient_ids = set()

    for csv_path in CSV_FILES:

        print(
            "Reading:",
            csv_path
        )

        df = pd.read_csv(
            csv_path
        )

        patient_ids.update(
            df["patientId"]
            .astype(str)
            .tolist()
        )

    patient_ids = sorted(
        patient_ids
    )

    print()
    print(
        "Total unique images:",
        len(patient_ids)
    )
    print()

    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    successful = 0
    failed = 0

    for index, patient_id in enumerate(
        patient_ids,
        start=1
    ):

        success = process_image(
            model,
            device,
            patient_id
        )

        if success:
            successful += 1
        else:
            failed += 1

        # Progress
        if index % 100 == 0:

            print(
                f"Processed "
                f"{index}/{len(patient_ids)}"
            )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("================================")
    print("U-NET PROCESSING COMPLETE")
    print("================================")

    print(
        "Successful:",
        successful
    )

    print(
        "Failed:",
        failed
    )

    print(
        "Lung masks:",
        MASK_OUTPUT
    )

    print(
        "Segmented images:",
        SEGMENTED_OUTPUT
    )


if __name__ == "__main__":
    main()