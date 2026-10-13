import os
import numpy as np
from PIL import Image
import torch

from src.models.unet import UNet


# ============================================================
# CONFIG
# ============================================================

IMAGE_SIZE = 64

MODEL_PATH = "models/unet_lung.pt"

IMAGE_ID = "01c0b605-0a82-401b-9649-81252304ac45"

IMAGE_ROOT = "data/raw/images"

OUTPUT_DIR = "results/unet_test"


# ============================================================
# MAIN
# ============================================================

def main():

    device = torch.device("cpu")

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = UNet().to(device)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        "Loaded U-Net"
    )

    print(
        "Saved validation IoU:",
        checkpoint.get("val_iou")
    )

    # --------------------------------------------------------
    # Find image
    # --------------------------------------------------------

    image_path = os.path.join(
        IMAGE_ROOT,
        IMAGE_ID + ".jpg"
    )

    if not os.path.exists(image_path):

        # Try JPEG
        image_path = os.path.join(
            IMAGE_ROOT,
            IMAGE_ID + ".jpeg"
        )

    if not os.path.exists(image_path):

        # Try PNG
        image_path = os.path.join(
            IMAGE_ROOT,
            IMAGE_ID + ".png"
        )

    if not os.path.exists(image_path):

        raise FileNotFoundError(
            f"Could not find image for {IMAGE_ID}"
        )

    print(
        "Image:",
        image_path
    )

    # --------------------------------------------------------
    # Load original
    # --------------------------------------------------------

    original = Image.open(
        image_path
    ).convert("L")

    original_array = np.array(
        original,
        dtype=np.float32
    )

    print(
        "Original image size:",
        original.size
    )

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

    mask = predicted_mask.squeeze().numpy()

    # --------------------------------------------------------
    # Resize mask back to original size
    # --------------------------------------------------------

    mask_image = Image.fromarray(
        (mask * 255).astype(np.uint8)
    )

    mask_image = mask_image.resize(
        original.size,
        Image.Resampling.NEAREST
    )

    mask_array = (
        np.array(
            mask_image,
            dtype=np.float32
        ) / 255.0
    )

    # --------------------------------------------------------
    # Create segmented image
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
    # Save files
    # --------------------------------------------------------

    original_output = os.path.join(
        OUTPUT_DIR,
        "original.png"
    )

    mask_output = os.path.join(
        OUTPUT_DIR,
        "lung_mask.png"
    )

    segmented_output = os.path.join(
        OUTPUT_DIR,
        "segmented.png"
    )

    original.save(
        original_output
    )

    Image.fromarray(
        (mask_array * 255).astype(np.uint8)
    ).save(
        mask_output
    )

    Image.fromarray(
        segmented
    ).save(
        segmented_output
    )

    print()
    print("================================")
    print("U-NET TEST COMPLETE")
    print("================================")

    print(
        "Original:",
        original_output
    )

    print(
        "Lung mask:",
        mask_output
    )

    print(
        "Segmented:",
        segmented_output
    )


if __name__ == "__main__":
    main()