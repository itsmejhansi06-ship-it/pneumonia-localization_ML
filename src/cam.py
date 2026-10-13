import numpy as np
import torch
import cv2


def generate_cam(
    model,
    image_tensor,
    class_index=1,
    device="cpu"
):
    """
    Generate Class Activation Map (CAM).

    Parameters
    ----------
    model : PneumoniaCNN
        Trained CNN model.

    image_tensor : torch.Tensor
        Image tensor of shape [1, 1, 128, 128].

    class_index : int
        0 = No Pneumonia
        1 = Pneumonia

    device : str
        cpu or cuda

    Returns
    -------
    cam : numpy array
        Normalized CAM in range [0, 1].
    """

    model.eval()

    image_tensor = image_tensor.to(device)

    with torch.no_grad():

        # ---------------------------------------------
        # Get CNN feature maps and classification logits
        # ---------------------------------------------

        logits, features = model(
            image_tensor,
            return_features=True
        )

        # ---------------------------------------------
        # Get FC weights for selected class
        # ---------------------------------------------

        weights = model.fc.weight[
            class_index
        ]

        # ---------------------------------------------
        # Remove batch dimension
        #
        # features:
        # [1, 128, H, W]
        #
        # becomes:
        # [128, H, W]
        # ---------------------------------------------

        feature_maps = features[0]

        # ---------------------------------------------
        # Weighted sum of feature maps
        # ---------------------------------------------

        cam = torch.sum(
            weights[:, None, None]
            * feature_maps,
            dim=0
        )

        # ---------------------------------------------
        # ReLU
        #
        # CAM should represent positive
        # contribution to the selected class.
        # ---------------------------------------------

        cam = torch.relu(cam)

        # ---------------------------------------------
        # Convert to NumPy
        # ---------------------------------------------

        cam = cam.cpu().numpy()

    # ---------------------------------------------
    # Normalize CAM to [0, 1]
    # ---------------------------------------------

    cam_min = cam.min()
    cam_max = cam.max()

    if cam_max > cam_min:

        cam = (
            cam - cam_min
        ) / (
            cam_max - cam_min
        )

    else:

        cam = np.zeros_like(cam)

    return cam


def resize_cam(
    cam,
    target_size
):
    """
    Resize CAM to image size.

    target_size:
        (width, height)
    """

    return cv2.resize(
        cam,
        target_size,
        interpolation=cv2.INTER_LINEAR
    )


def cam_to_heatmap(cam):
    """
    Convert normalized CAM into
    a color heatmap.
    """

    cam_uint8 = np.uint8(
        cam * 255
    )

    heatmap = cv2.applyColorMap(
        cam_uint8,
        cv2.COLORMAP_JET
    )

    return heatmap


def overlay_cam(
    image,
    cam,
    alpha=0.45
):
    """
    Overlay CAM heatmap on grayscale image.

    image:
        grayscale image in [0, 1]

    cam:
        normalized CAM in [0, 1]
    """

    image_uint8 = np.uint8(
        np.clip(image, 0, 1) * 255
    )

    image_color = cv2.cvtColor(
        image_uint8,
        cv2.COLOR_GRAY2BGR
    )

    heatmap = cam_to_heatmap(
        cam
    )

    overlay = cv2.addWeighted(
        image_color,
        1 - alpha,
        heatmap,
        alpha,
        0
    )

    return overlay