import cv2
import numpy as np


def threshold_cam(
    cam,
    threshold=0.50
):
    """
    Convert CAM into a binary mask.

    Pixels above the threshold are considered
    potential pneumonia regions.
    """

    mask = (
        cam >= threshold
    ).astype(np.uint8) * 255

    return mask


def find_connected_components(
    binary_mask
):
    """
    Find connected regions in the CAM mask.
    """

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary_mask,
            connectivity=8
        )
    )

    components = []

    for label in range(
        1,
        num_labels
    ):

        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]

        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]

        area = stats[label, cv2.CC_STAT_AREA]

        components.append(
            {
                "x": int(x),
                "y": int(y),
                "width": int(width),
                "height": int(height),
                "area": int(area)
            }
        )

    return components


def select_bounding_box(
    components,
    min_area=10
):
    """
    Select the largest meaningful connected component.
    """

    valid_components = [
        component
        for component in components
        if component["area"] >= min_area
    ]

    if not valid_components:
        return None

    # Select largest activated region
    best = max(
        valid_components,
        key=lambda x: x["area"]
    )

    x = best["x"]
    y = best["y"]

    width = best["width"]
    height = best["height"]

    return (
        x,
        y,
        x + width,
        y + height
    )


def extract_bounding_box(
    cam,
    threshold=0.50,
    min_area=10
):
    """
    Complete CAM → threshold → connected components
    → predicted bounding box pipeline.
    """

    binary_mask = threshold_cam(
        cam,
        threshold
    )

    components = find_connected_components(
        binary_mask
    )

    box = select_bounding_box(
        components,
        min_area
    )

    return box, binary_mask


def draw_bounding_box(
    image,
    box,
    color=(0, 255, 0),
    thickness=2
):
    """
    Draw predicted bounding box.

    image must be a BGR image.
    """

    output = image.copy()

    if box is None:
        return output

    x1, y1, x2, y2 = box

    cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        color,
        thickness
    )

    return output


def draw_ground_truth_boxes(
    image,
    boxes,
    color=(255, 0, 0),
    thickness=2
):
    """
    Draw one or more ground-truth boxes.

    boxes are expected as:
        [(x, y, w, h), ...]
    """

    output = image.copy()

    for box in boxes:

        x, y, w, h = box

        x1 = int(x)
        y1 = int(y)

        x2 = int(x + w)
        y2 = int(y + h)

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            thickness
        )

    return output