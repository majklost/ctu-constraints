import numpy as np
import torch


def create_one_hot_ellipses_masks(height, width, ellipses_list) -> np.ndarray:
    """
    ellipses_list: list of dicts with 'center' (h,k), 'axes' (a,b), 'class_id'
    """
    num_classes = max(e["class_id"] for e in ellipses_list) + 1
    masks = np.zeros((num_classes, height, width), dtype=np.uint8)

    # Create coordinate grids
    y, x = np.ogrid[:height, :width]

    for e in ellipses_list:
        h, k = e["center"]
        a, b = e["axes"]
        cls = e["class_id"]
        angle = e.get("angle", 0)  # Default angle is 0 if not provided

        transform_x = (x - h) * np.cos(angle) + (y - k) * np.sin(angle)
        transform_y = -(x - h) * np.sin(angle) + (y - k) * np.cos(angle)
        # Ellipse inequality calculation
        ellipse_mask = (transform_x**2 / a**2 + transform_y**2 / b**2) <= 1

        # Add to the specific class channel
        masks[cls] = np.logical_or(masks[cls], ellipse_mask)
        # Ensure the mask is binary (0 or 1)
        masks[cls] = masks[cls].astype(np.uint8)
        # zero out the overlapping areas in other classes
        for other_cls in range(num_classes):
            if other_cls != cls:
                masks[other_cls] = np.logical_and(
                    masks[other_cls], ~ellipse_mask
                ).astype(np.uint8)

    return np.transpose(masks, (1, 2, 0)) * 255


def get_standard_mask() -> torch.Tensor:
    return (
        torch.tensor(
            create_one_hot_ellipses_masks(
                256,
                256,
                [
                    {
                        "center": (128, 128),
                        "axes": (78, 78),
                        "class_id": 0,
                    },  # Red ellipse
                    {
                        "center": (128, 128),
                        "axes": (73, 73),
                        "class_id": 1,
                    },  # Green ellipse
                    {
                        "center": (95, 65),
                        "axes": (5, 28),
                        "class_id": 0,
                        "angle": np.pi / 3,
                    },  # RED TO CONNECT
                    {
                        "center": (95, 70),
                        "axes": (5, 30),
                        "class_id": 2,
                        "angle": np.pi / 3,
                    },  # Blue ellipse
                    {
                        "center": (191, 128),
                        "axes": (10, 28),
                        "class_id": 2,
                    },  # Blue ellipse
                ],
            )
        )
        .permute(2, 0, 1)
        .float()
        / 255.0
    )
