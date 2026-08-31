import numpy as np
from scipy import ndimage


def is_annular(mask_2d: np.ndarray, min_hole_area: int = 10) -> bool:
    """
    Checks if a 2D binary mask is annular (b0 = 1, b1 = 1).

    Args:
        mask_2d: 2D numpy array (binary or boolean).
        min_hole_area: Minimum pixel area to filter out tiny noise holes.

    Returns:
        bool: True if the mask is a closed ring with one hole, False otherwise.
    """
    binary_mask = mask_2d > 0
    if not binary_mask.any():
        return False

    # 1. Count foreground components (b0) using 8-connectivity / square structure
    structure_8 = ndimage.generate_binary_structure(2, 2)
    _, num_components = ndimage.label(binary_mask, structure=structure_8)
    if num_components != 1:
        return False

    # 2. Extract and count enclosed holes (b1)
    filled_mask = ndimage.binary_fill_holes(binary_mask)
    holes_mask = filled_mask & ~binary_mask

    # Define hole connectivity (TEDS-Net paper uses 4-connectivity for background holes)
    structure_4 = ndimage.generate_binary_structure(2, 1)
    labeled_holes, num_holes = ndimage.label(holes_mask, structure=structure_4)

    if num_holes == 0:
        return False

    # 3. Filter out single-pixel / acquisition noise holes if needed
    if min_hole_area > 1:
        valid_holes = 0
        for i in range(1, num_holes + 1):
            if np.sum(labeled_holes == i) >= min_hole_area:
                valid_holes += 1
        return valid_holes == 1

    return num_holes == 1


