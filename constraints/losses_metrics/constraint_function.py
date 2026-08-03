import numpy as np
import torch
from typing import cast
from scipy.ndimage import binary_dilation, generate_binary_structure, label
from ..datatools.datasets import ARTIFICIAL_MASK_LABEL_IDS


BG = ARTIFICIAL_MASK_LABEL_IDS["background"]
WALL = ARTIFICIAL_MASK_LABEL_IDS["boundary"]
LUMEN = ARTIFICIAL_MASK_LABEL_IDS["lumen"]
PLAQUE = ARTIFICIAL_MASK_LABEL_IDS["plaque"]


def _label_components(mask: np.ndarray, structure: np.ndarray) -> tuple[np.ndarray, int]:
    labeled, component_count = cast(tuple[np.ndarray, int], label(mask, structure=structure))
    return labeled, int(component_count)


def _as_single_label_map(prediction: torch.Tensor) -> np.ndarray:
    pred_np = prediction.detach().cpu().numpy()
    if pred_np.ndim == 3 and pred_np.shape[0] == 1:
        pred_np = pred_np[0]
    if pred_np.ndim != 2:
        raise ValueError(
            f"Expected prediction shape [H, W] or [1, H, W], got {tuple(pred_np.shape)}"
        )
    return pred_np


def does_violation_occur_with_wall(
    prediction: torch.Tensor, blob_threshold: int = 50, check_wall_integrity: bool = True
) -> tuple[bool, list[str]]:
    """Check Violations according to the following rules for 4-class vessel segmentation:
    
    Classes:
    - 0: Background (BG)
    - 1: Wall/boundary
    - 2: Lumen
    - 3: Plaque

    Violation if:
    - Rule 1 (Lumen Count): We do not have exactly one main connected component of Lumen
      with size >= blob_threshold.
    - Rule 2 (Wall Integrity): Background directly touches Lumen or Plaque (i.e. the Wall
      has a hole, tear, or missing segment that allows BG to leak inside).
    - Rule 3 (Plaque Placement): Plaque is floating inside Lumen without touching the Wall,
      or Plaque is embedded in Wall without touching the Lumen.
    - Rule 4 (Background Flow): Background component is enclosed inside the vessel/wall structure
      and does not touch the outer image edge.
    - Rule 5 (Wall Structure): Wall is split into more than one significant connected component
      (i.e. broken or fragmented ring).

    INPUT:
        - prediction: (H, W) or (1, H, W) tensor with values from ARTIFICIAL_MASK_LABEL_IDS
            representing the predicted segmentation mask
    - blob_threshold: minimum number of pixels required for a connected component to be considered a detection
    
    OUTPUT:
    - violation_occurred: bool indicating whether a violation occurred
    - violation_details: list of strings describing the specific violations that were found
    """
    violations = []

    # Move to CPU and convert to NumPy for morphological operations
    pred_np = _as_single_label_map(prediction)

    # Connectivity structures:
    # 8-connectivity for component grouping & spatial adjacency
    s8 = generate_binary_structure(2, 2)
    # 4-connectivity for background enclosure checks to avoid diagonal leakage
    s4 = generate_binary_structure(2, 1)

    bg_mask = pred_np == BG
    lumen_mask = pred_np == LUMEN
    plaque_mask = pred_np == PLAQUE
    wall_mask = pred_np == WALL

    # ---------------------------------------------------------
    # Rule 1: Exactly one main Lumen component
    # ---------------------------------------------------------
    lumen_labeled, num_lumen = _label_components(lumen_mask, structure=s8)
    significant_lumens = 0

    for i in range(1, num_lumen + 1):
        if np.sum(lumen_labeled == i) >= blob_threshold:
            significant_lumens += 1

    if significant_lumens != 1:
        violations.append(
            f"Lumen violation: Found {significant_lumens} main Lumen components (expected exactly 1)."
        )

    if check_wall_integrity:
        # ---------------------------------------------------------
        # Rule 2: Wall Continuity (BG must NOT touch Lumen or Plaque)
        # ---------------------------------------------------------
        bg_dilated = binary_dilation(bg_mask, structure=s8)

        if np.any(bg_dilated & lumen_mask):
            violations.append(
                "Wall integrity violation: Background touches Lumen directly (Wall gap/tear)."
            )

        if np.any(bg_dilated & plaque_mask):
            violations.append(
                "Wall integrity violation: Background touches Plaque directly (Wall gap/tear)."
            )

    # ---------------------------------------------------------
    # Rule 3: Plaque Adjacency Rules
    # ---------------------------------------------------------
    plaque_labeled, num_plaque = _label_components(plaque_mask, structure=s8)

    for i in range(1, num_plaque + 1):
        comp_mask = plaque_labeled == i
        if np.sum(comp_mask) < blob_threshold:
            continue

        dilated_comp = binary_dilation(comp_mask, structure=s8)

        touches_wall = np.any(dilated_comp & wall_mask)
        touches_lumen = np.any(dilated_comp & lumen_mask)

        if not touches_wall:
            violations.append(
                f"Plaque violation: Component {i} is floating in Lumen (does not touch Wall)."
            )
        if not touches_lumen:
            violations.append(
                f"Plaque violation: Component {i} does not touch Lumen (fully embedded in Wall or isolated)."
            )

    # ---------------------------------------------------------
    # Rule 4: Background must not be trapped inside
    # ---------------------------------------------------------
    bg_labeled, num_bg = _label_components(bg_mask, structure=s4)

    for i in range(1, num_bg + 1):
        comp_mask = bg_labeled == i

        touches_edge = (
            np.any(comp_mask[0, :])
            or np.any(comp_mask[-1, :])
            or np.any(comp_mask[:, 0])
            or np.any(comp_mask[:, -1])
        )

        if not touches_edge:
            violations.append(
                f"Background constraint violated: Background component {i} is enclosed and does not touch the image edge."
            )

    # ---------------------------------------------------------
    # Rule 5: Wall Component Integrity
    # ---------------------------------------------------------
    wall_labeled, num_wall = _label_components(wall_mask, structure=s8)
    significant_walls = 0
    for i in range(1, num_wall + 1):
        if np.sum(wall_labeled == i) >= blob_threshold:
            significant_walls += 1

    if significant_walls > 1:
        violations.append(
            f"Wall violation: Disconnected Wall components found ({significant_walls} main wall blobs)."
        )

    return len(violations) > 0, violations

def does_violation_occur_no_wall(
    prediction: torch.Tensor, blob_threshold: int = 50
) -> tuple[bool, list[str]]:
    BG = 0
    LUMEN = 1
    PLAQUE = 2
    """
    Check Violations according to the following rules:
    Violation if:
    - we have only one source of Lumen (i.e. one connected component of Lumen)
    - plaque has to have one edge connected to bg or be on the side of the image
        (i.e. it cannot be fully surrounded by lumen)
    - bg never flows in lumen or plaque (i.e. each connected component of bg has to be connected to the edge of the image)
    
    INPUT:
    - prediction: (H, W) tensor with values in {0, 1, 2} representing the predicted segmentation mask
    - blob_threshold: minimum number of pixels required for a connected component to be considered a detection of lumen
    OUTPUT:
    - violation_occurred: bool indicating whether a violation occurred
    - violation_details: list of strings describing the specific violations that were found
    """

    violations = []

    # Move to CPU and convert to NumPy for morphological operations
    pred_np = _as_single_label_map(prediction)

    # Use 8-connectivity (diagonals count as connected) to be strict about topology
    s = generate_binary_structure(2, 2)

    # ---------------------------------------------------------
    # Rule 1: One connected component of Lumen
    # ---------------------------------------------------------
    lumen_mask = pred_np == LUMEN
    lumen_detections, num_lumen = _label_components(lumen_mask, structure=s)

    # Note: Assuming the docstring meant "The rule is exactly one lumen; violation if not".

    if num_lumen != 1:
        lumen_pixel_counts = [
            np.sum(lumen_detections == i) for i in range(1, num_lumen + 1)
        ]
        one_over_threshold = False
        for i, count in enumerate(lumen_pixel_counts, start=1):
            if count > blob_threshold:
                if one_over_threshold:
                    violations.append(
                        f"Lumen violation: More than one connected component of Lumen with more than {blob_threshold} pixels (Component {i} has {count} pixels)."
                    )
                    break
                one_over_threshold = True

    # ---------------------------------------------------------
    # Rule 2: Plaque cannot be fully surrounded by lumen
    # ---------------------------------------------------------
    plaque_mask = pred_np == PLAQUE
    bg_mask = pred_np == BG
    plaque_labeled, num_plaque = _label_components(plaque_mask, structure=s)

    for i in range(1, num_plaque + 1):
        comp_mask = plaque_labeled == i
        comp_size = np.sum(comp_mask)
        if comp_size < blob_threshold:
            continue  # Ignore small plaque components

        # Check if the component touches the edge of the image
        touches_edge = (
            np.any(comp_mask[0, :])
            or np.any(comp_mask[-1, :])
            or np.any(comp_mask[:, 0])
            or np.any(comp_mask[:, -1])
        )

        if not touches_edge:
            # Dilate the component by 1 pixel to check its immediate neighbors
            dilated_comp = binary_dilation(comp_mask, structure=s)

            # Since it doesn't touch the edge, if it doesn't touch BG, it is surrounded by Lumen
            touches_bg = np.any(dilated_comp & bg_mask)
            if not touches_bg:
                violations.append(
                    f"Plaque constraint violated: Plaque component {i} is completely surrounded by Lumen."
                )

    # ---------------------------------------------------------
    # Rule 3: Background never flows in lumen or plaque
    # ---------------------------------------------------------
    bg_labeled, num_bg = _label_components(bg_mask, structure=s)

    for i in range(1, num_bg + 1):
        comp_mask = bg_labeled == i

        touches_edge = (
            np.any(comp_mask[0, :])
            or np.any(comp_mask[-1, :])
            or np.any(comp_mask[:, 0])
            or np.any(comp_mask[:, -1])
        )

        if not touches_edge:
            violations.append(
                f"Background constraint violated: Background component {i} is enclosed and does not touch the image edge."
            )

    return len(violations) > 0, violations


