from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap, to_rgba


def _as_torch_tensor(tensor: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(tensor, np.ndarray):
        if not tensor.flags.writeable:
            tensor = tensor.copy()

        return torch.from_numpy(tensor)

    return tensor


def to_label_map(x: torch.Tensor) -> torch.Tensor:
    """Convert logits/one-hot/labels to a batched integer label map [B, H, W]."""
    if x.ndim == 4:
        # [B, C, H, W] logits/one-hot or [B, 1, H, W] labels with channel dim.
        if x.shape[1] == 1:
            return x[:, 0].long()
        return x.argmax(dim=1).long()
    if x.ndim == 3:
        # Floating 3D tensors are treated as [C, H, W] logits/one-hot.
        # Integer 3D tensors are treated as [B, H, W] labels.
        if x.dtype.is_floating_point:
            return x.argmax(dim=0, keepdim=True).long()
        return x.long()
    if x.ndim == 2:
        return x.unsqueeze(0).long()
    raise ValueError(f"Unsupported tensor shape for label conversion: {tuple(x.shape)}")


def _default_palette(device: torch.device) -> torch.Tensor:
    # Distinct RGB colors in [0, 1]. Rows are class indices.
    return torch.tensor(
        [
            [0.0, 0.0, 0.0],  # background
            [0.90, 0.10, 0.10],  # red
            [0.10, 0.70, 0.10],  # green
            [0.10, 0.35, 0.95],  # blue
            [0.95, 0.65, 0.10],  # orange
            [0.75, 0.15, 0.75],  # magenta
            [0.10, 0.75, 0.75],  # cyan
            [0.90, 0.90, 0.20],  # yellow
            [0.55, 0.35, 0.20],  # brown
            [0.80, 0.80, 0.80],  # light gray
        ],
        dtype=torch.float32,
        device=device,
    )


def colorize_label_map(
    label_map: torch.Tensor, num_classes: int | None = None
) -> torch.Tensor:
    """Map integer labels [H, W] to RGB tensor [3, H, W]."""
    if label_map.ndim != 2:
        raise ValueError(
            f"Expected [H, W] label map, got shape {tuple(label_map.shape)}"
        )

    labels = label_map.long().clamp(min=0)
    inferred_classes = int(labels.max().item()) + 1
    class_count = max(inferred_classes, 1)
    if num_classes is not None:
        class_count = max(class_count, int(num_classes))

    palette = _default_palette(device=labels.device)
    if class_count > palette.shape[0]:
        repeats = int(np.ceil(class_count / palette.shape[0]))
        palette = palette.repeat(repeats, 1)

    rgb = palette[labels]  # [H, W, 3]
    return rgb.permute(2, 0, 1).contiguous()


def build_labels_triplet_image(
    gt_mask: torch.Tensor,
    warped_template: torch.Tensor,
    pred_mask_logits: torch.Tensor,
    sample_idx: int = 0,
    num_classes: int | None = None,
) -> torch.Tensor:
    """Return a side-by-side RGB composite: GT | warped | predicted as [3, H, 3W]."""
    gt_labels = to_label_map(gt_mask)
    warped_labels = to_label_map(warped_template)
    pred_labels = to_label_map(pred_mask_logits)

    batch_size = min(gt_labels.shape[0], warped_labels.shape[0], pred_labels.shape[0])
    if sample_idx < 0 or sample_idx >= batch_size:
        raise IndexError(
            f"sample_idx={sample_idx} out of range for effective batch size {batch_size}"
        )

    gt_rgb = colorize_label_map(gt_labels[sample_idx], num_classes=num_classes)
    warped_rgb = colorize_label_map(warped_labels[sample_idx], num_classes=num_classes)
    pred_rgb = colorize_label_map(pred_labels[sample_idx], num_classes=num_classes)

    separator = torch.ones(
        (3, gt_rgb.shape[1], 2), dtype=gt_rgb.dtype, device=gt_rgb.device
    )
    composite = torch.cat([gt_rgb, separator, warped_rgb, separator, pred_rgb], dim=2)
    return composite.detach().cpu().clamp(0.0, 1.0)


def show_torch_image(
    tensor: torch.Tensor | np.ndarray,
    title: str | None = None,
    cmap: str | None = None,
    save_path: Path | str | None = None,
):
    tensor = _as_torch_tensor(tensor)
    # 1. Prepare data safely
    image = tensor.detach().cpu().squeeze()

    if image.ndim == 3 and image.shape[0] in (1, 3, 4):
        image = image.permute(1, 2, 0)

    image = image.numpy()

    # 2. Use Explicit Object-Oriented API
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap=cmap)

    if title:
        ax.set_title(title)
    ax.axis("off")

    # Tight layout prevents cropped margins on remote rendering
    fig.tight_layout()

    # 3. Save the *entire figure* (including title/margins) if path exists
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    # 4. Explicitly display and close to free cluster node memory
    plt.show()
    plt.close(fig)


def show_torch_mask(
    mask: torch.Tensor | np.ndarray,
    title: str | None = None,
    sample_idx: int = 0,
    num_classes: int | None = None,
    save_path: Path | str | None = None,
):
    """Visualize a one-hot/logits/label mask with channel 0 as black background."""
    mask = _as_torch_tensor(mask)
    labels = to_label_map(mask)

    batch_size = labels.shape[0]
    if sample_idx < 0 or sample_idx >= batch_size:
        raise IndexError(
            f"sample_idx={sample_idx} out of range for effective batch size {batch_size}"
        )

    if num_classes is None and mask.ndim in (3, 4) and mask.dtype.is_floating_point:
        channel_dim = 0 if mask.ndim == 3 else 1
        num_classes = int(mask.shape[channel_dim])

    rgb_mask = colorize_label_map(labels[sample_idx], num_classes=num_classes)
    show_torch_image(rgb_mask, title=title, save_path=save_path)


def create_segmentation_overlay(
    image: torch.Tensor | np.ndarray,
    label_map: torch.Tensor | np.ndarray,
    cmap: ListedColormap | list | dict,
    alpha: float = 0.4,
    background_label: int = 0,
) -> np.ndarray:
    """Blends a grayscale image with a segmentation label map into an RGB image.

    Args:
        image: Grayscale tensor/array of shape (H, W) or (1, H, W), range
          [0, 1].
        label_map: Integer tensor/array of shape (H, W) containing class labels.
        cmap: Matplotlib ListedColormap, list/tuple of colors, or dict mapping
          label->color.
        alpha: Opacity factor for segmentation mask (0.0 to 1.0).
        background_label: Label index treated as transparent/unmasked background
          (set to None to color all labels).

    Returns:
        np.ndarray: Blended RGB float32 array of shape (H, W, 3) in range [0,
        1].
    """
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(label_map, torch.Tensor):
        label_map = label_map.detach().cpu().numpy()

    image = np.squeeze(image).astype(np.float32)
    label_map = np.squeeze(label_map).astype(np.int64)

    # Convert grayscale (H, W) to 3-channel RGB (H, W, 3)
    rgb_base = np.stack([image] * 3, axis=-1)
    rgb_base = np.clip(rgb_base, 0.0, 1.0)

    # Build color lookup table: shape (max_label + 1, 4) in RGBA
    max_label = int(label_map.max())
    if isinstance(cmap, dict):
        lut = np.zeros((max(max_label + 1, max(cmap.keys()) + 1), 4), dtype=np.float32)
        for lbl, color in cmap.items():
            lut[lbl] = to_rgba(color)
    elif isinstance(cmap, ListedColormap):
        colors = cmap.colors
        lut = np.array([to_rgba(c) for c in colors], dtype=np.float32)
    else:
        lut = np.array([to_rgba(c) for c in cmap], dtype=np.float32)

    # Map labels to RGB overlay
    overlay_rgb = lut[label_map, :3]

    # Create mask for pixels to blend (exclude background)
    if background_label is not None:
        mask = (label_map != background_label)[..., None]
    else:
        mask = np.ones((*label_map.shape, 1), dtype=bool)

    # Alpha blending on segmented regions
    blended = np.where(mask, (1.0 - alpha) * rgb_base + alpha * overlay_rgb, rgb_base)
    return np.clip(blended, 0.0, 1.0)
