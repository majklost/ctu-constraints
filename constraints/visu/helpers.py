from pathlib import Path

import matplotlib.pyplot as plt
import torch
import numpy as np


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
            [0.0, 0.0, 0.0],      # background
            [0.90, 0.10, 0.10],   # red
            [0.10, 0.70, 0.10],   # green
            [0.10, 0.35, 0.95],   # blue
            [0.95, 0.65, 0.10],   # orange
            [0.75, 0.15, 0.75],   # magenta
            [0.10, 0.75, 0.75],   # cyan
            [0.90, 0.90, 0.20],   # yellow
            [0.55, 0.35, 0.20],   # brown
            [0.80, 0.80, 0.80],   # light gray
        ],
        dtype=torch.float32,
        device=device,
    )


def colorize_label_map(label_map: torch.Tensor, num_classes: int | None = None) -> torch.Tensor:
    """Map integer labels [H, W] to RGB tensor [3, H, W]."""
    if label_map.ndim != 2:
        raise ValueError(f"Expected [H, W] label map, got shape {tuple(label_map.shape)}")

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

    separator = torch.ones((3, gt_rgb.shape[1], 2), dtype=gt_rgb.dtype, device=gt_rgb.device)
    composite = torch.cat([gt_rgb, separator, warped_rgb, separator, pred_rgb], dim=2)
    return composite.detach().cpu().clamp(0.0, 1.0)


def show_torch_image(
    tensor: torch.Tensor,
    title: str | None = None,
    cmap: str | None = None,
    save_path: Path | str | None = None,
):
    #if numpy array convert to torch tensor
    if isinstance(tensor, np.ndarray):
        if not tensor.flags.writeable:
            tensor = tensor.copy()

        tensor = torch.from_numpy(tensor)
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
