from pathlib import Path

import matplotlib.pyplot as plt
import torch


def show_torch_image(
    tensor: torch.Tensor,
    title: str | None = None,
    cmap: str | None = None,
    save_path: Path | str | None = None,
):
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
