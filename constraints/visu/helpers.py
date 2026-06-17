from pathlib import Path

import matplotlib.pyplot as plt


def show_torch_image(tensor, title=None, cmap=None, save_path: Path | None = None):
    image = tensor.detach().cpu().squeeze()

    if image.ndim == 3 and image.shape[0] in (1, 3, 4):
        image = image.permute(1, 2, 0)

    image = image.numpy()

    plt.imshow(image, cmap=cmap)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()
    if save_path:
        plt.imsave(save_path, image, cmap=cmap)
