import torch
import torch.nn.functional as F

from ..visu.helpers import to_label_map


class RawMaskCrossEntropyLoss(torch.nn.Module):
    """Cross entropy that accepts this project's raw channel-mask targets."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self._cross_entropy = torch.nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, logits: torch.Tensor, raw_target: torch.Tensor) -> torch.Tensor:
        target = to_label_map(raw_target)
        return self._cross_entropy(logits, target)


# TODO: Make _Weighted loss from it
class CentroidLoss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_one_hot: torch.Tensor, gt_one_hot: torch.Tensor):
        # pred_one_hot and gt_one_hot shape: [B, C, H, W]
        device = pred_one_hot.device
        B, C, H, W = pred_one_hot.size()

        # Create coordinate matrices normalized between -1 and 1
        x_coords = (
            torch.linspace(-1, 1, W, device=device).view(1, 1, 1, W).expand(B, C, H, W)
        )
        y_coords = (
            torch.linspace(-1, 1, H, device=device).view(1, 1, H, 1).expand(B, C, H, W)
        )

        # Calculate soft centroids for each class channel
        def get_centroids(prob_map):
            eps = 1e-6
            # Sum mass across spatial dimensions
            mass = torch.sum(prob_map, dim=(2, 3), keepdim=True) + eps

            centroid_x = torch.sum(prob_map * x_coords, dim=(2, 3), keepdim=True) / mass
            centroid_y = torch.sum(prob_map * y_coords, dim=(2, 3), keepdim=True) / mass
            return torch.cat([centroid_x, centroid_y], dim=-1)  # [B, C, 1, 2]

        pred_centroids = get_centroids(pred_one_hot)
        gt_centroids = get_centroids(gt_one_hot)

        # MSE loss between the class center points establishes a global gradient pull
        loss = torch.nn.functional.mse_loss(
            pred_centroids, gt_centroids, reduction=self.reduction
        )
        return loss


class BlurredMSELoss(torch.nn.Module):
    """
    MSE loss with Gaussian-blurred ground truth:
        loss = MSE(pred, gaussian_blur(gt))
    Expected input shape: [B, C, H, W]
    """

    def __init__(
        self,
        kernel_size: int = 7,
        sigma: float = 1.5,
        reduction: str = "mean",
        padding_mode: str = "reflect",
    ) -> None:
        super().__init__()

        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(
                f"Invalid reduction: {reduction}. "
                "Expected one of {'none', 'mean', 'sum'}."
            )
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        if sigma <= 0:
            raise ValueError("sigma must be > 0.")
        if padding_mode not in {"reflect", "replicate", "constant", "circular"}:
            raise ValueError(
                f"Invalid padding_mode: {padding_mode}. "
                "Expected one of {'reflect', 'replicate', 'constant', 'circular'}."
            )

        self.kernel_size = kernel_size
        self.sigma = sigma
        self.reduction = reduction
        self.padding_mode = padding_mode

        self.register_buffer(
            "kernel2d",
            self._make_gaussian_kernel2d(kernel_size, sigma),
            persistent=False,
        )

    @staticmethod
    def _make_gaussian_kernel2d(kernel_size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, kernel_size, kernel_size)  # [1,1,K,K]

    def _blur(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(
                f"Expected 4D tensor [B,C,H,W], got shape {tuple(x.shape)}."
            )

        b, c, h, w = x.shape
        pad = self.kernel_size // 2

        kernel2d = self.kernel2d
        assert isinstance(kernel2d, torch.Tensor)
        kernel = kernel2d.to(device=x.device, dtype=x.dtype).expand(
            c, 1, -1, -1
        )  # [C,1,K,K]
        x = F.pad(x, (pad, pad, pad, pad), mode=self.padding_mode)
        return F.conv2d(x, kernel, groups=c)

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        if pred.shape != gt.shape:
            raise ValueError(
                f"`pred` and `gt` must have the same shape, got {tuple(pred.shape)} vs {tuple(gt.shape)}."
            )
        if not pred.is_floating_point():
            raise TypeError(
                f"`pred` must be a floating-point tensor, got {pred.dtype}."
            )

        # LabelSchema produces integer one-hot masks.  Convolution requires a
        # floating-point input (notably on CUDA), so match the prediction dtype
        # before blurring the target.
        gt_blurred = self._blur(gt.to(dtype=pred.dtype))
        return F.mse_loss(pred, gt_blurred, reduction=self.reduction)


class OneSideSDFSquare(torch.nn.Module):
    r"""
    Compute the one-sided signed distance field loss between predicted and
    ground truth masks.

    The loss is defined as:

    .. math::

        L_{\mathrm{pull}}
        =
        \frac{1}{N}
        \sum_{i=1}^{N}
        \sum_{c=1}^{C}
        \left[
            P_{i,c} \max(0, \mathrm{SDF}_{i,c})^2
            +
            (1-P_{i,c}) \max(0, -\mathrm{SDF}_{i,c})^2
        \right]

    where :math:`P_{i,c}` denotes the predicted probability for sample
    :math:`i` and class :math:`c`, and :math:`\mathrm{SDF}_{i,c}` is the
    signed distance field of the corresponding ground-truth mask.
    """

    def __init__(self, reduction="mean") -> None:
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(
                f"Invalid reduction: {reduction}. "
                "Expected one of {'none', 'mean', 'sum'}."
            )

        self.reduction = reduction

    def forward(self, pred, sdf):
        """
        Predicted probabilities `pred` and signed distance field `sdf` should have the same shape [B, C, H, W].
        """
        loss = (
            pred * torch.clamp(sdf, min=0) ** 2
            + (1 - pred) * torch.clamp(-sdf, min=0) ** 2
        )

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # "none"
            return loss


class OneSideSDF(torch.nn.Module):
    r"""
    Compute the one-sided signed distance field loss between predicted and
    ground truth masks.

    The loss is defined as:

    .. math::

        L_{\mathrm{pull}}
        =
        \frac{1}{N}
        \sum_{i=1}^{N}
        \sum_{c=1}^{C}
        \left[
            P_{i,c} \max(0, \mathrm{SDF}_{i,c})
            +
            (1-P_{i,c}) \max(0, -\mathrm{SDF}_{i,c})
        \right]

    where :math:`P_{i,c}` denotes the predicted probability for sample
    :math:`i` and class :math:`c`, and :math:`\mathrm{SDF}_{i,c}` is the
    signed distance field of the corresponding ground-truth mask.
    """

    def __init__(self, reduction="mean") -> None:
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(
                f"Invalid reduction: {reduction}. "
                "Expected one of {'none', 'mean', 'sum'}."
            )

        self.reduction = reduction

    def forward(self, pred, sdf):
        loss = pred * torch.clamp(sdf, min=0) + (1 - pred) * torch.clamp(-sdf, min=0)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # "none"
            return loss
