import torch
import scipy

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
        loss = pred * torch.clamp(sdf, min=0) ** 2 + \
               (1 - pred) * torch.clamp(-sdf, min=0) ** 2

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
        loss = pred * torch.clamp(sdf, min=0) + \
               (1 - pred) * torch.clamp(-sdf, min=0)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # "none"
            return loss
        