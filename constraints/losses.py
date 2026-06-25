import torch


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


class SDF(torch.nn.Module):
    pass
