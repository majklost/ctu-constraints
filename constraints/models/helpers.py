import torch
import torch.nn as nn

from ..types import RigidParams, TransformSpec

"""
Building blocks for architectures
"""


class RigidTransformHead(nn.Module):
    """
    Given feature vectors, predict rigid transformation parameters (angle and translation).
    The angle is predicted as a 2D vector (sin, cos) to avoid discontinuities, and then converted to an angle using atan2.
    The translation is predicted as a 2D vector and scaled to a maximum translation value.
    """

    def __init__(self, max_translation, hidden=256):
        super().__init__()
        self.max_translation = max_translation
        self.mlp = nn.Sequential(
            nn.LazyLinear(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
        )
        # predict sin/cos instead of angle directly (see below)
        self.angle_head = nn.Linear(hidden // 2, 2)
        self.translation_head = nn.Linear(hidden // 2, 2)

    def forward(self, x):
        feat = self.mlp(x)
        sin_cos = self.angle_head(feat)
        sin_cos = sin_cos / sin_cos.norm(dim=1, keepdim=True).clamp(min=1e-6)
        translation = self.translation_head(feat)
        angle = torch.atan2(sin_cos[:, 0], sin_cos[:, 1]).view(-1, 1)
        translation = (
            torch.tanh(translation) * self.max_translation
        )  # scale translation to
        return angle, translation


def rigid_matrix_to_grid_params(R, t, H, W):
    """
    R: (B, 2, 2) rotation mapping source -> template in pixel space.
    t: (B, 2) translation mapping source -> template in pixel space (x, y).

    Returns angle (rad), dx, dy in PyTorch affine_grid convention [-1, 1]
    (align_corners=False), such that warping the template with them places it
    into the source frame.

    Assumes H == W: `affine_grid` applies the linear part in normalized
    coordinates, so for non-square images a pixel-space rotation is no longer a
    rotation there and cannot be expressed as a single angle.
    """
    # The moments/Kabsch result maps source -> template: p_tmpl = R @ p_src + t.
    # affine_grid is a *backward* map: for every output pixel (source frame) it
    # says where to sample the input (template). That is exactly the mapping we
    # already have, so A = R and b = t -- no inversion needed.
    #
    # The only subtlety is the origin: R and t are expressed about the pixel
    # corner (0, 0), while affine_grid rotates about the image centre c.
    # Writing p = p_c + c gives
    #     p_tmpl_c = R @ p_src_c + (R @ c - c + t)
    # so the centre-relative translation carries the extra (R - I) @ c term.

    # R = [[cos(a), -sin(a)], [sin(a), cos(a)]]
    angle = torch.atan2(R[:, 1, 0], R[:, 0, 0])

    # Image centre in pixel coordinates (align_corners=False).
    center = torch.tensor(
        [(W - 1) / 2.0, (H - 1) / 2.0], device=R.device, dtype=R.dtype
    )
    center = center.expand(R.shape[0], 2)  # (B, 2)

    # Centre-relative translation, still in pixels.
    b_pixel = torch.bmm(R, center.unsqueeze(2)).squeeze(2) - center + t  # (B, 2)

    # Convert to normalized grid space [-1, 1]: a shift of n pixels is 2n/W.
    scale = torch.tensor([2.0 / W, 2.0 / H], device=R.device, dtype=R.dtype)
    b = b_pixel * scale

    dx = b[:, 0]
    dy = b[:, 1]

    return angle, dx, dy


class MomentsAffineAlignment(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_centroids(tensor: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # Create coordinate grid matching (x, y)
        grid_y, grid_x = torch.meshgrid(
            torch.arange(tensor.shape[2], device=tensor.device, dtype=tensor.dtype),
            torch.arange(tensor.shape[3], device=tensor.device, dtype=tensor.dtype),
            indexing="ij",
        )
        cutout_tensor = tensor[:, 1:, :, :]  # drop background channel
        denom = cutout_tensor.sum(dim=(2, 3)) + eps  # (B, C)

        num_x = (grid_x.unsqueeze(0).unsqueeze(0) * cutout_tensor).sum(dim=(2, 3))
        num_y = (grid_y.unsqueeze(0).unsqueeze(0) * cutout_tensor).sum(dim=(2, 3))

        centroids = torch.stack([num_x / denom, num_y / denom], dim=-1)  # (B, C, 2)
        return centroids

    def forward(self, source: torch.Tensor, template: torch.Tensor) -> TransformSpec:
        if len(template.shape) == 3:
            template = template.unsqueeze(0)
            template = template.expand(source.shape[0], -1, -1, -1)

        centroids_source = self.get_centroids(source)  # (B, C, 2)
        centroids_template = self.get_centroids(template)  # (B, C, 2)

        mean_src = centroids_source.mean(dim=1)  # (B, 2)
        mean_tmpl = centroids_template.mean(dim=1)  # (B, 2)

        centered_src = centroids_source - mean_src.unsqueeze(1)
        centered_tmpl = centroids_template - mean_tmpl.unsqueeze(1)

        # Kabsch Algorithm: Cross-covariance matrix H = Src^T * Tmpl
        H = torch.bmm(centered_src.transpose(1, 2), centered_tmpl)  # (B, 2, 2)

        u, s, vh = torch.linalg.svd(H)
        v = vh.transpose(-2, -1)

        # With H = Src^T * Tmpl = U * S * V^T, the rotation mapping src -> tmpl
        # (as column vectors, p_tmpl = R @ p_src) is R = V * U^T.
        # To handle reflection: R = V * diag(1, det(V*U^T)) * U^T
        d = torch.det(torch.bmm(v, u.transpose(1, 2)))

        # Construct correction matrix for reflection handling
        e = torch.ones_like(v)
        e[:, :, 1] = d.unsqueeze(-1)

        R = torch.bmm(v * e, u.transpose(1, 2))

        # t = mean_tmpl - R * mean_src
        t = mean_tmpl - torch.bmm(R, mean_src.unsqueeze(2)).squeeze(2)
        angle, dx, dy = rigid_matrix_to_grid_params(
            R, t, source.shape[2], source.shape[3]
        )
        return TransformSpec(rigid=RigidParams(angle, dx, dy))
