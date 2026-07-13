import torch
import torch.nn as nn
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
        translation = torch.tanh(translation) * self.max_translation  # scale translation to
        return angle, translation
