from abc import ABC, abstractmethod

import torch
from torch import nn

from ..types import LossInput, LossResult, WeightedLossTerm
from .loss_terms import LossTerm


def _interaction_logs(a: torch.Tensor, b: torch.Tensor, refs: list[torch.Tensor | None]):
    if not torch.is_grad_enabled():
        return None
    refs = [ref for ref in refs if ref is not None and ref.requires_grad]
    if not refs:
        return None

    def grads(loss: torch.Tensor):
        return (
            torch.autograd.grad(loss, refs, retain_graph=True, allow_unused=True)
            if loss.requires_grad
            else (None,) * len(refs)
        )

    def flat(values):
        return torch.cat(
            [
                (torch.zeros_like(ref) if value is None else value).reshape(-1)
                for value, ref in zip(values, refs, strict=True)
            ]
        )

    vec_a, vec_b = flat(grads(a)), flat(grads(b))
    norm_a, norm_b = torch.linalg.vector_norm(vec_a), torch.linalg.vector_norm(vec_b)
    eps = 1e-12
    return {
        "coupling/segmentation_grad_norm": norm_a,
        "coupling/registration_grad_norm": norm_b,
        "coupling/grad_ratio_segmentation_to_registration": norm_a / (norm_b + eps),
        "coupling/grad_cosine": torch.dot(vec_a, vec_b) / (norm_a * norm_b + eps),
        "coupling/segmentation_grad_share": norm_a / (norm_a + norm_b + eps),
    }


class LossComputer(nn.Module, ABC):
    """Base class for a structured scalar optimization objective."""

    @abstractmethod
    def compute(self, loss_input: LossInput) -> LossResult:
        """Compute the scalar total and weighted components."""

    def forward(self, loss_input: LossInput) -> torch.Tensor:
        result = self.compute(loss_input)
        if result.total.ndim != 0:
            raise ValueError(f"LossResult.total must be scalar, got {result.total.shape}")
        return result.total


class ProjectLossComputer(LossComputer):
    """Project loss-computer marker type used by Lightning wrappers."""

    @abstractmethod
    def compute(self, loss_input: LossInput) -> LossResult:
        """Implement the shared LossInput contract."""


class CompositeLossComputer(ProjectLossComputer):
    """Compose explicitly weighted loss terms into one objective."""

    def __init__(self, terms: list[WeightedLossTerm], *, grad_diagnostics=False, preset_name=None):
        super().__init__()
        if not terms:
            raise ValueError("CompositeLossComputer requires at least one loss term")
        if any(spec.weight < 0 for spec in terms):
            raise ValueError("Loss-term weights must be non-negative")
        self.weights = [float(spec.weight) for spec in terms]
        self.terms = nn.ModuleList(spec.term for spec in terms)
        self.grad_diagnostics = grad_diagnostics
        self.preset_name = preset_name

    def compute(self, loss_input: LossInput) -> LossResult:
        components, weighted_losses, logs = {}, [], {}
        for weight, term in zip(self.weights, self.terms, strict=True):
            if not isinstance(term, LossTerm):
                raise TypeError(f"Expected LossTerm, got {type(term)}")
            if term.name in components:
                raise ValueError(f"Duplicate loss component name: {term.name}")
            weighted_loss = weight * term(loss_input)
            components[term.name] = weighted_loss
            weighted_losses.append(weighted_loss)
            if self.grad_diagnostics:
                logs.update(term.logs(loss_input, weighted_loss) or {})
        total = sum(weighted_losses)
        if self.grad_diagnostics and len(weighted_losses) >= 2:
            logs.update(_interaction_logs(weighted_losses[0], weighted_losses[1], [loss_input.segmentation_logits]) or {})
        return LossResult(total=total, components=components, logs=logs or None)
