"""
Test whether Kornia transform to sign distance function is differentiable.
"""
import json
import platform
from pathlib import Path
from typing import Any

import kornia
import matplotlib.pyplot as plt
import torch

from constraints.utils import (
	get_data_folder,
	get_experiment_folder,
	signed_distance_kornia_differentiable,
)

FOLDER = get_experiment_folder(Path("ex3") / "kornia_differentiable")
DATA = get_data_folder() / "artificial" / "custom"


def _build_soft_disk(
	size: int = 64,
	radius: float = 17.0,
	edge_width: float = 2.5,
	device: torch.device | None = None,
) -> torch.Tensor:
	coords = torch.linspace(-(size - 1) / 2, (size - 1) / 2, size, device=device)
	yy, xx = torch.meshgrid(coords, coords, indexing="ij")
	distance_from_center = torch.sqrt(xx.square() + yy.square())
	mask = torch.sigmoid((radius - distance_from_center) / edge_width)
	return mask.unsqueeze(0).unsqueeze(0).requires_grad_(True)


def _build_loss_weights(
	size: int,
	device: torch.device,
	dtype: torch.dtype,
) -> torch.Tensor:
	y_weights = torch.linspace(0.2, 1.0, size, device=device, dtype=dtype).view(
		1, 1, size, 1
	)
	x_weights = torch.linspace(1.0, 0.4, size, device=device, dtype=dtype).view(
		1, 1, 1, size
	)
	return y_weights * x_weights


def _tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
	detached = tensor.detach()
	return {
		"shape": list(detached.shape),
		"min": float(detached.min().item()),
		"max": float(detached.max().item()),
		"mean": float(detached.mean().item()),
		"norm": float(torch.linalg.vector_norm(detached).item()),
		"finite": bool(torch.isfinite(detached).all().item()),
		"nonzero_count": int((detached != 0).sum().item()),
	}


def _save_image(
	tensor: torch.Tensor,
	path: Path,
	title: str,
	cmap: str = "viridis",
) -> None:
	image = tensor.detach().cpu().squeeze().numpy()
	plt.figure(figsize=(5, 4))
	plt.imshow(image, cmap=cmap)
	plt.title(title)
	plt.colorbar(fraction=0.046, pad=0.04)
	plt.tight_layout()
	plt.savefig(path, dpi=160)
	plt.close()


def _compute_scalar_loss(mask: torch.Tensor) -> torch.Tensor:
	sdf = signed_distance_kornia_differentiable(mask)
	weights = _build_loss_weights(mask.shape[-1], mask.device, mask.dtype)
	return (sdf * weights).mean()


def _finite_difference_check(
	mask: torch.Tensor,
	direction: torch.Tensor,
	eps: float,
) -> float:
	with torch.no_grad():
		plus = (mask.detach() + eps * direction).clamp(0.0, 1.0)
		minus = (mask.detach() - eps * direction).clamp(0.0, 1.0)
	finite_difference = (_compute_scalar_loss(plus) - _compute_scalar_loss(minus)) / (
		2 * eps
	)
	return float(finite_difference.item())


def _run_check(device: torch.device) -> dict[str, Any]:
	torch.manual_seed(0)

	mask = _build_soft_disk(device=device)
	sdf = signed_distance_kornia_differentiable(mask)
	loss = _compute_scalar_loss(mask)
	loss.backward()

	grad = mask.grad
	assert grad is not None, "mask.grad is None after backward()"

	direction = torch.randn_like(mask)
	direction = direction / torch.linalg.vector_norm(direction)
	autograd_directional_derivative = float((grad * direction).sum().item())
	finite_difference = {
		str(eps): _finite_difference_check(mask, direction, eps)
		for eps in (1e-2, 1e-3, 1e-4)
	}

	prefix = device.type
	_save_image(
		mask,
		FOLDER / f"{prefix}_input_soft_disk.png",
		f"{prefix} input soft disk",
		cmap="gray",
	)
	_save_image(
		sdf,
		FOLDER / f"{prefix}_kornia_distance_transform.png",
		f"{prefix} Kornia distance transform",
	)
	_save_image(
		grad,
		FOLDER / f"{prefix}_input_gradient.png",
		f"{prefix} gradient d(loss)/d(input)",
		cmap="coolwarm",
	)
	_save_image(
		grad.abs(),
		FOLDER / f"{prefix}_input_gradient_abs.png",
		f"{prefix} absolute gradient",
		cmap="magma",
	)

	result = {
		"device": str(mask.device),
		"autograd": {
			"sdf_requires_grad": bool(sdf.requires_grad),
			"sdf_has_grad_fn": sdf.grad_fn is not None,
			"loss": float(loss.item()),
			"autograd_directional_derivative": autograd_directional_derivative,
			"finite_difference_directional_derivative": finite_difference,
		},
		"input_stats": _tensor_stats(mask),
		"sdf_stats": _tensor_stats(sdf),
		"gradient_stats": _tensor_stats(grad),
	}

	if not sdf.requires_grad or sdf.grad_fn is None:
		raise RuntimeError(
			"Kornia distance transform output is not attached to autograd graph "
			f"on {device}"
		)
	if not torch.isfinite(grad).all():
		raise RuntimeError(f"Gradient contains non-finite values on {device}")
	if torch.linalg.vector_norm(grad) == 0:
		raise RuntimeError(f"Gradient is exactly zero everywhere on {device}")

	return result


def main() -> None:
	FOLDER.mkdir(parents=True, exist_ok=True)

	devices = [torch.device("cpu")]
	if torch.cuda.is_available():
		devices.append(torch.device("cuda"))

	report = {
		"environment": {
			"python": platform.python_version(),
			"platform": platform.platform(),
			"torch": torch.__version__,
			"kornia": kornia.__version__,
			"cuda_available": torch.cuda.is_available(),
			"cuda_device_count": torch.cuda.device_count(),
			"cuda_device_name": torch.cuda.get_device_name(0)
			if torch.cuda.is_available()
			else None,
		},
		"checks": {},
		"output_folder": str(FOLDER),
	}

	for device in devices:
		report["checks"][device.type] = _run_check(device)

	report_path = FOLDER / "report.json"
	report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

	print(json.dumps(report, indent=2), flush=True)
	print(f"Saved images and report to: {FOLDER}", flush=True)


if __name__ == "__main__":
	main()
