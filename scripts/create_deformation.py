"""Create a named deformation collection for an artificial source dataset."""

from argparse import ArgumentParser
from pathlib import Path

from constraints.generators.factories import create_deformation_collection
from constraints.generators.types import (
    DeformationConfig,
    DeformationRejectionConfig,
)


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("name")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--scales", type=float, nargs="+", default=[14.0])
    parser.add_argument("--magnitude", type=float, default=7.0)
    parser.add_argument("--integrations", type=int, default=2)
    parser.add_argument("--voxsize", type=float, default=1.0)
    parser.add_argument(
        "--fractal-mode",
        choices=("blur", "upsample"),
        default="blur",
    )
    parser.add_argument("--minimum-jacobian", type=float, default=0.0)
    parser.add_argument("--minimum-foreground-margin", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="generation device; auto prefers CUDA, then MPS, then CPU",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scales = args.scales[0] if len(args.scales) == 1 else args.scales
    create_deformation_collection(
        args.source_root,
        args.name,
        DeformationConfig(
            scales=scales,
            magnitude=args.magnitude,
            integrations=args.integrations,
            voxsize=args.voxsize,
            fractal_mode=args.fractal_mode,
        ),
        DeformationRejectionConfig(
            minimum_jacobian=args.minimum_jacobian,
            minimum_foreground_margin_px=args.minimum_foreground_margin,
            max_attempts=args.max_attempts,
        ),
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
