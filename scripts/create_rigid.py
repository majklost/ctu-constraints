"""Create a source-level or deformation-dependent rigid preset."""

from argparse import ArgumentParser
from pathlib import Path

from constraints.generators.factories import create_rigid_collection
from constraints.generators.types import (
    FloatRange,
    RigidConfig,
    RigidRejectionConfig,
)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("name")
    parser.add_argument("--deformation")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--angle-min", type=float, default=0.0)
    parser.add_argument("--angle-max", type=float, default=0.0)
    parser.add_argument("--dx-min", type=float, default=0.0)
    parser.add_argument("--dx-max", type=float, default=0.0)
    parser.add_argument("--dy-min", type=float, default=0.0)
    parser.add_argument("--dy-max", type=float, default=0.0)
    parser.add_argument("--minimum-foreground-margin", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=20)
    args = parser.parse_args()

    create_rigid_collection(
        args.source_root,
        args.name,
        RigidConfig(
            angle=FloatRange(args.angle_min, args.angle_max),
            dx=FloatRange(args.dx_min, args.dx_max),
            dy=FloatRange(args.dy_min, args.dy_max),
        ),
        RigidRejectionConfig(
            minimum_foreground_margin_px=args.minimum_foreground_margin,
            max_attempts=args.max_attempts,
        ),
        deformation=args.deformation,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
