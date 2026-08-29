"""Create or validate every artifact required by a checked-in recipe."""

from argparse import ArgumentParser
from pathlib import Path

from constraints.generators.recipes import Recipe


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument("--sdf-batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recipe = Recipe.load_json(args.recipe)
    report = recipe.ensure(
        args.source_root,
        overwrite=args.overwrite,
        device=args.device,
        sdf_batch_size=args.sdf_batch_size,
        progress=True,
    )
    for label in report.created:
        print(f"Created {label}")
    for label in report.replaced:
        print(f"Replaced {label}")
    for label in report.reused:
        print(f"Reused {label}")
    if report.sdf_cache is not None:
        print(f"SDF cache: {report.sdf_cache}")


if __name__ == "__main__":
    main()
