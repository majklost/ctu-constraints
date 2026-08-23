"""Write bad_indices.csv for an existing artificial dataset cache."""

from argparse import ArgumentParser
from pathlib import Path

from constraints.datatools.datasets import BAD_INDICES_FILENAME, write_bad_indices


def main() -> None:
    parser = ArgumentParser(
        description="Validate cached artificial masks and record invalid source indices."
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Directory containing mask.npy, for example data/artificial/rigid_deformed/trn",
    )
    parser.add_argument(
        "--check-wall-integrity",
        action="store_true",
        help="Also reject masks where background touches lumen or plaque.",
    )
    args = parser.parse_args()

    mask_path = args.dataset_dir / "mask.npy"
    if not mask_path.is_file():
        parser.error(f"Dataset mask file does not exist: {mask_path}")

    bad_indices = write_bad_indices(
        args.dataset_dir, check_wall_integrity=args.check_wall_integrity
    )
    output_path = args.dataset_dir / BAD_INDICES_FILENAME
    print(f"Wrote {len(bad_indices)} invalid indices to {output_path}")


if __name__ == "__main__":
    main()
