"""Check training and validation CSV files for overlapping sample indices."""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from constraints import get_data_folder


def read_indices(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    if "sample_index" not in frame.columns:
        raise ValueError(f"{path} has no 'sample_index' column")
    return frame["sample_index"]


def main(train_path: Path, validation_path: Path) -> None:
    train = read_indices(train_path)
    validation = read_indices(validation_path)
    overlap = sorted(set(train) & set(validation))

    print(f"Training:   {len(train)} rows, {train.nunique()} unique IDs")
    print(f"Validation: {len(validation)} rows, {validation.nunique()} unique IDs")
    print(f"Training duplicates:   {int(train.duplicated().sum())}")
    print(f"Validation duplicates: {int(validation.duplicated().sum())}")
    print(f"Overlap: {len(overlap)} IDs")
    if overlap:
        print(overlap)
    else:
        print("OK: training and validation splits are disjoint")


if __name__ == "__main__":
    split_folder = get_data_folder() / "artificial/samples5000/splits"
    parser = ArgumentParser()
    parser.add_argument(
        "--train",
        type=Path,
        default=split_folder / "trn_samples.csv",
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=split_folder / "val_samples.csv",
    )
    args = parser.parse_args()
    main(args.train, args.validation)
