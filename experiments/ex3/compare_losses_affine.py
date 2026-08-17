import argparse
import time
from pathlib import Path

import pandas as pd
import torch

from constraints import get_data_folder, get_experiment_folder
from constraints.datatools import CachedArtificialDataset
from constraints.datatools.datasets.types import Sample
from constraints.losses_metrics import (
    BlurredMSELoss,
    CentroidLoss,
    OneSideSDF,
    OneSideSDFSquare,
)
from constraints.transforms import differentiable_rigid

FOLDER = get_experiment_folder(Path("ex3") / "affine_losscomp")
DATA = get_data_folder() / "artificial" / "custom"
AFFINE_SMALL = DATA / "affine_small"
AFFINE_LARGE = DATA / "affine_large"
RESULTS_CSV = FOLDER / "violation_summary.csv"
RESULTS_MD = FOLDER / "violation_summary.md"

DATASET_SPECS = {
    "small_scipy": (AFFINE_SMALL, "scipy"),
    "large_scipy": (AFFINE_LARGE, "scipy"),
    "small_kornia": (AFFINE_SMALL, "kornia"),
    "large_kornia": (AFFINE_LARGE, "kornia"),
}


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def build_datasets(dataset_names: list[str]) -> dict:
    datasets = {}
    for name in dataset_names:
        folder, sdf_mode = DATASET_SPECS[name]
        log(f"Loading dataset '{name}' from {folder} (sdf_mode={sdf_mode})")
        datasets[name] = CachedArtificialDataset(folder, sdf_mode=sdf_mode)
        log(f"Loaded dataset '{name}' with {len(datasets[name])} samples")
    return datasets


class DirectOptimizer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._sigma = torch.nn.Parameter(torch.tensor(0.0))
        self._tx = torch.nn.Parameter(torch.tensor(0.0))
        self._ty = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, template):
        return differentiable_rigid(template, self._sigma, self._tx, self._ty)


LOSSES = {
    "OneSideSDFSquare": OneSideSDFSquare(),
    "CentroidLoss": CentroidLoss(),
    "OneSideSDF": OneSideSDF(),
    "BlurredMSELoss": BlurredMSELoss(),
    "MSELoss": torch.nn.MSELoss(),
}

LOSS_NEEDS_SDF = {
    "OneSideSDFSquare": True,
    "CentroidLoss": False,
    "OneSideSDF": True,
    "BlurredMSELoss": False,
    "MSELoss": False,
}

LOSS_EXPECTS_BATCH_DIM = {
    "OneSideSDFSquare": False,
    "CentroidLoss": True,
    "OneSideSDF": False,
    "BlurredMSELoss": True,
    "MSELoss": False,
}


def _maybe_add_batch_dim(x: torch.Tensor, add: bool) -> torch.Tensor:
    if add and x.dim() == 3:
        return x.unsqueeze(0)
    return x


def loss_fn_wrapper(
    loss_name: str,
    loss_fn: torch.nn.Module,
    needs_sdf: bool,
    sample: Sample,
    value: torch.Tensor,
):
    key = "sdf" if needs_sdf else "mask"
    add_batch_dim = LOSS_EXPECTS_BATCH_DIM[loss_name]
    pred = _maybe_add_batch_dim(value, add_batch_dim)
    target = _maybe_add_batch_dim(sample[key], add_batch_dim)
    return loss_fn(pred, target)


def _results_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[c]) for c in cols]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def save_results_table(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        return

    df = df.sort_values(by=["loss", "dataset"]).reset_index(drop=True)
    df.to_csv(RESULTS_CSV, index=False)

    df_md = df.copy()
    df_md["violation_rate"] = (df_md["violation_rate"] * 100).map(lambda x: f"{x:.2f}%")
    df_md["avg_final_mse"] = df_md["avg_final_mse"].map(lambda x: f"{x:.6f}")
    df_md["avg_final_loss"] = df_md["avg_final_loss"].map(lambda x: f"{x:.6f}")
    df_md["elapsed_sec"] = df_md["elapsed_sec"].map(lambda x: f"{x:.1f}")

    RESULTS_MD.write_text(_results_to_markdown(df_md), encoding="utf-8")


def _upsert_result(rows: list[dict], row: dict) -> list[dict]:
    updated = [
        r
        for r in rows
        if not (r["loss"] == row["loss"] and r["dataset"] == row["dataset"])
    ]
    updated.append(row)
    return updated


def load_existing_results() -> list[dict]:
    if not RESULTS_CSV.exists():
        return []

    try:
        rows = pd.read_csv(RESULTS_CSV).to_dict(orient="records")
        log(f"Loaded {len(rows)} existing rows from {RESULTS_CSV}")
        return rows
    except Exception as exc:
        log(f"Failed to read existing CSV ({exc}); starting fresh")
        return []


def _combo_key(loss_name: str, dataset_name: str) -> tuple[str, str]:
    return (loss_name, dataset_name)


def _completed_combo_keys(
    rows: list[dict],
    expected_num_runs: int,
    expected_num_iterations: int,
    expected_mse_threshold: float,
    expected_learning_rate: float,
) -> set[tuple[str, str]]:
    completed = set()
    for row in rows:
        if "loss" not in row or "dataset" not in row:
            continue

        # Only skip if row was produced with the same run configuration.
        if int(row.get("num_runs", -1)) != expected_num_runs:
            continue
        if int(row.get("num_iterations", -1)) != expected_num_iterations:
            continue
        if abs(float(row.get("mse_threshold", -1.0)) - expected_mse_threshold) > 1e-12:
            continue
        if abs(float(row.get("learning_rate", -1.0)) - expected_learning_rate) > 1e-12:
            continue

        completed.add(_combo_key(str(row["loss"]), str(row["dataset"])))
    return completed


def test_loss(
    loss_fn_name: str,
    dataset_name: str,
    dataset,
    max_runs: int = 100,
    num_iterations: int = 200,
    mse_threshold: float = 0.01,
    lr: float = 0.05,
    progress_every: int = 5,
) -> dict:
    start = time.time()
    violations = 0
    num_runs = min(max_runs, len(dataset))
    final_mse_sum = 0.0
    final_loss_sum = 0.0
    total_steps = num_runs * num_iterations
    log(
        f"Starting combo {loss_fn_name} x {dataset_name}: "
        f"{num_runs} samples x {num_iterations} iters = {total_steps} optimizer steps"
    )

    for k in range(num_runs):
        sample_start = time.time()
        sample = dataset[k]
        net = DirectOptimizer()
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        mse_calc = torch.nn.MSELoss()

        for _ in range(num_iterations):
            optimizer.zero_grad()
            transformed_template = net(sample["template"])
            loss = loss_fn_wrapper(
                loss_name=loss_fn_name,
                loss_fn=LOSSES[loss_fn_name],
                needs_sdf=LOSS_NEEDS_SDF[loss_fn_name],
                sample=sample,
                value=transformed_template,
            )
            mse_err = mse_calc(transformed_template, sample["mask"])
            loss.backward()
            optimizer.step()

        final_mse = float(mse_err.item())
        final_loss = float(loss.item())
        final_mse_sum += final_mse
        final_loss_sum += final_loss
        if final_mse > mse_threshold:
            violations += 1

        current = k + 1
        if current == 1 or current % progress_every == 0 or current == num_runs:
            elapsed = time.time() - start
            per_sample = elapsed / current
            eta = per_sample * (num_runs - current)
            log(
                f"  [{loss_fn_name} x {dataset_name}] sample {current}/{num_runs} "
                f"done in {time.time() - sample_start:.2f}s, "
                f"latest_mse={final_mse:.6f}, elapsed={elapsed:.1f}s, eta~{eta:.1f}s"
            )

    avg_final_mse = final_mse_sum / max(num_runs, 1)
    avg_final_loss = final_loss_sum / max(num_runs, 1)
    violation_portion = violations / max(num_runs, 1)
    elapsed = time.time() - start

    row = {
        "loss": loss_fn_name,
        "dataset": dataset_name,
        "violations": violations,
        "num_runs": num_runs,
        "violation_rate": violation_portion,
        "mse_threshold": mse_threshold,
        "num_iterations": num_iterations,
        "learning_rate": lr,
        "avg_final_mse": avg_final_mse,
        "avg_final_loss": avg_final_loss,
        "elapsed_sec": elapsed,
    }

    print(
        f"{loss_fn_name:>16} on {dataset_name:<12} "
        f"-> violations {violations}/{num_runs} ({violation_portion:.2%}), "
        f"avg_mse={avg_final_mse:.6f}, avg_loss={avg_final_loss:.6f}",
        flush=True,
    )
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare affine optimization violations across losses and datasets."
    )
    parser.add_argument(
        "--loss", nargs="*", default=None, help="Optional subset of loss names to run."
    )
    parser.add_argument(
        "--dataset",
        nargs="*",
        default=None,
        help="Optional subset of dataset names to run.",
    )
    parser.add_argument("--max-runs", type=int, default=50)
    parser.add_argument("--num-iterations", type=int, default=100)
    parser.add_argument("--mse-threshold", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5,
        help="Heartbeat period in samples within one combo.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing CSV by skipping completed loss/dataset combos.",
    )
    return parser.parse_args()


def _validate_subset(
    selected: list[str] | None, available: dict, label: str
) -> list[str]:
    if selected is None:
        return list(available.keys())

    missing = sorted(set(selected) - set(available.keys()))
    if missing:
        raise ValueError(
            f"Unknown {label}: {missing}. Available: {list(available.keys())}"
        )

    return selected


def main() -> None:
    args = parse_args()
    log(f"Output folder: {FOLDER}")
    log("Starting compare_losses_affine experiment")
    selected_losses = _validate_subset(args.loss, LOSSES, "losses")
    selected_datasets = _validate_subset(args.dataset, DATASET_SPECS, "datasets")

    datasets = build_datasets(selected_datasets)

    rows: list[dict] = load_existing_results() if args.resume else []
    completed_combo_keys = _completed_combo_keys(
        rows=rows,
        expected_num_runs=args.max_runs,
        expected_num_iterations=args.num_iterations,
        expected_mse_threshold=args.mse_threshold,
        expected_learning_rate=args.lr,
    )
    total = len(selected_losses) * len(selected_datasets)
    idx = 0

    for loss_name in selected_losses:
        for dataset_name in selected_datasets:
            idx += 1
            key = _combo_key(loss_name, dataset_name)
            if key in completed_combo_keys:
                log(
                    f"[{idx}/{total}] Skipping {loss_name} x {dataset_name} (already in CSV)"
                )
                continue

            log(f"[{idx}/{total}] Running {loss_name} x {dataset_name}")
            row = test_loss(
                loss_fn_name=loss_name,
                dataset_name=dataset_name,
                dataset=datasets[dataset_name],
                max_runs=args.max_runs,
                num_iterations=args.num_iterations,
                mse_threshold=args.mse_threshold,
                lr=args.lr,
                progress_every=args.progress_every,
            )
            rows = _upsert_result(rows, row)
            completed_combo_keys.add(key)
            save_results_table(rows)
            log(f"Saved intermediate results to: {RESULTS_CSV}")

    log(f"Finished. Final summary saved to: {RESULTS_CSV}")
    log(f"Markdown table saved to: {RESULTS_MD}")


if __name__ == "__main__":
    main()
