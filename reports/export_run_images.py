"""Export W&B validation overlays of selected runs into `report/images`.

For every run group below the script produces::

    report/images/<group>/<pretty-loss-label>/val_s0.png
    report/images/<group>/<pretty-loss-label>/val_s1.png

Each figure shows the input image, the GT label map, the segmentation
prediction and (when the run logged one) the registration-warped label map,
all as semi-transparent overlays on the image.

The overlays come from the W&B media logged under
``val/labels_overlay_val_s0`` / ``val/labels_overlay_val_s1``
(see `constraints/computers/metric_computers.py` and
`constraints/lightning_wrappers/modules.py`).  The last logged epoch is used.

Usage::

    python report/export_run_images.py                    # all groups
    python report/export_run_images.py --groups affine    # a single group
    python report/export_run_images.py --limit 2          # smoke test
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import wandb
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from PIL import Image

WANDB_ENTITY = "ksicht"
WANDB_PROJECT = "Constraints"

REPORT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = REPORT_DIR / "images"
CACHE_DIR = REPORT_DIR / ".wandb_media_cache"

RUN_GROUPS: dict[str, list[str]] = {
    "aff_def_calc": [
        "vk3apl02",
        "4vxhl05w",
        "wb2xwzn9",
        "lxdl3mu7",
        "4nqu3mui",
        "lbgji4zv",
        "ppldeco4",
        "mgg8qo91",
        "04z0atxs",
        "tmmwrkxs",
    ],
    "aff_def_deep": [
        "bntgp3wg",
        "6fet518o",
        "mm3bfw6q",
        "x1yo0k4l",
        "i1f46g1b",
        "e0ovj3fn",
        "ak6ycc71",
        "0eagi7f2",
        "2h2glhea",
        "lbdct475",
    ],
    "deformed": [
        "em4dijjw",
        "tfw4xn42",
        "3gpxcu6t",
        "pxrrdjll",
        "ej63m8js",
        "qxq4bvyy",
        "u77wm5lb",
        "11cw7gee",
        "ja77p42h",
        "yayf0obh",
        "qa18kium",
    ],
    "affine": [
        "it32ine1",
        "qmjah66l",
        "ia68rsh8",
        "lch6c6wo",
        "i1jsndpi",
        "7vrb8be8",
        "b7p9l4gv",
        "n482fjuk",
        "c3olvmv0",
        "ljs4tsfe",
        "myxyg3e0",
    ],
}

REG_LOSS_LABEL = {
    "UNET": "—",
    "BCE_BCE": "BCE (seg BCE)",
    "BCE_OneSideSDFPlain": "One-side SDF (seg BCE)",
    "BCE_OneSideSDFSquared": "One-side SDFSquared (seg BCE)",
    "BCE_CentroidLoss": "Centroid (seg BCE)",
    "BCE_BlurredLoss": "Blurred MSE (seg BCE)",
    "BCE_DSDF_MSE": "DSDF MSE (seg BCE)",
    "BCE_SDFTEMPLATE_MSE": "SDF-template MSE (seg BCE)",
    "BCE_SDFTEMPLATE_OneSideSDFSQUARE": "SDF-template one-side (seg BCE)",
    "OneSideSDFPlain_OneSideSDFPlain": "One-side SDF (seg One-side)",
    "OneSideSDFSquared_OneSideSDFSquared": "One-side SDFSquared (seg One-side)",
}

CLASS_NAMES = ["background", "wall", "lumen", "plaque"]
CLASS_COLORS = ["#000000", "#ff0000", "#09ff00", "#9500ff"]
CLASS_CMAP = ListedColormap(CLASS_COLORS)
CLASS_LEGEND = [
    Patch(facecolor=c, edgecolor="black", label=n)
    for c, n in zip(CLASS_COLORS, CLASS_NAMES, strict=True)
]

MASK_ALPHA = 0.45
# Painting class 0 would just darken the image, so background stays transparent.
BACKGROUND_ALPHA = 0.0

# W&B mask key -> subplot title.
MASK_PANELS = [
    ("ground_truth", "Ground truth"),
    ("predicted", "Segmentation"),
    ("warped", "Registration"),
]

SAMPLE_KEYS = ["val/labels_overlay_val_s0", "val/labels_overlay_val_s1"]


def sanitize(name: str) -> str:
    """Turn a display label into a filename, e.g. 'One-side SDF (seg BCE)'
    -> 'One-side-SDF-seg-BCE'."""
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "-", name).strip("-_.")
    return cleaned or "unnamed"


def last_overlay_entry(run, history_key: str) -> dict | None:
    """Return the media dict logged last under `history_key` (None if never)."""
    rows = run.history(keys=[history_key], pandas=False)
    for row in reversed(rows):
        value = row.get(history_key)
        if isinstance(value, dict) and value.get("filenames"):
            return value
    return None


def download_media(run, path: str) -> Path:
    """Download a run media file into the local cache and return its path."""
    target = CACHE_DIR / run.id / path
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        run.file(path).download(root=str(CACHE_DIR / run.id), exist_ok=True)
    return target


def load_image(path: Path) -> np.ndarray:
    array = np.asarray(Image.open(path))
    if array.ndim == 3 and array.shape[2] == 4:
        array = array[:, :, :3]
    return array


def show_overlay(ax, image: np.ndarray, mask: np.ndarray | None, title: str) -> None:
    if image.ndim == 2:
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    else:
        ax.imshow(image)
    if mask is not None:
        alpha = np.where(mask == 0, BACKGROUND_ALPHA, MASK_ALPHA)
        ax.imshow(
            np.clip(mask, 0, len(CLASS_COLORS) - 1),
            cmap=CLASS_CMAP,
            vmin=0,
            vmax=len(CLASS_COLORS) - 1,
            alpha=alpha,
            interpolation="nearest",
        )
    ax.set_title(title)
    ax.set_axis_off()


def export_sample(run, entry: dict, out_path: Path, figure_title: str) -> None:
    image = load_image(download_media(run, entry["filenames"][0]))
    all_masks = entry.get("all_masks") or [entry.get("masks") or {}]
    mask_specs = all_masks[0] or {}

    masks: dict[str, np.ndarray] = {}
    for mask_key, _ in MASK_PANELS:
        spec = mask_specs.get(mask_key)
        if not spec or not spec.get("path"):
            continue
        masks[mask_key] = np.asarray(Image.open(download_media(run, spec["path"])))

    panels = [(key, title) for key, title in MASK_PANELS if key in masks]
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(3.2 * (len(panels) + 1), 3.6))
    axes = np.atleast_1d(axes)

    show_overlay(axes[0], image, None, "Image")
    for ax, (mask_key, title) in zip(axes[1:], panels, strict=True):
        show_overlay(ax, image, masks[mask_key], title)

    fig.suptitle(figure_title, y=1.04)
    fig.legend(
        handles=CLASS_LEGEND,
        loc="lower center",
        ncol=len(CLASS_LEGEND),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def export_run(api, group: str, run_id: str, output_dir: Path) -> None:
    run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
    mode = str(run.config.get("mode", run.name))
    label = REG_LOSS_LABEL.get(mode, mode)
    # 'UNET' maps to an em dash, which is no filename at all.
    run_folder = sanitize(label)
    if run_folder == "unnamed":
        run_folder = sanitize(mode)
    run_dir = output_dir / sanitize(group) / run_folder

    print(f"[{group}] {run_id} mode={mode}")
    for history_key in SAMPLE_KEYS:
        entry = last_overlay_entry(run, history_key)
        if entry is None:
            print(f"  ! no media logged under {history_key}, skipping")
            continue
        sample_name = history_key.rsplit("/", 1)[-1].rsplit("_", 1)[-1]  # s0 / s1
        out_path = run_dir / f"val_{sample_name}.png"
        title = f"{group} | {mode}"
        if label not in (mode, "—"):
            title += f" | {label}"
        export_sample(run, entry, out_path, title)
        print(f"  -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=sorted(RUN_GROUPS),
        default=sorted(RUN_GROUPS),
        help="Groups to export (default: all).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Export at most N runs per group (for quick tests).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output root (default: {OUTPUT_DIR}).",
    )
    args = parser.parse_args()

    api = wandb.Api()
    for group in args.groups:
        run_ids = RUN_GROUPS[group]
        if args.limit is not None:
            run_ids = run_ids[: args.limit]
        for run_id in run_ids:
            try:
                export_run(api, group, run_id, args.output)
            except Exception as exc:  # keep going, report at the end of the line
                print(f"  ! failed for {run_id}: {exc}")


if __name__ == "__main__":
    main()
