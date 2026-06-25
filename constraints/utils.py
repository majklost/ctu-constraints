import os
from pathlib import Path

import numpy as np


def get_repo_root() -> Path:
    """Traverses upwards to find the repository root marked by a .git or .mutagen folder."""
    current = Path.cwd().resolve()
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists() or (parent / ".mutagen").exists():
            return parent
    return current  # Fallback to cwd if marker not found


# Define your base paths relative to the repo root
REPO_ROOT = get_repo_root()


def rad2deg(rad):
    return rad * 180 / np.pi


def deg2rad(deg):
    return deg * np.pi / 180
