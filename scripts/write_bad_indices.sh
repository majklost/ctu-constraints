#!/usr/bin/env bash
set -euo pipefail

for dataset in rigid rigid_deformed deformed; do
	for split in trn val; do
		python scripts/write_bad_indices.py "data/artificial/$dataset/$split"
	done
done
