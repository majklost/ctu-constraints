#!/usr/bin/env bash
set -uo pipefail  # NOT -e: we want to keep going and report failures, not abort

LOSS_MODES=("sanityS" "sanityD" "naive" "fullSDF" "fullCE")
MODALITIES=("affine" "deformed")
SCRIPT="experiments/ex3/project_arch_initial.py"

FAILED=()

echo "=== SMOKE TEST: all loss_mode x modality combos ==="
for modality in "${MODALITIES[@]}"; do
  for loss_mode in "${LOSS_MODES[@]}"; do
    echo "--- smoke: modality=${modality} loss_mode=${loss_mode} ---"
    if ! python "${SCRIPT}" \
        --modality "${modality}" \
        --loss_mode "${loss_mode}" \
        --batch_size 4 \
        --num_workers 0 \
        --smoke_test; then
      echo "!!! SMOKE TEST FAILED: modality=${modality} loss_mode=${loss_mode}"
      FAILED+=("${modality}/${loss_mode}")
    fi
  done
done

if [ ${#FAILED[@]} -ne 0 ]; then
  echo "=== SMOKE TESTS FAILED FOR: ==="
  printf '  %s\n' "${FAILED[@]}"
  echo "Fix these before launching the real sweep. Aborting."
  exit 1
fi

echo "=== All smoke tests passed. Launching full sweep. ==="
for modality in "${MODALITIES[@]}"; do
  for loss_mode in "${LOSS_MODES[@]}"; do
    echo "--- full run: modality=${modality} loss_mode=${loss_mode} ---"
    ./remote_submit.sh "${SCRIPT}" \
      --modality "${modality}" \
      --loss_mode "${loss_mode}" 
  done
done

echo "=== Sweep complete ==="
