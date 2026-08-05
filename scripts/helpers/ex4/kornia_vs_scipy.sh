#!/usr/bin/env bash
set -uo pipefail  # Keep testing combinations and report every failure.
seed=42
# seed=1234
# seed=219
MODES=(
    "BCE_OneSideSDFSquared"
    "BCE_OneSideSDFPlain"
    "BCE_DSDF_MSE"
    "BCE_SDFTEMPLATE_MSE"
    "BCE_SDFTEMPLATE_OneSideSDFSQUARE"
    "OneSideSDFSquared_OneSideSDFSquared"
    "OneSideSDFPlain_OneSideSDFPlain"
)
MODALITIES=("affine" "deformed")
SCRIPT="experiments/ex4/initial_decoupled.py"

FAILED=()

echo "=== SMOKE TEST: all mode x modality combinations ==="
for modality in "${MODALITIES[@]}"; do
    for mode in "${MODES[@]}"; do
        echo "--- smoke: modality=${modality} mode=${mode} ---"
        if ! uv run python "${SCRIPT}" \
            --modality "${modality}" \
            --mode "${mode}" \
            --batch_size 4 \
            --num_workers 0 \
            --seed "${seed}" \
            --smoke_test; then
            echo "!!! SMOKE TEST FAILED: modality=${modality} mode=${mode}"
            FAILED+=("${modality}/${mode}")
        fi
    done
done

if [ "${#FAILED[@]}" -ne 0 ]; then
    echo "=== SMOKE TESTS FAILED FOR: ==="
    printf '  %s\n' "${FAILED[@]}"
    echo "Fix these before launching the full sweep. Aborting."
    exit 1
fi

echo "=== All smoke tests passed. Launching full sweep. ==="
for modality in "${MODALITIES[@]}"; do
    for mode in "${MODES[@]}"; do
        echo "--- submit: modality=${modality} mode=${mode} ---"
        ./remote_submit.sh "${SCRIPT}" \
            --modality "${modality}" \
            --mode "${mode}" \
            --seed "${seed}" \
            --sdf_mode "kornia"
    done
done

echo "=== Sweep submission complete ==="
