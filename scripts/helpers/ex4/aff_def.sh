#!/usr/bin/env bash
set -uo pipefail  # Keep testing combinations and report every failure.
seed=42
# seed=1234
# seed=219
MODES=(
    "BCE_OneSideSDFSquared"
    "BCE_OneSideSDFPlain"
    "BCE_BCE"
    "BCE_CentroidLoss"
    "BCE_BlurredLoss"
    "BCE_DSDF_MSE"
    "BCE_SDFTEMPLATE_MSE"
    "BCE_SDFTEMPLATE_OneSideSDFSQUARE"
    "OneSideSDFSquared_OneSideSDFSquared"
    "OneSideSDFPlain_OneSideSDFPlain"
)
RIGID_DEF_MODES=("calc" "deep")
MODALITY="both"
SCRIPT="experiments/ex4/initial_decoupled.py"
# UNET never touches the registration net, so it is run once rather than once per
# rigid_def mode. --rigid_def_mode is still passed because --modality both requires it.
UNET_RIGID_DEF_MODE="calc"

FAILED=()

echo "=== SMOKE TEST: all rigid_def_mode x mode combinations ==="
echo "--- smoke: modality=${MODALITY} mode=UNET ---"
if ! uv run python "${SCRIPT}" \
    --modality "${MODALITY}" \
    --mode "UNET" \
    --rigid_def_mode "${UNET_RIGID_DEF_MODE}" \
    --batch_size 4 \
    --num_workers 0 \
    --seed "${seed}" \
    --smoke_test; then
    echo "!!! SMOKE TEST FAILED: modality=${MODALITY} mode=UNET"
    FAILED+=("${MODALITY}/UNET")
fi

for rigid_def_mode in "${RIGID_DEF_MODES[@]}"; do
    for mode in "${MODES[@]}"; do
        echo "--- smoke: modality=${MODALITY} rigid_def=${rigid_def_mode} mode=${mode} ---"
        if ! uv run python "${SCRIPT}" \
            --modality "${MODALITY}" \
            --rigid_def_mode "${rigid_def_mode}" \
            --mode "${mode}" \
            --batch_size 4 \
            --num_workers 0 \
            --seed "${seed}" \
            --smoke_test; then
            echo "!!! SMOKE TEST FAILED: rigid_def=${rigid_def_mode} mode=${mode}"
            FAILED+=("${rigid_def_mode}/${mode}")
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
echo "--- submit: modality=${MODALITY} mode=UNET ---"
./remote_submit.sh "${SCRIPT}" \
    --modality "${MODALITY}" \
    --mode "UNET" \
    --rigid_def_mode "${UNET_RIGID_DEF_MODE}" \
    --seed "${seed}" \
    --special_tag "rigid_def_unet"

for rigid_def_mode in "${RIGID_DEF_MODES[@]}"; do
    for mode in "${MODES[@]}"; do
        echo "--- submit: modality=${MODALITY} rigid_def=${rigid_def_mode} mode=${mode} ---"
        # The W&B group/run name does not encode rigid_def_mode, so the tag is the
        # only thing separating the calc and deep sweeps in the UI.
        ./remote_submit.sh "${SCRIPT}" \
            --modality "${MODALITY}" \
            --rigid_def_mode "${rigid_def_mode}" \
            --mode "${mode}" \
            --seed "${seed}" \
            --special_tag "rigid_def_${rigid_def_mode}"
    done
done

echo "=== Sweep submission complete ==="
