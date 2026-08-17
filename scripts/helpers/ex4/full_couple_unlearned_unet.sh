#!/usr/bin/env bash
set -uo pipefail  # Keep testing combinations and report every failure.
# USE_UNLEARNED_SEGMENTATOR=false
USE_UNLEARNED_SEGMENTATOR=true
seed=42
# seed=1234
# seed=219
MODES=(
    "UNET"
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
MODALITIES=("affine" "deformed")
SCRIPT="experiments/ex4/initial_decoupled.py"
SPECIAL_TAG="learned_segmentator_coupled"
FAILED=()
SEGMENTATOR_ARGS=()
if [[ "${USE_UNLEARNED_SEGMENTATOR}" == "true" ]]; then
    SEGMENTATOR_ARGS+=("--segmentator_unlearned")
    SPECIAL_TAG="unlearned_segmentator_coupled"
fi

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
            --smoke_test \
            "${SEGMENTATOR_ARGS[@]}"; then
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
# for modality in "${MODALITIES[@]}"; do
#     for mode in "${MODES[@]}"; do
#         echo "--- submit: modality=${modality} mode=${mode} ---"
#         ./remote_submit.sh "${SCRIPT}" \
#             --modality "${modality}" \
#             --mode "${mode}" \
#             --seed "${seed}" \
#             --learning_sample_strategy "no_gt" \
#             --validation_sample_strategy "always_gt" \
#             --special_tag "${SPECIAL_TAG}" \
#             "${SEGMENTATOR_ARGS[@]}"

#     done
# done

echo "=== Sweep submission complete ==="
