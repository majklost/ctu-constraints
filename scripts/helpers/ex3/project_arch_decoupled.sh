#!/usr/bin/env bash
set -uo pipefail  # NOT -e: we want to keep going and report failures, not abort

MODES=("decoupledOneSideSDF" "decoupledCE" "decoupledStandard" "decoupledDSDF" "decoupledCentroid" "decoupledBlurred")
MODALITIES=("affine" "deformed")
SCRIPT="experiments/ex3/project_arch_decoupled.py"

FAILED=()

echo "=== SMOKE TEST: all mode x modality combos ==="
for modality in "${MODALITIES[@]}"; do
	for mode in "${MODES[@]}"; do
		echo "--- smoke: modality=${modality} mode=${mode} ---"
		if ! python "${SCRIPT}" \
				--modality "${modality}" \
				--mode "${mode}" \
				--batch_size 4 \
				--num_workers 0 \
				--smoke_test; then
			echo "!!! SMOKE TEST FAILED: modality=${modality} mode=${mode}"
			FAILED+=("${modality}/${mode}")
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
	for mode in "${MODES[@]}"; do
		echo "--- full run: modality=${modality} mode=${mode} ---"
		./remote_submit.sh "${SCRIPT}" \
			--modality "${modality}" \
			--mode "${mode}"
	done
done

echo "=== Sweep complete ==="
