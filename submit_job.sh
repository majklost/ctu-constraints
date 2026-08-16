#!/bin/bash
#SBATCH --job-name=dl_job
#SBATCH --partition=amdgpufast,gpufast,h200fast
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=synced/slurm/slurm-%j.out
#SBATCH --error=synced/slurm/slurm-%j.err

# ---- Configuration ----
PROJECT_DIR="/mnt/personal/mrkosmic/synced/constraints"
VENV_DIR="${PROJECT_DIR}/.venv"

# ---- Setup ----
# Module is loaded on the login node (via remote_submit.sh) and propagated via --export=ALL

# Create venv if it doesn't exist (--system-site-packages to inherit torch from module)
if [ ! -d "${VENV_DIR}" ]; then
	echo "Creating venv at ${VENV_DIR} ..."
	python -m venv --system-site-packages "${VENV_DIR}"
fi

# Activate the venv
source "${VENV_DIR}/bin/activate"

# Install the project and its dependencies in editable mode (skips already-satisfied packages, ignores optional/extras)
pip install --quiet -e "${PROJECT_DIR}"

# Move to project directory
cd "${PROJECT_DIR}" || exit 1

# Create logs directory if it doesn't exist
mkdir -p logs

# ---- Run ----
# Usage: sbatch submit_job.sh labs/04/cifar10_competition.py --arg1 val1
if [ $# -eq 0 ]; then
	echo "Error: No Python script specified."
	echo "Usage: sbatch submit_job.sh <script.py> [args...]"
	exit 1
fi

SCRIPT="$1"
shift

echo "=== Job Info ==="
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       $(hostname)"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python:     $(python --version)"
echo "Torch:      $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A')"
echo "Script:     ${SCRIPT}"
echo "Args:       $*"
echo "Started:    $(date)"
echo "================"

python -u "${SCRIPT}" "$@"

EXIT_CODE=$?
echo "Finished:   $(date)"
echo "Exit code:  ${EXIT_CODE}"
exit ${EXIT_CODE}
