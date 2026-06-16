#!/bin/bash
# Submit a SLURM job on RCI directly from your local machine.
# Usage: ./remote_submit.sh labs/04/cifar10_competition.py --epochs 10
# example: 
# REMOTE="mrkosmic@rci"
# PROJECT_DIR="/mnt/personal/mrkosmic/DL/synced"
mutagen project start
mutagen project resume
REMOTE="mrkosmic@rci"
PROJECT_DIR="/mnt/personal/mrkosmic/synced/constraints"

if [ $# -eq 0 ]; then
	echo "Error: No Python script specified."
	echo "Usage: ./remote_submit.sh <script.py> [args...]"
	exit 1
fi

# Build the sbatch command with all arguments properly quoted
ARGS=$(printf ' %q' "$@")

# Pipe commands to ssh so they run in a login shell where 'ml' works.
# The module environment is propagated to the compute node via --export=ALL.
ssh "${REMOTE}" << EOF
ml PyTorch-Lightning/2.5.5-foss-2025b-CUDA-12.9.1
echo "HPC modules loaded successfully."
cd ${PROJECT_DIR} && sbatch --export=ALL submit_job.sh${ARGS}
EOF
