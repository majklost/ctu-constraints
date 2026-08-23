#!/bin/bash
# Run this locally on your Mac to launch, tunnel, and clean up a cluster Jupyter server.

REMOTE="mrkosmic@rci"
PROJECT_DIR="/mnt/personal/mrkosmic/synced/constraints"
LOCAL_PORT=8080

# Clean muting for Mutagen startup checks
mutagen project start >/dev/null 2>&1
mutagen project resume >/dev/null 2>&1
mutagen sync flush constraints >/dev/null 2>&1
echo "=== Submitting Jupyter job to SLURM ==="

# FIX: Added -q (quiet) and -T (no pseudo-terminal) flags.
# Replaced Heredoc with a single inline execution string to completely suppress login node banners.
JOB_INFO=$(ssh -q -T "${REMOTE}" "source /etc/profile && ml PyTorch-Lightning/2.5.5-foss-2025a-CUDA-12.8.0 && cd ${PROJECT_DIR} && source ./.venv/bin/activate && sbatch --export=ALL start_jupyter.sh")

# Strict extraction: Pulls out the numerical field from the line containing "batch job"
JOB_ID=$(echo "$JOB_INFO" | grep "batch job" | awk '{print $NF}')

if [ -z "$JOB_ID" ] || ! [[ "$JOB_ID" =~ ^[0-9]+$ ]]; then
    echo "Failed to split or parse a pure SLURM Job ID."
    echo "----------------------------------------------------"
    echo "Raw output received was:"
    echo "$JOB_INFO"
    echo "----------------------------------------------------"
    exit 1
fi

echo "Job submitted successfully! SLURM Job ID: ${JOB_ID}"
LOG_FILE="notebooks/logs/jupyter_log_${JOB_ID}.txt"

cleanup() {
    echo -e "\n=== Shutting down... ==="
    echo "Canceling SLURM job ${JOB_ID} on cluster to free up GPUs..."
    ssh -q -T "${REMOTE}" "scancel ${JOB_ID}" >/dev/null 2>&1
    echo "Closing local SSH tunnels..."
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

echo "Waiting for the job to start allocation and generate the log file..."
while true; do
    LOG_DATA=$(ssh -q -T "${REMOTE}" "cat ${PROJECT_DIR}/${LOG_FILE} 2>/dev/null")
    if echo "$LOG_DATA" | grep -q "TARGET_CONNECTION_DATA"; then
        break
    fi
    sleep 2
    echo -n "."
done
echo -e "\nJob is running!"
sleep 3

CONNECTION_LINE=$(echo "$LOG_DATA" | grep "TARGET_CONNECTION_DATA")
NODE_NAME=$(echo "$CONNECTION_LINE" | cut -d':' -f2)
JUPYTER_PORT=$(echo "$CONNECTION_LINE" | cut -d':' -f3)

if [ -z "$NODE_NAME" ] || [ -z "$JUPYTER_PORT" ]; then
    echo "Error parsing node name or port from log file."
    exit 1
fi

echo "=================================================================="
echo "CONNECTED TO REMOTE GPU NODE: ${NODE_NAME} on Port: ${JUPYTER_PORT}"
echo "------------------------------------------------------------------"
echo "IN YOUR LOCAL VS CODE JUPYTER KERNEL CONFIGURATION, PASTE:"
echo "http://localhost:${LOCAL_PORT}/?token=ctu-gpu-token"
echo "=================================================================="
echo "KEEP THIS TERMINAL RUNNING. PRESS [Ctrl+C] TO STOP WORK AND CANCEL GPU JOB."

ssh -N -L ${LOCAL_PORT}:${NODE_NAME}:${JUPYTER_PORT} "${REMOTE}"
