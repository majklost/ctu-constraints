#!/bin/bash
#SBATCH --job-name=remote-kernel
#SBATCH --partition=amdgpufast
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=notebooks/logs/jupyter_log_%j.txt

mkdir -p notebooks/logs

NODE_NAME=$(hostname)
JUPYTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

echo "TARGET_CONNECTION_DATA:${NODE_NAME}:${JUPYTER_PORT}"

source .venv/bin/activate

# Use the lighter 'jupyter server' instead of 'jupyterlab' interface
# since your local VS Code is providing the actual UI layer.
python -m jupyter server \
    --no-browser \
    --ip=0.0.0.0 \
    --port=${JUPYTER_PORT} \
    --ServerApp.token='ctu-gpu-token' \
    --ServerApp.allow_remote_access=True