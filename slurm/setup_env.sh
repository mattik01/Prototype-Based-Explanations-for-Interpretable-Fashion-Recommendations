#!/bin/bash
# One-time environment setup on LEO5. Run from login node.
set -e

echo "=== LEO5 ProtoMF Environment Setup ==="

# 1. Scratch directories
echo "Creating scratch directories..."
mkdir -p /scratch/c7031336/{protomf_data,protomf_logs,ray_tmp,ray_results,wandb_logs,wandb_cache,venvs}

# 2. Python venv
echo "Creating Python venv..."
module load python/3.11.6-pytorch-2.5.1-cuda-conda-2026.02
python -m venv /scratch/c7031336/venvs/protomf --system-site-packages
source /scratch/c7031336/venvs/protomf/bin/activate

# 3. Dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install "ray[tune]" wandb hyperopt scipy pandas bottleneck

# 4. W&B key
echo ""
echo "Place your W&B API key:"
echo "  echo 'YOUR_KEY' > /home/c703/c7031336/.wandb_key"

# 5. Verify
echo ""
echo "=== Verification ==="
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python -c "import ray; print(f'Ray {ray.__version__}')"
python -c "import wandb; print(f'W&B {wandb.__version__}')"
python -c "import hyperopt; print(f'Hyperopt {hyperopt.__version__}')"

echo ""
echo "=== Done ==="
echo "Next steps:"
echo "  1. Transfer data: rsync -avz data/ml-1m/*.csv leo5:/scratch/c7031336/protomf_data/ml-1m/"
echo "  2. Place W&B key: echo 'YOUR_KEY' > /home/c703/c7031336/.wandb_key"
echo "  3. Test: sbatch --gres=gpu:a30:1 --time=00:30:00 slurm/run_combo.sbatch debug ml-1m 38210573"
