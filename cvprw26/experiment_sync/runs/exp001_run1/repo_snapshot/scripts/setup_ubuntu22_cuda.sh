#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-bright_cvprw26}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CUDA_WHL_URL="${CUDA_WHL_URL:-https://download.pytorch.org/whl/cu121}"
CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[1/6] Basic system packages"
sudo apt-get update
sudo apt-get install -y git wget curl tmux unzip build-essential htop

echo "[2/6] Check NVIDIA driver"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  cat <<'EOF'
nvidia-smi not found.
Install the NVIDIA driver first, then reboot, then rerun this script.

Recommended commands on Ubuntu 22.04:
  sudo apt-get update
  sudo apt-get install -y ubuntu-drivers-common
  sudo ubuntu-drivers autoinstall
  sudo reboot
EOF
  exit 1
fi

nvidia-smi

echo "[3/6] Install Miniconda if missing"
if [ ! -d "$CONDA_HOME" ]; then
  TMP_INSTALLER="/tmp/miniconda.sh"
  wget -O "$TMP_INSTALLER" "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
  bash "$TMP_INSTALLER" -b -p "$CONDA_HOME"
fi

source "$CONDA_HOME/etc/profile.d/conda.sh"

echo "[4/6] Create or update conda env: $ENV_NAME"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda activate "$ENV_NAME"
else
  conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
  conda activate "$ENV_NAME"
fi

python -m pip install --upgrade pip setuptools wheel

echo "[5/6] Install PyTorch with CUDA wheels"
python -m pip install torch torchvision --index-url "$CUDA_WHL_URL"

echo "[6/6] Install project"
cd "$REPO_DIR"
python -m pip install -e .

echo
echo "Environment ready."
echo "Repo: $REPO_DIR"
echo "Env : $ENV_NAME"
echo
echo "Next:"
echo "  1. Upload data into: $REPO_DIR/BRIGHT_DATA/"
echo "  2. Start training:"
echo "     bash scripts/train_server_tmux.sh"
