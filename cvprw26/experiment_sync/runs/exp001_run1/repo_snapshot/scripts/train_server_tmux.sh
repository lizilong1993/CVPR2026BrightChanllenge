#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-bright}"
ENV_NAME="${ENV_NAME:-bright_cvprw26}"
CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"
CONFIG_PATH="${CONFIG_PATH:-config/disaster.server.yaml}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed."
  exit 1
fi

if [ ! -f "$CONDA_HOME/etc/profile.d/conda.sh" ]; then
  echo "Conda not found at $CONDA_HOME"
  exit 1
fi

if [ ! -f "$REPO_DIR/$CONFIG_PATH" ]; then
  echo "Config not found: $REPO_DIR/$CONFIG_PATH"
  exit 1
fi

OUTPUT_DIR="$(python - <<PY
from pathlib import Path
import yaml

config_path = Path(r"$REPO_DIR") / r"$CONFIG_PATH"
with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
print(cfg["train"]["output_dir"])
PY
)"

if [ -d "$REPO_DIR/BRIGHT_DATA" ]; then
  for required_dir in post-event pre-event; do
    if [ ! -d "$REPO_DIR/BRIGHT_DATA/$required_dir" ]; then
      echo "Missing data directory: $REPO_DIR/BRIGHT_DATA/$required_dir"
      exit 1
    fi
  done
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME"
  echo "Attach with: tmux attach -t $SESSION_NAME"
  exit 1
fi

tmux new-session -d -s "$SESSION_NAME" \
  "source \"$CONDA_HOME/etc/profile.d/conda.sh\" && \
   conda activate \"$ENV_NAME\" && \
   cd \"$REPO_DIR\" && \
   python -u -m src.train --config \"$CONFIG_PATH\""

echo "Training started in tmux session: $SESSION_NAME"
echo "Attach : tmux attach -t $SESSION_NAME"
echo "Detach : Ctrl+b then d"
echo "Logs   : tail -f $REPO_DIR/$OUTPUT_DIR/train.log"
