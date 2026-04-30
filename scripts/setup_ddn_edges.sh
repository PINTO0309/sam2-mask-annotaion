#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
DDN_VENV="${DDN_VENV:-${ROOT_DIR}/.venv-ddn}"

cd "$ROOT_DIR"

uv venv "$DDN_VENV" --python "$PYTHON_VERSION"
uv pip install --python "$DDN_VENV/bin/python" \
  torch==2.4.0 \
  torchvision==0.19.0 \
  timm==0.4.12 \
  gdown==5.2.0 \
  numpy==2.2.6 \
  pillow==12.2.0

echo "DDN edge environment ready: ${DDN_VENV}"
echo "The DDN checkpoint downloads automatically on first use into checkpoints/ddn/."
