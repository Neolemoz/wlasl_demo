#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip wheel setuptools

# CPU-only PyTorch (no CUDA; AMD GPU not used in MVP)
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

# Remaining deps
python -m pip install -r requirements.txt

echo "OK: venv ready at .venv/"
echo "Next: python scripts/healthcheck.py"
