#!/usr/bin/env bash
set -euo pipefail
: "${COMFY_ROOT:?Set COMFY_ROOT to the pinned ComfyUI checkout}"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${COMFY_PYTHON:-${COMFY_ROOT}/.venv/bin/python}"
export YOLO_CONFIG_DIR="${COMFY_ROOT}/.rgb240-ultralytics"
export YOLO_OFFLINE=true
export YOLO_AUTOINSTALL=false
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
"${python_bin}" "${repo_dir}/scripts/models.py" --model-root "${COMFY_ROOT}/models"
exec "${python_bin}" "${COMFY_ROOT}/main.py" --cpu --listen 127.0.0.1 \
  --port "${COMFY_PORT:-8188}" --disable-auto-launch --offline --disable-assets \
  --disable-all-custom-nodes --whitelist-custom-nodes product_studio_rgb240 ComfyUI_BiRefNet_ll
