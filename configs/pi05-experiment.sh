#!/bin/bash
set -e
. /opt/supervisor-scripts/utils/logging.sh
. /opt/supervisor-scripts/utils/environment.sh
export HF_HOME=/workspace/.cache/huggingface
export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_DISABLE_XET=1
export MUJOCO_GL=egl
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
cd /workspace/pi05-libero
exec /venv/main/bin/python -u scripts/run_experiment.py
