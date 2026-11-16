#!/bin/bash
set -e
. /opt/supervisor-scripts/utils/logging.sh
. /opt/supervisor-scripts/utils/environment.sh
export HF_HOME=/workspace/.cache/huggingface
exec /venv/main/bin/python -u /workspace/pi05-libero/scripts/sync_artifacts.py
