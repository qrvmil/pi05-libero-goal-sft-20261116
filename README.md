# Pi0.5 — three independent full SFT runs on LIBERO-goal

This private repository preserves the experiment code, configuration, logs, measured results, rollout videos, tokenizer, checkpoint metadata and an exact source archive of the LeRobot version used.

**Full SFT:** all 4,143,404,816 parameters were trainable. No LoRA/PEFT, no frozen vision encoder or VLM. FP32 model parameters and optimizer; BF16 autocast. See `results/full-sft-audit.json` and training logs. Each independent model starts from `lerobot/pi05_base` and reaches 1000 optimizer steps. Checkpoints at 250/500/750/1000 each have five evaluation attempts (60 attempts total).

## Results

# Pi0.5: LIBERO-goal single-task SFT

Each run starts independently from `lerobot/pi05_base`. Five evaluation attempts per checkpoint.
SR = successful attempts / 5. Missing entries have not been evaluated.

| Task | Step 250 | Step 500 | Step 750 | Step 1000 |
|---|---:|---:|---:|---:|
| put the bowl on the plate | 1/5 (20%) | 0/5 (0%) | 3/5 (60%) | 5/5 (100%) |
| open the middle drawer of the cabinet | 0/5 (0%) | 4/5 (80%) | 1/5 (20%) | 2/5 (40%) |
| push the plate to the front of the stove | 1/5 (20%) | 1/5 (20%) | 2/5 (40%) | 3/5 (60%) |
| Overall across 3 tasks | 2/15 (13.3%) | 5/15 (33.3%) | 6/15 (40.0%) | 10/15 (66.7%) |

Completed evaluations: 12/12 (60/60 attempts).

Training: full SFT; FP32 parameters/optimizer, BF16 autocast; batch size 8; AdamW; LR 5e-5, warmup 50 steps, cosine decay to 5e-6 at step 1000; seed 1000.
Evaluation: LIBERO-goal, task IDs 8/0/5, initial states 0–4, environment seeds 1000–1004, hard resets, 256×256 images, 10 actions per inference, 300-step limit.
Dataset: lerobot/libero, filtered to the exact task. Shared full-dataset mean/std normalization.

Resumption: bowl_on_plate and middle_drawer were resumed from their step-250 training states during uploader orchestration changes; existing evaluations were retained. See resume-events.json in the experiment backup for details.

Five attempts give a coarse SR estimate with 20 percentage-point increments.

## Complete model checkpoints (Hugging Face)

The multi-GB model and optimizer tensors are stored on Hugging Face, not duplicated in GitHub:

- put the bowl on the plate: https://huggingface.co/qrvmil/pi05-libero-goal-bowl_on_plate-20261116
- open the middle drawer of the cabinet: https://huggingface.co/qrvmil/pi05-libero-goal-middle_drawer-20261116
- push the plate to the front of the stove: https://huggingface.co/qrvmil/pi05-libero-goal-plate_front_stove-20261116

Each model repository contains all four complete checkpoint trees under `checkpoints/<step>/`, including model weights, processor statistics, optimizer, scheduler and RNG state. The final inference model is also at the repository root. All twelve checkpoint trees and their step tags were verified after upload. See `results/FINAL_VERIFICATION.json` and `results/*-remote-verification.json`.

The private continuous experiment backup (including logs, scripts and all 48 recorded rollout videos) is https://huggingface.co/qrvmil/pi05-libero-goal-sft-20261116 . This GitHub repository also preserves those logs and videos.

## Contents

- `scripts/`: task preparation, exact episode filters, full-SFT training, evaluation recovery, upload orchestration, integrity tests, reports and remote verification.
- `configs/`: launch commands, LIBERO paths and supervisor configuration.
- `checkpoint_metadata/`: exact saved training/policy/processor configs, normalization statistics, optimizer parameter groups, scheduler state and RNG state for all 12 checkpoints. Large model and optimizer tensors are on HF.
- `results/`: episode manifests, source revisions, package versions, SR CSV/JSON/report, per-checkpoint evaluation outcomes and videos.
- `logs/`: actual training, preparation and uploader logs.
- `tokenizer/`: validated tokenizer built from the official public Big Vision PaliGemma vocabulary.
- `upstream/lerobot-source.tar.gz`: complete clean source tree at the exact upstream commit; its SHA256 is in `upstream/provenance.json`.

## Restore on another machine

The experiment scripts use `/workspace/pi05-libero` and `/venv/main`. Clone this repository to `/workspace/pi05-libero` on a fresh instance, then restore the bundled source:

```bash
cd /workspace/pi05-libero
mkdir -p lerobot
tar -xzf upstream/lerobot-source.tar.gz -C lerobot
```

Use Python 3.12 and the package versions in `results/requirements.txt` / `results/manifest.json`. The actual GPU runtime was PyTorch 2.10.0 with CUDA 12.8. The original public demonstration dataset is `lerobot/libero`, pinned by the revision recorded in `results/source-metadata.json`; task episode IDs are in `results/manifest.json`.

Download the needed complete checkpoint from its HF repository. For example:

```python
from huggingface_hub import snapshot_download
snapshot_download(
    "qrvmil/pi05-libero-goal-bowl_on_plate-20261116",
    allow_patterns="checkpoints/001000/*",
    local_dir="outputs/bowl_on_plate",
)
```

Completed-run markers are intentionally preserved in `results/`: the queue will skip finished SFTs. Review paths and checkpoint selection before deliberately starting another experiment.

Credential files, environment secrets, virtualenvs, caches and local multi-GB model/optimizer copies are excluded. Public dependencies and datasets can be fetched again; all experiment-specific artifacts and full trained checkpoints are preserved off the instance.
