"""Publish source/protocol and actual evaluation outcomes beside each public model."""
import hashlib
import json
from pathlib import Path
from huggingface_hub import CommitOperationAdd

ROOT=Path('/workspace/pi05-libero')


def publish_cards(api, state):
    path=ROOT/'results/manifest.json'
    if not path.exists():
        return
    manifest=json.loads(path.read_text())
    for task in manifest['tasks']:
        operations=[]
        rows=[]
        for step in [250,500,750,1000]:
            result=ROOT/'results'/task['slug']/f'{step:06d}'/'eval.json'
            if result.exists():
                data=result.read_bytes()
                record=json.loads(data)
                overall=record['evaluation']['overall']
                rows.append(f'| {step} | {overall["n_success"]}/5 | {overall["pc_success"]:.0f}% |')
                key=f'published/{task["slug"]}/{step}'
                digest=hashlib.sha256(data).hexdigest()
                if state.get(key)!=digest:
                    operations.append(CommitOperationAdd(path_in_repo=f'evaluations/{step:06d}/eval.json',path_or_fileobj=data))
                    state[key]=digest
            else:
                rows.append(f'| {step} | pending | pending |')
        card=f'''---
library_name: lerobot
license: apache-2.0
tags:
- pi05
- libero
- robotics
- single-task-sft
base_model: lerobot/pi05_base
datasets:
- lerobot/libero
---

# Pi0.5 SFT: {task['language']}

An independent 1000-step full SFT run initialized from `lerobot/pi05_base`.
This is an experiment with evaluations in progress; consult the table for completed results.

- Task suite: LIBERO-goal; task ID: {task['libero_task_id']}.
- Training examples: {task['episode_count']} demonstrations, {task['frame_count']} frames, filtered by exact task.
- Dataset revision: `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`.
- LeRobot commit: `{manifest['lerobot_commit']}`.
- Full SFT, all 4,143,404,816 parameters trainable; no LoRA or frozen backbone.
- FP32 weights and AdamW state; BF16 autocast; batch 8; seed 1000.
- Learning rate 5e-5, warmup 50 steps, cosine decay to 5e-6 at step 1000.
- Mean/std normalization from the original dataset; two cameras plus one masked camera.
- Action chunk length 50, execute 10 actions; 10 flow-matching inference steps.

## Checkpoints and evaluation

Complete resumable checkpoints are uploaded under `checkpoints/000250`, `000500`, `000750`, `001000`.
Each has a `pretrained_model` directory and `training_state` (including the optimizer and RNG).
The repository root is populated with the final inference model after training completes.

Every checkpoint is evaluated with 5 attempts: initial states 0–4, environment seeds 1000–1004,
hard resets, 256×256 pixels, relative end-effector control at 20 Hz, 300-step episode limit.
SR is the number of successes divided by 5. Five attempts give a coarse estimate in 20-point increments.

| Step | Successes | SR |
|---:|---:|---:|
'''+ '\n'.join(rows)+f'''

Raw outcomes are saved under `evaluations/<step>/eval.json` as each evaluation completes.

## Load a saved checkpoint

```python
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
policy = PI05Policy.from_pretrained("{task['repo_id']}", revision="000250", subfolder="checkpoints/000250/pretrained_model")
```

Alternatively, download the checkpoint subtree and pass its `pretrained_model` directory to `lerobot-eval`.
The tokenizer uses the official public Big Vision PaliGemma vocabulary with verified matching token IDs,
saved at `{manifest['tokenizer_repo']}`.
'''
        # Subfolder support differs by LeRobot release; give the supported snapshot path recipe.
        card=card[:card.index('## Load a saved checkpoint')]+f'''## Load a saved checkpoint

```python
from huggingface_hub import snapshot_download
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
root = snapshot_download("{task['repo_id']}", allow_patterns="checkpoints/000250/pretrained_model/*")
policy = PI05Policy.from_pretrained(root + "/checkpoints/000250/pretrained_model")
```

The tokenizer uses the official public Big Vision PaliGemma vocabulary with verified matching token IDs,
saved at `{manifest['tokenizer_repo']}`.
'''
        digest=hashlib.sha256(card.encode()).hexdigest()
        key=f'published/{task["slug"]}/README'
        if state.get(key)!=digest:
            operations.append(CommitOperationAdd(path_in_repo='README.md',path_or_fileobj=card.encode()))
        if operations:
            api.create_commit(repo_id=task['repo_id'],operations=operations,commit_message='Record SFT protocol and completed evaluations')
            state[key]=digest
