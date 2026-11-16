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
