# Pi0.5 single-task LIBERO-goal SFT

Three independent runs initialized from lerobot/pi05_base (not LIBERO-finetuned weights).
Tasks: put the bowl on the plate; open the middle drawer of the cabinet; push the plate to the front of the stove.
Each run: 1000 optimizer steps. Save/evaluate at steps 250, 500, 750, 1000; five episodes per checkpoint with identical seeds/initial-state protocol.
Filter demonstration episodes by exact task, verify no other tasks enter each run.
Use a GPU-compatible CUDA >=12.8 PyTorch wheel; full SFT if memory permits, record all precision/freeze settings.
Upload each completed resumable checkpoint and evaluation results immediately to public HF model repositories in the machine-authenticated qrvmil namespace. Save code, config, provenance and logs continuously; supervisor manages long processes.
Report successes/5 and SR per task/checkpoint. Never substitute unrun evaluations with zero.

## Execution
- [x] Verify GPU tensors, HF write access, GitHub access.
- [x] Install pinned LeRobot and LIBERO runtime; inspect exact benchmark tasks and demonstration mapping.
- [x] Build and verify per-task datasets and training/evaluation configuration.
- [x] Run three 1000-step SFT jobs, save immutable checkpoints locally, upload in the background, and evaluate at the matching step.
- [x] Evaluate each checkpoint on five attempts and upload episode outcomes.
- [x] Publish aggregate SR table and verify all remote artifacts.

## Verified outcome

All three independent 1000-step SFT runs, all 12 complete HF checkpoints, and all 60 evaluation attempts are verified. Final SR: bowl 5/5 (100%), drawer 2/5 (40%), plate 3/5 (60%); mean 10/15 (66.7%). See results/FINAL_VERIFICATION.json and results/REPORT.md.
