"""Run upstream trainer with durable checkpoint/evaluation hooks, no model changes."""
import contextlib
import io
import json
import os
import runpy
import shutil
import sys
import time
import gc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path('/workspace/pi05-libero')
sys.path.insert(0, str(ROOT/'scripts'))
from sync_artifacts import sync_once


def main():
    from parallel_lfs import install
    install()
    slug = sys.argv[1]
    manifest = json.loads((ROOT/'results/manifest.json').read_text())
    task = next(t for t in manifest['tasks'] if t['slug'] == slug)
    output = ROOT/'outputs'/slug
    current_checkpoint = None
    uploader = ThreadPoolExecutor(max_workers=1, thread_name_prefix='hf-checkpoint')
    pending_uploads = []

    import lerobot.common.train_utils as train_utils
    import lerobot.scripts.lerobot_eval as eval_module
    import lerobot.policies as policies_module
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    # Upstream loader can silently return randomly initialized weights after a download
    # failure. Require the documented all-keys-success message before any training.
    original_load = PI05Policy.from_pretrained.__func__

    @classmethod
    def verified_load(cls, *args, **kwargs):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            policy = original_load(cls, *args, **kwargs)
        message = buffer.getvalue()
        print(message, flush=True)
        if 'All keys loaded successfully!' not in message:
            raise RuntimeError('Pretrained weights were not completely loaded; refusing to train')
        return policy

    PI05Policy.from_pretrained = verified_load
    original_processors = policies_module.make_pre_post_processors

    def configured_processors(*args, **kwargs):
        # The base checkpoint stores its own tokenizer name in its processor JSON;
        # overriding policy.text_tokenizer_name alone does not replace that field.
        overrides = kwargs.setdefault('preprocessor_overrides', {})
        overrides['tokenizer_processor'] = {'tokenizer_name': manifest['tokenizer_repo']}
        return original_processors(*args, **kwargs)

    policies_module.make_pre_post_processors = configured_processors
    original_push = train_utils.push_checkpoint_to_hub

    def upload_checkpoint(checkpoint_dir, repo_id, **kwargs):
        checkpoint_dir = Path(checkpoint_dir)
        for attempt in range(5):
            try:
                original_push(checkpoint_dir, repo_id, **kwargs)
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(10)
        (checkpoint_dir/'.uploaded').write_text(repo_id)
        (ROOT/'results'/slug).mkdir(exist_ok=True,parents=True)
        (ROOT/'results'/slug/f'{int(checkpoint_dir.name):06d}-upload.json').write_text(json.dumps({
            'step':int(checkpoint_dir.name),'repo_id':repo_id,'remote_uploaded':True},indent=2))
        print('CHECKPOINT_UPLOAD_COMPLETE',slug,checkpoint_dir.name,flush=True)
        # Never remove the latest local resume state. Completed earlier checkpoints remain on HF.
        latest=(checkpoint_dir.parent/'last').resolve()
        for old in checkpoint_dir.parent.iterdir():
            if old.is_dir() and not old.is_symlink() and old.resolve() != latest:
                if (old/'.uploaded').exists() and (old/'.evaluated').exists():
                    shutil.rmtree(old)
        # The independent pi05-sync service backs up metadata. Running another
        # Xet upload in this process can block the GPU thread behind large files.

    def durable_push(checkpoint_dir, repo_id, **kwargs):
        nonlocal current_checkpoint
        current_checkpoint = Path(checkpoint_dir)
        # Save has finished: immutable files can be uploaded while the GPU evaluates/trains.
        for future in pending_uploads:
            if future.done():
                future.result()
        pending_uploads.append(uploader.submit(upload_checkpoint,checkpoint_dir,repo_id,**kwargs))

    train_utils.push_checkpoint_to_hub = durable_push
    original_save = train_utils.save_checkpoint

    def save_with_space(*args, **kwargs):
        # At most one GPU trainer writes checkpoints; uploads free older local copies.
        while shutil.disk_usage(ROOT).free < 55 * 1024**3:
            print('WAITING_FOR_UPLOADS_TO_FREE_DISK',slug,flush=True)
            for future in pending_uploads:
                if future.done():
                    future.result()
            time.sleep(15)
        return original_save(*args, **kwargs)

    train_utils.save_checkpoint = save_with_space
    original_eval = eval_module.eval_policy_all

    def durable_eval(*args, **kwargs):
        info = original_eval(*args, **kwargs)
        assert current_checkpoint is not None
        step = int(current_checkpoint.name)
        destination = ROOT/'results'/slug/f'{step:06d}'
        destination.mkdir(parents=True,exist_ok=True)
        record = {'slug': slug, 'language': task['language'], 'step': step,
                  'checkpoint_repo': task['repo_id'], 'checkpoint_path': f'checkpoints/{current_checkpoint.name}',
                  'n_episodes': 5, 'seed': 1000, 'libero_task_id': task['libero_task_id'],
                  'evaluation': info}
        (destination/'eval.json').write_text(json.dumps(record,indent=2,default=lambda x: x.tolist() if hasattr(x,'tolist') else str(x)))
        for video in (output/'eval'/f'videos_step_{current_checkpoint.name}').rglob('*.mp4'):
            shutil.copy2(video, destination/video.name)
        (current_checkpoint/'.evaluated').write_text(str(destination/'eval.json'))
        print('EVALUATION_SAVED',slug,step,json.dumps(info.get('overall',{}),default=str),flush=True)
        return info

    eval_module.eval_policy_all = durable_eval
    argv = [
        'lerobot-train', '--dataset.repo_id=lerobot/libero',
        f'--dataset.root={ROOT}/data/libero',
        f'--dataset.episodes={json.dumps(task["episodes"])}',
        '--dataset.video_backend=pyav',
        '--policy.type=pi05', '--policy.pretrained_path=lerobot/pi05_base',
        f'--policy.text_tokenizer_name={manifest["tokenizer_repo"]}',
        '--policy.normalization_mapping={"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}',
        '--policy.n_action_steps=10', '--policy.empty_cameras=1',
        '--policy.freeze_vision_encoder=false', '--policy.train_expert_only=false',
        '--policy.gradient_checkpointing=true', '--policy.dtype=float32',
        '--policy.device=cuda', '--policy.optimizer_fused=true',
        '--policy.optimizer_lr=0.00005', '--policy.scheduler_warmup_steps=50',
        '--policy.scheduler_decay_steps=1000', '--policy.scheduler_decay_lr=0.000005',
        f'--policy.repo_id={task["repo_id"]}', '--policy.private=false',
        '--policy.push_to_hub=false', '--save_checkpoint_to_hub=true',
        '--accelerator.mixed_precision=bf16',
        f'--output_dir={output}', f'--job_name=pi05_{slug}',
        '--batch_size=8', '--num_workers=4', '--steps=1000',
        '--save_freq=250', '--env_eval_freq=250', '--log_freq=10', '--seed=1000',
        '--env.type=libero', '--env.task=libero_goal',
        f'--env.task_ids=[{task["libero_task_id"]}]',
        '--env.observation_height=256', '--env.observation_width=256',
        '--env.init_states=true', '--env.hard_reset=true',
        '--eval.n_episodes=5', '--eval.batch_size=5', '--eval.use_async_envs=true',
    ]
    last = output/'checkpoints/last/pretrained_model/train_config.json'
    if last.exists():
        argv = ['lerobot-train', f'--config_path={last}', '--resume=true', '--policy.push_to_hub=false']
    # Recover complete local checkpoints whose upload was interrupted. Their
    # files stay immutable while the resumed GPU run advances independently.
    for checkpoint in sorted((output/'checkpoints').glob('[0-9]*')):
        if (checkpoint/'training_state/training_step.json').exists() and not (checkpoint/'.uploaded').exists():
            pending_uploads.append(uploader.submit(upload_checkpoint,checkpoint,task['repo_id'],private=False))
    (ROOT/'configs').mkdir(exist_ok=True)
    (ROOT/'configs'/f'{slug}-command.json').write_text(json.dumps(argv,indent=2))
    sys.argv = argv
    already_trained = (output/'checkpoints/001000/training_state/training_step.json').exists() and all(
        (ROOT/'results'/slug/f'{step:06d}'/'eval.json').exists() for step in [250,500,750,1000])
    if already_trained:
        print('RECOVERING_UPLOADS_ONLY_FOR_COMPLETED_TRAINING',slug,flush=True)
    else:
        runpy.run_module('lerobot.scripts.lerobot_train', run_name='__main__')
    # Let the next independent SFT use the GPU while this process finishes its HF queue.
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    allocated=torch.cuda.memory_allocated()
    assert allocated < 1024**3, f'GPU still holds {allocated} bytes after training; cannot launch next SFT'
    (ROOT/'results'/f'{slug}-training-finished.json').write_text(json.dumps({'steps':1000,'status':'uploading','gpu_bytes_remaining':allocated}))
    for future in pending_uploads:
        future.result()
    uploader.shutdown(wait=True)
    # Publish the already-uploaded final model at the repository root without transferring it again.
    from huggingface_hub import HfApi, CommitOperationCopy
    api=HfApi()
    final_dir=output/'checkpoints/001000/pretrained_model'
    operations=[CommitOperationCopy(src_path_in_repo=f'checkpoints/001000/pretrained_model/{p.relative_to(final_dir)}',
                path_in_repo=str(p.relative_to(final_dir))) for p in final_dir.rglob('*') if p.is_file()]
    api.create_commit(repo_id=task['repo_id'],operations=operations,commit_message='Publish final 1000-step policy from uploaded checkpoint')
    (ROOT/'results'/f'{slug}-completed.json').write_text(json.dumps({'steps':1000,'repo_id':task['repo_id']}))


if __name__ == '__main__':
    main()
