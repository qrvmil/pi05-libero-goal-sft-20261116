"""Recover an evaluation if the instance stopped after checkpoint upload."""
import json
import shutil
import sys
from pathlib import Path

import torch
from huggingface_hub import HfApi
from lerobot.configs import PreTrainedConfig
from lerobot.envs import make_env, make_env_pre_post_processors, close_envs
from lerobot.envs.configs import LiberoEnvConfig
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.random_utils import set_seed

ROOT = Path('/workspace/pi05-libero')
from sync_artifacts import sync_once


def main(slug, checkpoint_dir):
    task = next(t for t in json.loads((ROOT/'results/manifest.json').read_text())['tasks'] if t['slug']==slug)
    checkpoint_dir=Path(checkpoint_dir)
    step=int(checkpoint_dir.name)
    pretrained=checkpoint_dir/'pretrained_model'
    cfg=PreTrainedConfig.from_pretrained(pretrained)
    cfg.device='cuda'
    cfg.pretrained_path=pretrained
    set_seed(1000)
    torch.backends.cuda.matmul.allow_tf32=True
    env_cfg=LiberoEnvConfig(task='libero_goal',task_ids=[task['libero_task_id']],
                           observation_height=256,observation_width=256)
    envs=make_env(env_cfg,n_envs=5,use_async_envs=True)
    try:
        policy=make_policy(cfg=cfg,env_cfg=env_cfg)
        pre,post=make_pre_post_processors(policy_cfg=cfg,pretrained_path=pretrained,
            preprocessor_overrides={'device_processor':{'device':'cuda'}})
        env_pre,env_post=make_env_pre_post_processors(env_cfg=env_cfg,policy_cfg=cfg)
        destination=ROOT/'results'/slug/f'{step:06d}'
        destination.mkdir(parents=True,exist_ok=True)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            info=eval_policy_all(envs=envs,policy=policy,env_preprocessor=env_pre,
                env_postprocessor=env_post,preprocessor=pre,postprocessor=post,n_episodes=5,
                start_seed=1000,max_episodes_rendered=5,videos_dir=destination/'videos')
        record={'slug':slug,'language':task['language'],'step':step,'checkpoint_repo':task['repo_id'],
                'checkpoint_path':f'checkpoints/{checkpoint_dir.name}','n_episodes':5,'seed':1000,
                'libero_task_id':task['libero_task_id'],'evaluation':info,'recovered_evaluation':True}
        (destination/'eval.json').write_text(json.dumps(record,indent=2))
        sync_once()
        (checkpoint_dir/'.evaluated').write_text(str(destination/'eval.json'))
    finally:
        close_envs(envs)


if __name__=='__main__':
    main(sys.argv[1],sys.argv[2])
