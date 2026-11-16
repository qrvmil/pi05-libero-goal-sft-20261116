"""Verify checkpoint presence, distinct trained weights, and final-model copy on HF."""
import json
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

ROOT=Path('/workspace/pi05-libero')


def verify(task):
    api=HfApi()
    repo=task['repo_id']
    info=api.model_info(repo,files_metadata=True)
    files={f.rfilename:f for f in info.siblings}
    tags={ref.name for ref in api.list_repo_refs(repo).tags}
    weight_ids=[]
    for step in ['000250','000500','000750','001000']:
        assert step in tags,(repo,step,'missing tag')
        for path in ['pretrained_model/model.safetensors','pretrained_model/config.json',
                     'pretrained_model/policy_preprocessor.json','pretrained_model/policy_postprocessor.json',
                     'training_state/optimizer_state.safetensors','training_state/rng_state.safetensors',
                     'training_state/scheduler_state.json','training_state/training_step.json']:
            name=f'checkpoints/{step}/{path}'
            assert name in files and files[name].size>0,(repo,name)
        path=f'checkpoints/{step}/pretrained_model/model.safetensors'
        weight_ids.append(files[path].blob_id)
        state=json.load(open(hf_hub_download(repo,f'checkpoints/{step}/training_state/training_step.json')))
        assert state['step']==int(step),(repo,state)
        evaluation=json.load(open(hf_hub_download(repo,f'evaluations/{step}/eval.json')))
        assert evaluation['evaluation']['overall']['n_episodes']==5
        assert evaluation['slug']==task['slug'] and evaluation['step']==int(step)
        assert evaluation['libero_task_id']==task['libero_task_id']
        successes=evaluation['evaluation']['per_task'][0]['metrics']['successes']
        assert len(successes)==5 and sum(successes)==evaluation['evaluation']['overall']['n_success']
    assert len(set(weight_ids))==4,(repo,'weights did not change across checkpoints')
    assert files['model.safetensors'].blob_id==files['checkpoints/001000/pretrained_model/model.safetensors'].blob_id
    cfg=json.load(open(hf_hub_download(repo,'config.json')))
    assert cfg['type']=='pi05' and cfg['output_features']['action']['shape']==[7]
    train_cfg=json.load(open(hf_hub_download(repo,'train_config.json')))
    assert train_cfg['steps']==1000
    assert train_cfg['dataset']['episodes']==task['episodes']
    assert train_cfg['env']['task']=='libero_goal'
    assert train_cfg['env']['task_ids']==[task['libero_task_id']]
    assert train_cfg['policy']['freeze_vision_encoder'] is False
    assert train_cfg['policy']['train_expert_only'] is False
    assert train_cfg['accelerator']['mixed_precision']=='bf16'
    assert not info.private
    record={'repo_id':repo,'commit':info.sha,'tags':sorted(tags),'file_count':len(files),
            'all_four_checkpoints_verified':True,'four_distinct_trained_weight_files':True,
            'final_root_weights_equal_step_1000':True,'all_evaluations_have_five_attempts':True,
            'files':{name:{'size':f.size,'blob_id':f.blob_id} for name,f in files.items()}}
    (ROOT/'results'/f'{task["slug"]}-remote-verification.json').write_text(json.dumps(record,indent=2))
    print(repo,'VERIFIED',info.sha,flush=True)
    return record


if __name__=='__main__':
    records=[]
    for task in json.loads((ROOT/'results/manifest.json').read_text())['tasks']:
        if (ROOT/'results'/f'{task["slug"]}-completed.json').exists():
            records.append(verify(task))
    if len(records)==3:
        assert len({r['files']['model.safetensors']['blob_id'] for r in records})==3
        rates=json.loads((ROOT/'results/success_rates.json').read_text())
        assert len(rates)==12 and sum(r['n_episodes'] for r in rates)==60
        remote_rates=json.load(open(hf_hub_download(records[0]['repo_id'],'success_rates.json')))
        assert rates==remote_rates
        summary={'status':'verified','sft_runs':3,'steps_per_run':1000,'checkpoints':12,
                 'evaluation_attempts':60,'distinct_final_models':3,
                 'repositories':[{'repo_id':r['repo_id'],'commit':r['commit']} for r in records]}
        (ROOT/'results/FINAL_VERIFICATION.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps(summary,indent=2))
