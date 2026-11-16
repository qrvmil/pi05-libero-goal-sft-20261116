"""Prepare exact single-task manifests; fail closed on mismatched data or tokenizer."""
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers.convert_slow_tokenizer import GemmaConverter, GemmaSentencePieceExtractor
from tokenizers.processors import TemplateProcessing
from types import SimpleNamespace

ROOT = Path('/workspace/pi05-libero')
TASKS = [
    ('bowl_on_plate', 'put the bowl on the plate'),
    ('middle_drawer', 'open the middle drawer of the cabinet'),
    ('plate_front_stove', 'push the plate to the front of the stove'),
]

# transformers 5.22's legacy Gemma converter still reads .sp, although its
# base extractor now initializes only .proto. Supply the original public reader.
class CompatibleGemmaExtractor(GemmaSentencePieceExtractor):
    def __init__(self, model):
        super().__init__(model)
        self.sp = spm.SentencePieceProcessor(model_file=model)


class CompatibleGemmaConverter(GemmaConverter):
    SpmExtractor = CompatibleGemmaExtractor


def main():
    api = HfApi()
    assert torch.cuda.is_available()
    result = (torch.arange(128, device='cuda', dtype=torch.float32) ** 2).sum().item()
    assert result == sum(x*x for x in range(128))
    gpu = {'torch': torch.__version__, 'cuda': torch.version.cuda,
           'gpu': torch.cuda.get_device_name(), 'test_sum': result,
           'capability': list(torch.cuda.get_device_capability())}
    (ROOT/'results/gpu-test.json').write_text(json.dumps(gpu, indent=2))
    print('GPU computation verified:', gpu, flush=True)

    tokenizer_dir = ROOT/'tokenizer'
    backend = CompatibleGemmaConverter(SimpleNamespace(vocab_file=str(tokenizer_dir/'tokenizer.model'),
        pad_token='<pad>',eos_token='<eos>',bos_token='<bos>',unk_token='<unk>',
        add_prefix_space=False)).converted()
    backend.post_processor = TemplateProcessing(single='<bos> $A',pair='<bos> $A <bos> $B',
                                                special_tokens=[('<bos>',2)])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend,pad_token='<pad>',
        eos_token='<eos>',bos_token='<bos>',unk_token='<unk>',padding_side='right')
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_dir/'tokenizer.model'))
    for _, task in TASKS:
        prompt = f'Task: {task}, State: 1 2 3 4 5 6 7 8;\nAction: '
        assert tokenizer.encode(prompt) == [sp.bos_id()] + sp.encode(prompt), 'Tokenizer differs from official OpenPI vocabulary'
    tokenizer.save_pretrained(tokenizer_dir)
    tok_repo = 'qrvmil/pi05-paligemma-tokenizer-20261116'
    api.create_repo(tok_repo, private=True, exist_ok=True)
    api.upload_folder(repo_id=tok_repo, folder_path=str(tokenizer_dir), commit_message='Official public Big Vision PaliGemma vocabulary with validated Gemma tokenizer config')
    reloaded = AutoTokenizer.from_pretrained(tok_repo)
    assert reloaded.encode(prompt) == tokenizer.encode(prompt)
    (ROOT/'results/tokenizer.json').write_text(json.dumps({
        'repo_id': tok_repo, 'source': 'https://storage.googleapis.com/big_vision/paligemma_tokenizer.model',
        'sha256': hashlib.sha256((tokenizer_dir/'tokenizer.model').read_bytes()).hexdigest(),
        'sentencepiece_equivalence_verified': True,
    }, indent=2))

    data_root = ROOT/'data/libero'
    snapshot_download('lerobot/libero', repo_type='dataset', local_dir=data_root, max_workers=8)
    tasks = pd.read_parquet(data_root/'meta/tasks.parquet')
    episodes = pd.read_parquet(data_root/'meta/episodes/chunk-000/file-000.parquet')
    frame_tasks = pd.concat([pd.read_parquet(p,columns=['episode_index','task_index'])
        for p in sorted((data_root/'data').rglob('*.parquet'))],ignore_index=True)
    episode_tasks = frame_tasks.groupby('episode_index')['task_index'].agg(['min','max','count'])
    assert (episode_tasks['min'] == episode_tasks['max']).all(), 'Multi-task episode detected'
    print('Task table:', tasks.to_string(), flush=True)
    print('Episode columns:', list(episodes), flush=True)
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()['libero_goal']()
    suite_tasks = {suite.get_task(i).language: i for i in range(suite.n_tasks)}
    print('Benchmark tasks:', suite_tasks, flush=True)
    manifest = {'tokenizer_repo': tok_repo, 'tasks': []}
    for slug, language in TASKS:
        task_rows = tasks.loc[tasks.index == language] if language in tasks.index else tasks.loc[tasks.astype(str).eq(language).any(axis=1)]
        assert len(task_rows) == 1, (language, task_rows)
        task_index = int(task_rows.iloc[0]['task_index'])
        matching_ids = episode_tasks.index[episode_tasks['min'] == task_index]
        selected = episodes[episodes['episode_index'].isin(matching_ids)]
        assert len(selected) > 0, language
        assert len(selected) == len(matching_ids)
        episode_ids = [int(x) for x in selected['episode_index']]
        task_id = suite_tasks[language]
        repo = f'qrvmil/pi05-libero-goal-{slug}-20261116'
        api.create_repo(repo, private=True, exist_ok=True)
        item = {'slug': slug, 'language': language, 'task_index': task_index,
                'libero_task_id': task_id, 'episodes': episode_ids,
                'episode_count': len(episode_ids), 'frame_count': int(selected['length'].sum()),
                'repo_id': repo}
        manifest['tasks'].append(item)
        print(json.dumps(item), flush=True)
    manifest['lerobot_commit'] = subprocess.check_output(['git','-C',str(ROOT/'lerobot'),'rev-parse','HEAD'],text=True).strip()
    manifest['packages'] = {p:importlib.metadata.version(p) for p in ['lerobot','torch','torchvision','torchcodec','transformers','hf-libero','huggingface-hub']}
    (ROOT/'results/manifest.json').write_text(json.dumps(manifest,indent=2))
    subprocess.run(['/venv/main/bin/python','-m','pip','freeze'],stdout=(ROOT/'results/requirements.txt').open('w'),check=False)
    print('Preparation complete',flush=True)


if __name__ == '__main__':
    main()
