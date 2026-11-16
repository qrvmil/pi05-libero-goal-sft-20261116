"""Publish final measured SR table without exposing private machine diagnostics."""
import json
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd
from report import build_report

ROOT=Path('/workspace/pi05-libero')


def main():
    assert build_report()==12,'All 60 attempts must finish before publishing the comparison'
    api=HfApi()
    records=[]
    for task in json.loads((ROOT/'results/manifest.json').read_text())['tasks']:
        operations=[CommitOperationAdd(path_in_repo=name,path_or_fileobj=(ROOT/'results'/source).read_bytes())
                    for name,source in [('COMPARISON.md','REPORT.md'),('success_rates.csv','success_rates.csv'),
                                        ('success_rates.json','success_rates.json')]]
        commit=api.create_commit(repo_id=task['repo_id'],operations=operations,
                                 commit_message='Publish 60-attempt comparison of all three single-task SFT runs')
        records.append({'repo_id':task['repo_id'],'comparison_commit':commit.oid})
        print(task['repo_id'],commit.oid,flush=True)
    (ROOT/'results/comparison-publication.json').write_text(json.dumps(records,indent=2))


if __name__=='__main__':
    main()
