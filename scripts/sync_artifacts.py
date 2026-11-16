"""Continuously back up experiment metadata, scripts, and console logs."""
import hashlib
import json
import time
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd

ROOT = Path('/workspace/pi05-libero')
REPO = 'qrvmil/pi05-libero-goal-sft-20261116'


def sync_once():
    if (ROOT/'scripts/report.py').exists():
        from report import build_report
        build_report()
    api = HfApi()
    state_path = ROOT / '.sync-state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if (ROOT/'scripts/publish_cards.py').exists():
        from publish_cards import publish_cards
        publish_cards(api, state)
        state_path.write_text(json.dumps(state))
    operations = []
    changed = {}
    paths = [ROOT / 'PLAN.md']
    for folder in ['scripts', 'results', 'logs', 'configs']:
        paths.extend(p for p in (ROOT / folder).rglob('*') if p.is_file() and '__pycache__' not in str(p))
    for path in paths:
        if path.suffix in ['.pyc']:
            continue
        data = path.read_bytes()
        key = str(path.relative_to(ROOT))
        digest = hashlib.sha256(data).hexdigest()
        if state.get(key) != digest:
            operations.append(CommitOperationAdd(path_in_repo=key, path_or_fileobj=data))
            changed[key] = digest
    if operations:
        commit = api.create_commit(repo_id=REPO, operations=operations, commit_message='Back up experiment progress')
        state.update(changed)
        state_path.write_text(json.dumps(state))
        print(f'Uploaded {len(operations)} artifacts: {commit.oid}', flush=True)


if __name__ == '__main__':
    while True:
        try:
            sync_once()
        except Exception as exc:
            print(f'Sync will retry: {type(exc).__name__}: {exc}', flush=True)
        time.sleep(30)
