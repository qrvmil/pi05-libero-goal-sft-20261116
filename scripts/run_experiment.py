"""Sequential independent jobs on one GPU. Supervisor restarts from saved states."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/workspace/pi05-libero')
sys.path.insert(0, str(ROOT/'scripts'))
from sync_artifacts import sync_once


def main():
    manifest = json.loads((ROOT/'results/manifest.json').read_text())
    running=[]
    for task in manifest['tasks']:
        slug = task['slug']
        if (ROOT/'results'/f'{slug}-completed.json').exists():
            continue
        print('STARTING',slug,flush=True)
        (ROOT/'results'/f'{slug}-training-finished.json').unlink(missing_ok=True)
        (ROOT/'results/queue-status.json').write_text(json.dumps({'status':'running','task':slug},indent=2))
        # Checkpoints are durable before evaluations start. If interrupted between them,
        # finish that evaluation before resuming training at the next step.
        for checkpoint in sorted((ROOT/'outputs'/slug/'checkpoints').glob('[0-9]*')):
            if (checkpoint/'.uploaded').exists() and not (checkpoint/'.evaluated').exists():
                subprocess.run([sys.executable,'-u',str(ROOT/'scripts/eval_checkpoint.py'),slug,str(checkpoint)],check=True)
        with (ROOT/'logs'/f'{slug}-train.log').open('a',buffering=1) as log:
            process = subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/train_task.py'),slug],
                                       stdout=log,stderr=subprocess.STDOUT)
        running.append((slug,process))
        while process.poll() is None and not (ROOT/'results'/f'{slug}-training-finished.json').exists():
            time.sleep(5)
        if process.poll() not in (None,0):
            (ROOT/'results/queue-status.json').write_text(json.dumps({'status':'error',
                 'task':slug,'returncode':process.returncode},indent=2))
            sync_once()
            raise SystemExit(process.returncode)
    for slug,process in running:
        code=process.wait()
        if code:
            raise SystemExit(code)
    print('ALL_TRAINING_AND_EVALUATIONS_COMPLETED',flush=True)
    (ROOT/'results/queue-status.json').write_text(json.dumps({'status':'completed'}))
    sync_once()


if __name__ == '__main__':
    main()
