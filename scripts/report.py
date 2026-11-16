"""Build SR table only from completed evaluations; never fill missing trials with failures."""
import csv
import json
from pathlib import Path

ROOT = Path('/workspace/pi05-libero')


def build_report():
    manifest_path = ROOT/'results/manifest.json'
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    rows = []
    table = ['# Pi0.5: LIBERO-goal single-task SFT', '',
             'Each run starts independently from `lerobot/pi05_base`. Five evaluation attempts per checkpoint.',
             'SR = successful attempts / 5. Missing entries have not been evaluated.', '',
             '| Task | Step 250 | Step 500 | Step 750 | Step 1000 |',
             '|---|---:|---:|---:|---:|']
    completed = 0
    for task in manifest['tasks']:
        cells = []
        for step in [250,500,750,1000]:
            path = ROOT/'results'/task['slug']/f'{step:06d}'/'eval.json'
            if not path.exists():
                cells.append('pending')
                continue
            record = json.loads(path.read_text())
            data = record['evaluation']
            assert len(data['per_task']) == 1
            per_task = data['per_task'][0]
            assert per_task['task_group'] == 'libero_goal'
            assert per_task['task_id'] == task['libero_task_id']
            successes = per_task['metrics']['successes']
            assert len(successes) == 5
            assert all(isinstance(x,bool) for x in successes)
            n = sum(successes)
            assert data['overall']['n_success'] == n
            assert data['overall']['n_episodes'] == 5
            assert abs(data['overall']['pc_success'] - n*20) < 1e-5
            cells.append(f'{n}/5 ({n*20}%)')
            rows.append({'task':task['language'],'step':step,'n_success':n,'n_episodes':5,
                         'sr':n/5,'checkpoint_repo':task['repo_id'],
                         'checkpoint_path':record['checkpoint_path']})
            completed += 1
        table.append('| '+task['language']+' | '+' | '.join(cells)+' |')
    averages=[]
    for step in [250,500,750,1000]:
        selected=[r for r in rows if r['step']==step]
        averages.append(f'{sum(r["n_success"] for r in selected)}/15 ({sum(r["n_success"] for r in selected)/15*100:.1f}%)'
                        if len(selected)==3 else 'pending')
    table.append('| Overall across 3 tasks | '+' | '.join(averages)+' |')
    table += ['',f'Completed evaluations: {completed}/12 ({completed*5}/60 attempts).',
              '', 'Training: full SFT; FP32 parameters/optimizer, BF16 autocast; batch size 8; AdamW; '
              'LR 5e-5, warmup 50 steps, cosine decay to 5e-6 at step 1000; seed 1000.',
              'Evaluation: LIBERO-goal, task IDs 8/0/5, initial states 0–4, environment seeds 1000–1004, '
              'hard resets, 256×256 images, 10 actions per inference, 300-step limit.',
              'Dataset: lerobot/libero, filtered to the exact task. Shared full-dataset mean/std normalization.',
              '', 'Resumption: bowl_on_plate and middle_drawer were resumed from their step-250 training states '
              'during uploader orchestration changes; existing evaluations were retained. '
              'See resume-events.json in the experiment backup for details.',
              '', 'Five attempts give a coarse SR estimate with 20 percentage-point increments.']
    report = '\n'.join(table)+'\n'
    target = ROOT/'results/REPORT.md'
    if not target.exists() or target.read_text() != report:
        target.write_text(report)
    csv_path = ROOT/'results/success_rates.csv'
    with csv_path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['task','step','n_success','n_episodes','sr','checkpoint_repo','checkpoint_path'])
        writer.writeheader()
        writer.writerows(rows)
    (ROOT/'results/success_rates.json').write_text(json.dumps(rows,indent=2))
    return completed


if __name__ == '__main__':
    print('Completed evaluations:',build_report())
