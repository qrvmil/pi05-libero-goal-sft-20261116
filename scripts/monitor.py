import json
import re
import shutil
from pathlib import Path

root=Path('/workspace/pi05-libero')
manifest=json.loads((root/'results/manifest.json').read_text())
for t in manifest['tasks']:
    log=root/'logs'/f'{t["slug"]}-train.log'
    matches=re.findall(r'step:([0-9.]+[KM]?).*?loss:([0-9.eE+-]+)',log.read_text() if log.exists() else '')
    step,loss=matches[-1] if matches else ('0','pending')
    step=int(float(step.rstrip('KM'))*({'K':1000,'M':1000000}.get(step[-1],1)))
    scores={}
    for result in sorted((root/'results'/t['slug']).glob('*/eval.json')):
        r=json.loads(result.read_text());scores[r['step']]=r['evaluation']['overall']['n_success']
    uploaded=[]
    confirmed={int(p.name.split('-')[0]) for p in (root/'results'/t['slug']).glob('*-upload.json')}
    verification=root/'results'/f'{t["slug"]}-remote-verification.json'
    if verification.exists() and json.loads(verification.read_text()).get('all_four_checkpoints_verified'):
        confirmed.update([250,500,750,1000])
    for cp in sorted((root/'outputs'/t['slug']/'checkpoints').glob('[0-9]*')):
        if (cp/'.uploaded').exists():uploaded.append(int(cp.name))
    print(json.dumps({'task':t['slug'],'step':step,'loss':loss,'eval_successes_out_of_5':scores,
                      'uploaded_local_checkpoints':uploaded,
                      'confirmed_remote_steps':sorted(confirmed),
                      'training_finished':(root/'results'/f'{t["slug"]}-training-finished.json').exists(),
                      'completed':(root/'results'/f'{t["slug"]}-completed.json').exists()}))
print(json.dumps({'disk_free_GB':round(shutil.disk_usage(root).free/1e9,1),
                  'network_tx_GB':round(int(Path('/sys/class/net/eth0/statistics/tx_bytes').read_text())/1e9,1)}))
