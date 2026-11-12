from pathlib import Path
from unittest.mock import patch
import torch
from confdiff.train import run
import tempfile
root=Path(tempfile.mkdtemp(prefix='conformer-resume-check-'))
kwargs=dict(steps=60,batch=4,n_conf=2,n_samples=1,T=20)
run(**kwargs,ckpt=str(root/'full.pt'),out=str(root/'full.json'),samples_out=str(root/'full.npz'))
calls=0

def clock():
 global calls
 calls+=1
 return 0. if calls<=31 else 2.

with patch('confdiff.train.time.time',clock):
 assert run(**kwargs,ckpt=str(root/'split.pt'),out=str(root/'split.json'),samples_out=str(root/'split.npz'),max_seconds=1) is None
c=torch.load(root/'split.pt',weights_only=False);print('PAUSED',c['step']);assert c['step']==30
run(**kwargs,ckpt=str(root/'split.pt'),out=str(root/'split.json'),samples_out=str(root/'split.npz'))
a=torch.load(root/'full.pt',weights_only=False);b=torch.load(root/'split.pt',weights_only=False)
assert all(torch.equal(v,b['model'][k]) for k,v in a['model'].items())
assert a['sched']==b['sched']
import numpy as np,json
for k,v in np.load(root/'full.npz').items():
 assert np.array_equal(v,np.load(root/'split.npz')[k])
assert json.load(open(root/'full.json'))==json.load(open(root/'split.json'))
print('60step interrupted/resumed model, scheduler, evaluation coordinates and metrics identical')
