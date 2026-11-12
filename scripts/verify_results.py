import json, numpy as np, torch
from confdiff.data import build, pad_batch, SMILES, HELD_OUT
from confdiff.model import Denoiser, Diffusion
from confdiff import evaluate as ev
from rdkit import Chem
from pathlib import Path
r=json.load(open('results/metrics.json')); archive=np.load('results/samples.npz')
canonical=lambda s:Chem.MolToSmiles(Chem.MolFromSmiles(s))
assert not (set(map(canonical,HELD_OUT)) & set(map(canonical,[s for s in SMILES if s not in HELD_OUT])))
checks=0; maximum=0.
for group in ('train_molecules','held_out_molecules'):
 for rowidx,row in enumerate(r[group]['rows']):
  sm=row['smiles'];k=rowidx%8;m=build([sm],n_conf=8)[0]
  prefix=f'{group}/{sm}/'; refs=archive[prefix+'reference_conformers'];xyz=archive[prefix+'generated'][k];rnd=archive[prefix+'random_baseline'][k]
  np.testing.assert_allclose(refs,m.conf,atol=2e-6,rtol=0)
  vals=dict(bond_err_gen=ev.bond_error(m.mol,xyz),bond_err_random=ev.bond_error(m.mol,rnd),rmsd_gen=ev.best_rmsd(m.mol,refs,xyz),rmsd_random=ev.best_rmsd(m.mol,refs,rnd),e_gen=ev.mmff_energy(m.mol,xyz),e_gen_relaxed=ev.mmff_energy(m.mol,xyz,True),e_ref=ev.mmff_energy(m.mol,refs[0]),clash_heavy=ev.heavy_atom_clash(xyz,m.mol))
  for key,value in vals.items():
   np.testing.assert_allclose(value,row[key],atol=3e-5,rtol=2e-6);checks+=1
  # Independent orthogonal Procrustes allowing reflections (same fixed atom mapping).
  heavy=np.array([a.GetAtomicNum()>1 for a in m.mol.GetAtoms()]);x=xyz[heavy].astype(float);x-=x.mean(0)
  ds=[]
  for ref in refs:
   y=ref[heavy].astype(float);y-=y.mean(0);u,s,v=np.linalg.svd(x.T@y);ds.append(np.sqrt(np.mean(np.sum((x@(u@v)-y)**2,axis=1))))
  np.testing.assert_allclose(min(ds),row['rmsd_gen'],atol=2e-6,rtol=1e-5);checks+=1
  heavyids=np.where(heavy)[0];pairs=[(i,j) for i in heavyids for j in heavyids if i<j and m.mol.GetBondBetweenAtoms(int(i),int(j)) is None]
  assert any(np.linalg.norm(xyz[i]-xyz[j])<1.2 for i,j in pairs)==row['clash_heavy'];checks+=1
 print(group,'48 rows metrics independently recomputed')
print('CHECKS',checks)
# Recreate saved samples directly from final checkpoint under this environment.
c=torch.load('results/ckpt.pt',weights_only=False);assert c['step']==4000
model=Denoiser();model.load_state_dict(c['model']);model.eval();d=Diffusion();torch.manual_seed(12345)
for group,smiles in [('train_molecules',r['config']['train_molecules_evaluated']),('held_out_molecules',HELD_OUT)]:
 for sm in smiles:
  m=build([sm],n_conf=8)[0];b=pad_batch([m]*8,[m.conf[0]]*8)
  xyz=d.sample(model,b).numpy();noise=d.centre(torch.randn(8,m.n,3),b['mask']).numpy()*float(np.std(m.conf[0]))
  saved=archive[f'{group}/{sm}/generated'];delta=float(np.max(np.abs(xyz-saved)));maximum=max(maximum,delta)
  np.testing.assert_allclose(xyz,saved,rtol=2e-4,atol=2e-4)
  np.testing.assert_allclose(noise,archive[f'{group}/{sm}/random_baseline'],rtol=1e-6,atol=1e-6)
print('REGENERATED_CHECKPOINT_SAMPLES_MAX_ABS',maximum)
