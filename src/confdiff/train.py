"""Train a DDPM on RDKit conformers of the training molecules; evaluate on seen and held-out molecules.

Training can be paused (--max-seconds) and resumed. The checkpoint stores the optimiser, scheduler and all RNG
states, so a resumed run is identical to an uninterrupted one on the same machine/library versions.
"""
from __future__ import annotations

import argparse, json, os, platform, random, time
import numpy as np
import torch
import rdkit

from .data import SMILES, HELD_OUT, build, pad_batch
from .model import Denoiser, Diffusion
from . import evaluate as ev


def _rng_state():
    return dict(py=random.getstate(), np=np.random.get_state(), torch=torch.get_rng_state())


def _set_rng_state(s):
    random.setstate(s["py"]); np.random.set_state(s["np"]); torch.set_rng_state(s["torch"])


def run(steps=4000, batch=32, n_conf=8, n_samples=8, seed=0, out="results/metrics.json", T=200,
        ckpt="results/ckpt.pt", samples_out="results/samples.npz", max_seconds=None):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    train_s = [s for s in SMILES if s not in HELD_OUT]
    train = build(train_s, n_conf, seed)
    held = build(HELD_OUT, n_conf, seed)
    model, diff = Denoiser(), Diffusion(T=T)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t0, log, start = time.time(), [], 1
    if ckpt and os.path.exists(ckpt):
        c = torch.load(ckpt, weights_only=False)
        model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"])
        start, log = c["step"] + 1, c["log"]
        _set_rng_state(c["rng"])
    for step in range(start, steps + 1):
        if max_seconds and time.time() - t0 > max_seconds:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                            step=step - 1, log=log, rng=_rng_state()), ckpt)
            print(f"paused at step {step-1}"); return None
        mols = [random.choice(train) for _ in range(batch)]
        coords = []
        for m in mols:
            x = m.conf[random.randrange(len(m.conf))]
            q, _ = np.linalg.qr(np.random.randn(3, 3))  # random orthogonal augmentation
            coords.append((x @ q).astype(np.float32))
        loss = diff.loss(model, pad_batch(mols, coords))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 250 == 0:
            log.append((step, float(loss.detach())))
            print(f"step {step} loss {float(loss.detach()):.4f}", flush=True)
    if ckpt:
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        step=steps, log=log, rng=_rng_state()), ckpt)

    model.eval()
    torch.manual_seed(seed + 12345)  # evaluation noise is seeded separately from training
    result = {"config": dict(steps=steps, batch=batch, n_conf=n_conf, T=T, seed=seed, n_samples_per_molecule=n_samples,
                             n_train_mols=len(train), n_heldout_mols=len(held),
                             train_molecules_evaluated=[m.smiles for m in train[:6]],
                             heldout_molecules=HELD_OUT),
              "versions": dict(python=platform.python_version(), torch=torch.__version__,
                               rdkit=rdkit.__version__, numpy=np.__version__),
              "train_loss_log": log}
    saved = {}
    for name, mols in (("train_molecules", train[:6]), ("held_out_molecules", held)):
        rows = []
        for m in mols:
            batch_m = [m] * n_samples
            pb = pad_batch(batch_m, [m.conf[0]] * n_samples)
            gen = diff.sample(model, pb)
            noise = Diffusion.centre(torch.randn(n_samples, gen.shape[1], 3), pb["mask"])
            saved[f"{name}/{m.smiles}/generated"] = gen[:, :m.n].numpy()
            saved[f"{name}/{m.smiles}/random_baseline"] = (noise[:, :m.n] * float(np.std(m.conf[0]))).numpy()
            saved[f"{name}/{m.smiles}/reference_conformers"] = m.conf
            for k in range(n_samples):
                xyz = gen[k, :m.n].numpy()
                rnd = saved[f"{name}/{m.smiles}/random_baseline"][k]
                rows.append(dict(
                    smiles=m.smiles,
                    bond_err_gen=ev.bond_error(m.mol, xyz), bond_err_random=ev.bond_error(m.mol, rnd),
                    rmsd_gen=ev.best_rmsd(m.mol, m.conf, xyz), rmsd_random=ev.best_rmsd(m.mol, m.conf, rnd),
                    e_gen=ev.mmff_energy(m.mol, xyz), e_gen_relaxed=ev.mmff_energy(m.mol, xyz, True),
                    e_ref=ev.mmff_energy(m.mol, m.conf[0]),
                    clash_heavy=ev.heavy_atom_clash(xyz, m.mol)))
        agg = {k: float(np.nanmedian([r[k] for r in rows])) for k in rows[0] if k not in ("smiles", "clash_heavy")}
        agg["heavy_atom_clash_rate"] = float(np.mean([r["clash_heavy"] for r in rows]))
        agg["n_samples"] = len(rows)
        result[name] = dict(summary=agg, rows=rows)
    if samples_out:
        np.savez_compressed(samples_out, **saved)
    json.dump(result, open(out, "w"), indent=1)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--out", default="results/metrics.json")
    ap.add_argument("--max-seconds", type=float, default=None)
    a = ap.parse_args()
    r = run(steps=a.steps, out=a.out, max_seconds=a.max_seconds)
    if r is None:
        raise SystemExit(0)
    for k in ("train_molecules", "held_out_molecules"):
        print(k, json.dumps(r[k]["summary"], indent=1))
