import numpy as np
import torch
from confdiff.data import build, pad_batch
from confdiff.model import Denoiser, Diffusion
from confdiff import evaluate as ev

MOLS = build(["CCO", "c1ccccc1"], n_conf=2)


def rot():
    q, _ = np.linalg.qr(np.random.randn(3, 3))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return torch.tensor(q, dtype=torch.float32)


def test_rdkit_graph_and_centering():
    m = MOLS[0]
    assert m.n == 9 and m.conf.shape == (2, 9, 3)
    assert np.allclose(m.conf.mean(1), 0, atol=1e-5)
    assert (m.bonds == m.bonds.T).all() and (m.bonds > 0).sum() == 2 * m.mol.GetNumBonds()


def test_denoiser_is_rotation_equivariant_and_translation_free():
    torch.manual_seed(0)
    model = Denoiser(hidden=32, layers=2).eval()
    b = pad_batch(MOLS, [m.conf[0] for m in MOLS])
    t = torch.tensor([10, 50])
    R = rot()
    out = model(b["x"], t, b["z"], b["bond"], b["mask"])
    out_r = model(b["x"] @ R.T, t, b["z"], b["bond"], b["mask"])
    assert torch.allclose(out @ R.T, out_r, atol=1e-4)
    assert torch.allclose((out * b["mask"][..., None]).sum(1), torch.zeros(2, 3), atol=1e-4)


def test_padding_does_not_change_output():
    torch.manual_seed(0)
    model = Denoiser(hidden=32, layers=2).eval()
    alone = pad_batch(MOLS[:1], [MOLS[0].conf[0]])
    both = pad_batch(MOLS, [m.conf[0] for m in MOLS])
    t = torch.tensor([20])
    a = model(alone["x"], t, alone["z"], alone["bond"], alone["mask"])
    c = model(both["x"], torch.tensor([20, 20]), both["z"], both["bond"], both["mask"])
    assert torch.allclose(a[0], c[0, :a.shape[1]], atol=1e-4)


def test_forward_noising_schedule():
    d = Diffusion(T=200)
    assert (d.abar[1:] < d.abar[:-1]).all() and d.abar[-1] < 0.05
    b = pad_batch(MOLS, [m.conf[0] for m in MOLS])
    xt, eps = d.noise(b["x"], torch.tensor([0, 0]), b["mask"])
    assert (xt - b["x"]).abs().max() < 0.2          # t=0 barely noised
    assert (xt.sum(1).abs() < 1e-4).all()           # stays centred


def test_loss_decreases_on_tiny_run():
    torch.manual_seed(0)
    model, d = Denoiser(hidden=32, layers=2), Diffusion(T=50)
    b = pad_batch(MOLS, [m.conf[0] for m in MOLS])
    opt = torch.optim.Adam(model.parameters(), 3e-3)
    first = np.mean([float(d.loss(model, b).detach()) for _ in range(5)])
    for _ in range(150):
        l = d.loss(model, b); opt.zero_grad(); l.backward(); opt.step()
    last = np.mean([float(d.loss(model, b).detach()) for _ in range(5)])
    assert last < 0.8 * first


def test_rdkit_metrics_sane():
    m = MOLS[0]
    ref = m.conf[0]
    assert ev.bond_error(m.mol, ref) < 1e-4
    assert ev.best_rmsd(m.mol, m.conf, ref) < 1e-3
    rng = np.random.default_rng(0)
    assert ev.bond_error(m.mol, rng.normal(size=ref.shape)) > 0.3
    assert ev.mmff_energy(m.mol, rng.normal(size=ref.shape) * 1.0) > ev.mmff_energy(m.mol, ref)


def test_padded_output_is_exactly_zero():
    torch.manual_seed(0)
    model = Denoiser(hidden=32, layers=2).eval()
    b = pad_batch(MOLS, [m.conf[0] for m in MOLS])   # CCO (9 atoms) is padded to benzene's 12
    out = model(b["x"], torch.tensor([30, 30]), b["z"], b["bond"], b["mask"])
    assert (out[~b["mask"]] == 0).all()


def test_loss_ignores_padding():
    torch.manual_seed(0)
    model, d = Denoiser(hidden=32, layers=2), Diffusion(T=50)
    b = pad_batch(MOLS, [m.conf[0] for m in MOLS])
    torch.manual_seed(1); l1 = d.loss(model, b)
    b2 = {k: v.clone() for k, v in b.items()}
    b2["x"][~b2["mask"]] = 5.0                     # garbage in padded slots must not matter
    torch.manual_seed(1); l2 = d.loss(model, b2)
    assert torch.allclose(l1, l2, atol=1e-5)


def test_clash_metric_is_heavy_only():
    from rdkit import Chem
    m = MOLS[0]                                      # ethanol
    ref = m.conf[0].copy()
    assert not ev.heavy_atom_clash(ref, m.mol)
    h = [a.GetIdx() for a in m.mol.GetAtoms() if a.GetAtomicNum() == 1]
    bad_h = ref.copy(); bad_h[h[0]] = bad_h[h[1]]   # two hydrogens on top of each other
    assert not ev.heavy_atom_clash(bad_h, m.mol)      # hydrogens ignored
    heavy = [a.GetIdx() for a in m.mol.GetAtoms() if a.GetAtomicNum() > 1]
    bad = ref.copy(); bad[heavy[0]] = bad[heavy[2]] + 0.3   # C and O far in the graph, overlapped
    assert ev.heavy_atom_clash(bad, m.mol)
