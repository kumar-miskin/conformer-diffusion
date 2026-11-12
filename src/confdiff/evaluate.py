"""RDKit-based checks for generated conformers."""
from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign
from rdkit.Geometry import Point3D


def with_coords(mol, xyz):
    m = Chem.Mol(mol)
    m.RemoveAllConformers()
    conf = Chem.Conformer(m.GetNumAtoms())
    for i, p in enumerate(np.asarray(xyz, dtype=float)):
        conf.SetAtomPosition(i, Point3D(*p))
    m.AddConformer(conf, assignId=True)
    return m


def bond_error(mol, xyz):
    """Mean |d - d_ref| over bonds, where d_ref is the MMFF-optimised bond length of the reference."""
    ref = mol.GetConformer(0).GetPositions()
    errs = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        errs.append(abs(np.linalg.norm(xyz[i] - xyz[j]) - np.linalg.norm(ref[i] - ref[j])))
    return float(np.mean(errs))


def mmff_energy(mol, xyz, relax=False):
    m = with_coords(mol, xyz)
    props = AllChem.MMFFGetMoleculeProperties(m)
    if props is None:
        return float("nan")
    ff = AllChem.MMFFGetMoleculeForceField(m, props)
    if relax:
        ff.Minimize(maxIts=2000)
    return float(ff.CalcEnergy())


def best_rmsd(mol, ref_confs, xyz):
    """Heavy-atom RMSD (Kabsch, also mirror-aware via both enantiomeric images) to closest reference conformer."""
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    best = np.inf
    for sign in (1.0, -1.0):
        probe = with_coords(mol, np.asarray(xyz) * np.array([sign, 1, 1]))
        for r in ref_confs:
            ref = with_coords(mol, r)
            amap = [(i, i) for i in heavy]
            best = min(best, rdMolAlign.AlignMol(probe, ref, atomMap=amap))
    return float(best)


def heavy_atom_clash(xyz, mol, cutoff=1.2):
    """True if any two non-bonded HEAVY atoms (Z > 1) are closer than `cutoff` angstrom. Hydrogens are ignored."""
    heavy = np.array([a.GetAtomicNum() > 1 for a in mol.GetAtoms()])
    xyz = np.asarray(xyz)
    d = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    skip = np.zeros_like(d, dtype=bool)
    for b in mol.GetBonds():
        skip[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] = skip[b.GetEndAtomIdx(), b.GetBeginAtomIdx()] = True
    skip |= ~(heavy[:, None] & heavy[None])
    np.fill_diagonal(skip, True)
    return bool((d[~skip] < cutoff).any())
