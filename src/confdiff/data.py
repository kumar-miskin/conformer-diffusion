"""RDKit side: SMILES -> graph + reference conformers (ETKDG + MMFF), padded tensors."""
from __future__ import annotations

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

ELEMENTS = [1, 6, 7, 8, 9, 16, 17]  # H C N O F S Cl
BOND_TYPES = {None: 0, Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2,
              Chem.BondType.TRIPLE: 3, Chem.BondType.AROMATIC: 4}
N_BOND = len(BOND_TYPES)

# Small drug-like / solvent-like molecules (<= ~22 atoms incl. H).
SMILES = [
    "CCO", "CC(=O)O", "CCN", "CC(C)O", "CCCO", "CC(=O)N", "NCC(=O)O", "CC(N)C(=O)O",
    "c1ccccc1", "Cc1ccccc1", "Oc1ccccc1", "Nc1ccccc1", "c1ccncc1", "c1ccoc1", "c1ccsc1",
    "CC(=O)Nc1ccccc1", "CCOC(C)=O", "CC(C)=O", "CCCC(=O)O", "OCC(O)CO", "CCOCC", "C1CCOC1",
    "C1CCNCC1", "CC#N", "ClCCO", "FC(F)c1ccccc1", "CS(C)=O", "O=C1CCCN1", "c1ccc2ccccc2c1",
    "CC(C)Cc1ccccc1", "COc1ccccc1", "OC(=O)c1ccccc1", "CCSC", "NC(=O)c1ccccn1",
]
HELD_OUT = ["Cc1ccccc1", "CCOC(C)=O", "OCC(O)CO", "c1ccncc1", "CC(=O)Nc1ccccc1", "COc1ccccc1"]


class Molecule:
    def __init__(self, smiles: str, n_conf: int = 8, seed: int = 0):
        self.smiles = smiles
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_conf, randomSeed=seed))
        if not cids:
            raise ValueError(f"embedding failed for {smiles}")
        AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=500)
        self.mol = mol
        self.z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
        self.n = mol.GetNumAtoms()
        self.bonds = np.zeros((self.n, self.n), dtype=np.int64)
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            self.bonds[i, j] = self.bonds[j, i] = BOND_TYPES[b.GetBondType()]
        xyz = np.stack([mol.GetConformer(c).GetPositions() for c in cids]).astype(np.float32)
        self.conf = xyz - xyz.mean(axis=1, keepdims=True)  # centred


def build(smiles_list, n_conf=8, seed=0):
    return [Molecule(s, n_conf, seed) for s in smiles_list]


def pad_batch(mols, coords=None, device="cpu"):
    """coords: list of (n,3) arrays aligned with mols; returns dict of padded tensors."""
    nmax = max(m.n for m in mols)
    B = len(mols)
    z = torch.zeros(B, nmax, dtype=torch.long)       # 0 = pad
    bond = torch.zeros(B, nmax, nmax, dtype=torch.long)
    mask = torch.zeros(B, nmax, dtype=torch.bool)
    x = torch.zeros(B, nmax, 3)
    idx = {e: i + 1 for i, e in enumerate(ELEMENTS)}
    for b, m in enumerate(mols):
        z[b, :m.n] = torch.tensor([idx[int(a)] for a in m.z])
        bond[b, :m.n, :m.n] = torch.from_numpy(m.bonds)
        mask[b, :m.n] = True
        if coords is not None:
            x[b, :m.n] = torch.as_tensor(coords[b])
    return {k: v.to(device) for k, v in dict(z=z, bond=bond, mask=mask, x=x).items()}
