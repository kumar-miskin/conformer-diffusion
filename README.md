# conformer-diffusion

A small denoising diffusion model (DDPM) that generates 3D conformers of small molecules, conditioned on the molecular graph. RDKit builds the data and scores the output. The denoiser is an E(3)-equivariant graph network written in PyTorch.

This is a compact, CPU-only experiment, built to understand the mechanics of molecular diffusion and how generated geometries are checked. It is not a drug-discovery benchmark.

## What it does

- **Data (RDKit):** 34 small molecules (solvents, amino-acid-like fragments, aromatic rings, up to 24 atoms with hydrogens; list in `src/confdiff/data.py`). For each: hydrogens added, 8 conformers from ETKDG, MMFF94 optimised. 28 molecules are used for training, 6 are held out (split by molecule) (`HELD_OUT`).
- **Model:** EGNN-style noise predictor. Inputs are noisy coordinates, atom types, bond-type matrix and the diffusion timestep. The output is a predicted noise vector per atom, built from relative position vectors, so it rotates with the input and ignores translation.
- **Diffusion:** DDPM with a linear beta schedule (T=200), noise-prediction loss, ancestral sampling. Coordinates are centred (zero unweighted coordinate centroid) at every step. Random rotations are applied to training conformers.
- **Evaluation (RDKit):** bond-length error against the MMFF reference, best heavy-atom RMSD to the reference conformer set (mirror images allowed), MMFF energy of the raw sample, heavy-atom clash rate (definitions below). Each is compared to a random-coordinate baseline at the same scale.

## Results

4000 training steps, batch 32, seed 0, about 8 minutes on 2 CPU cores. 8 samples per molecule, 6 molecules per group (48 samples each). Bond error, RMSD and energy are medians over samples; clash rate is the fraction of samples with a clash. Raw per-sample numbers are in `results/metrics.json`; the generated coordinates, random baselines and reference conformers behind them are in `results/samples.npz` (keys `<group>/<smiles>/{generated,random_baseline,reference_conformers}`), and the trained weights are in `results/ckpt.pt`.

| | bond error (A) | heavy-atom RMSD (A) | heavy-atom clash rate |
|---|---|---|---|
| Training molecules (first 6 of 28), generated | 0.05 | 0.07 | 0.00 |
| Training molecules, random baseline | 1.14 | 0.87 | - |
| Held-out molecules (6), generated | 0.24 | 0.76 | 0.10 |
| Held-out molecules, random baseline | 1.82 | 1.75 | - |

Held-out median heavy-atom RMSD by molecule: toluene 0.72, ethyl acetate 0.72, glycerol 0.65, pyridine 0.57, acetanilide 1.38, methoxybenzene 0.83 (A).

Metric definitions: bond error is the mean absolute difference between generated and MMFF-reference bond lengths over all bonds (including to H). RMSD is the heavy-atom RMSD after optimal alignment (RDKit `AlignMol`), minimised over the 8 reference conformers and over a mirror image. The clash flag is set when two non-bonded heavy atoms (Z > 1) are closer than 1.2 A; hydrogens are ignored. The random baseline is Gaussian noise (centred) scaled to the standard deviation of the reference coordinates.

## Limitations (read these before drawing conclusions)

- On molecules seen in training the model essentially reproduces the stored conformers. That is memorisation of a very small set, not generation.
- The train/held-out split is molecule-disjoint (no molecule appears in both), not scaffold-disjoint: held-out molecules share rings and functional groups with training molecules.
- On held-out molecules geometry is clearly better than random but poor in absolute terms: median heavy-atom RMSD 0.76 A, acetanilide 1.38 A, 10% of samples have a heavy-atom clash, and the median MMFF energy of raw samples is about 1300 kcal/mol against about 17 for the reference. Samples need force-field relaxation before any use. (The relaxed energies in the JSON are close to the reference values. Similar energies alone do not establish identical geometry or the same basin, and relaxation is not evidence that the raw model is accurate.)
- 28 training molecules, one seed, no confidence intervals, no comparison to ETKDG sampling speed or quality, no chirality or torsion-specific metrics. There is no property prediction, no binding or docking, and no protein component.
- The model generates 3D geometry for a given molecular graph. It does not generate new molecular graphs.

## Run it

```
pip install -e ".[test]"
pytest -q                                   # 9 tests, a few seconds
python -m confdiff.train --steps 4000       # writes results/metrics.json, results/samples.npz, results/ckpt.pt
```

Environment used for the reported numbers: Python 3.10.12, PyTorch 2.14.1 (CPU), RDKit 2026.09.2, NumPy 2.2.6 (also recorded in `results/metrics.json`). Other library versions may give slightly different numbers.

Training can be paused and resumed with `--max-seconds N`. The checkpoint stores the optimiser, scheduler and all RNG states, and a paused-and-resumed run was checked to produce identical weights to an uninterrupted one (60-step check). The reported run was made this way, in four sessions (pauses after steps 970, 1998 and 3023, then a final session that finished and evaluated).

Tests cover: RDKit graph and centring, rotation equivariance of the denoiser, translation-free output, padding invariance, exactly-zero output on padded atoms, loss unaffected by padded slots, forward noising schedule, loss decrease on a tiny run, RDKit metric sanity, and that the clash metric ignores hydrogens but catches non-bonded heavy-atom overlaps.

## Provenance

All code in this repository was written for this project (model, schedule and sampler follow the standard DDPM formulation of Ho et al. 2020, and the layer design follows the EGNN of Satorras et al. 2021). No external datasets or pretrained weights are used; all conformers are generated locally by RDKit from the SMILES in the source. Development was AI-assisted.

## Possible next steps

Larger and more drug-like molecule sets (e.g. a QM9 or GEOM subset), torsional diffusion, multiple seeds, and comparison to ETKDG on coverage and matching metrics.

## Independent verification

From the repository root after installing the package:

```bash
OMP_NUM_THREADS=2 python scripts/verify_results.py
OMP_NUM_THREADS=2 python scripts/verify_resume.py
```

The first script recomputes all 96 sample rows, including 960 metric/oracle checks, compares RMSD to an independent SVD alignment allowing reflections, checks molecule-disjointness and reconstructs the saved generated and random coordinates from the final checkpoint. A separate Python 3.12.15 / PyTorch 2.15.0+cpu / RDKit 2026.09.2 / NumPy 2.5.4 environment reproduced the saved coordinates exactly. The second script runs 60 training steps continuously and with a pause after step 30, verifying identical final model weights, scheduler, evaluation coordinates and metrics. These are reproducibility checks, not a second full training run or a multi-seed benchmark.

The checkpoint contains Python/NumPy RNG state and is loaded with `weights_only=False`. Only load trusted project checkpoints; do not use arbitrary downloaded checkpoint files with this training or audit code. The default training command resumes the included completed checkpoint, so it regenerates evaluation rather than retraining. To train from scratch, move `results/ckpt.pt` aside first or call `run(ckpt=...)` with a fresh path. Keep the published evidence if you want to compare your rerun.
