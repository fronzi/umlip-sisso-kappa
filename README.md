# umlip-sisso-kappa

Reproducibility package for *"Interpretable descriptors of lattice thermal
conductivity from foundation machine-learning interatomic potentials: a
symbolic-regression workflow under systematic potential-energy-surface softening."*

The pipeline takes a curated list of crystalline insulators with reference
$\kappa_L(300\,\mathrm{K})$ and produces:

1. **Tier T0** — composition-only features (matminer / magpie).
2. **Tier T1** — harmonic uMLIP features: relaxed structure, equation-of-state
   bulk modulus $B_0,B'$, elastic tensor, sound velocities, phonon-DOS moments,
   Debye temperature $\Theta_\mathrm{D}$.
3. **Tier T2** — anharmonic uMLIP features: quasi-harmonic mode-Grüneisen
   parameters $\gamma_{qs}$ and their moments, Knoop anharmonicity score
   $\sigma^A$ from short NVT MD.
4. **Symbolic regression** (PySR; optional SISSO++) on each tier to discover
   a closed-form, low-dimensional descriptor of $\log_{10}\kappa_L$.
5. **Softening-bias analysis** by swapping uMLIP features for DFT references
   on a held-out subset.
6. **Figures** for the manuscript.

---

## Layout

```
umlip-sisso-kappa/
├── README.md
├── environment.yml                # pinned conda environment
├── data/
│   └── benchmark_seed.csv         # mp_id, formula, kappa_ref, ref_doi, ...
├── configs/
│   └── default.yaml               # all knobs in one place
├── src/umlip_kappa/
│   ├── __init__.py
│   ├── io_utils.py                # checkpoint, JSON/CSV I/O
│   ├── calculators.py             # uMLIP ASE-calculator factory
│   ├── dataset.py                 # benchmark curation
│   ├── features_T0.py             # composition (matminer)
│   ├── features_T1.py             # harmonic uMLIP features
│   ├── features_T2.py             # anharmonic uMLIP features
│   ├── slack.py                   # Slack baseline κ_L
│   ├── sr.py                      # PySR + SISSO wrappers
│   ├── validate.py                # softening-bias analysis
│   └── plots.py                   # publication figures
├── scripts/
│   ├── 01_assemble_dataset.py
│   ├── 02_compute_T0.py
│   ├── 03_compute_T1.py
│   ├── 04_compute_T2.py
│   ├── 05_run_sisso.py
│   ├── 06_validate_softening.py
│   └── 07_make_figures.py
└── notebooks/
    └── exploration.ipynb
```

---

## Installation

```bash
mamba env create -f environment.yml
mamba activate umlip-kappa
pip install -e .
# pick whichever uMLIPs you want; each one is independent
pip install mace-torch                  # MACE-MP-0
pip install orb-models                  # ORB-v3
pip install mattersim                   # MatterSim
# symbolic regression
pip install pysr                        # PySR (Julia-backed)
python -c "import pysr; pysr.install()" # one-time PySR/Julia setup
# optional: SISSO++
# build from https://gitlab.com/sissopp_developers/sissopp and put `sisso++` on PATH
```

A GPU is **strongly recommended** for T1/T2; CPU works for ~50 materials but
gets slow above that. Tested on Linux, CUDA 12.1, Python 3.11.

---

## Running the pipeline

```bash
# (~minutes, no compute)
python scripts/01_assemble_dataset.py --config configs/default.yaml

# (~minutes, no compute)
python scripts/02_compute_T0.py --config configs/default.yaml

# (~1–2 days on 1 GPU for N≈150, ×3 uMLIPs)
python scripts/03_compute_T1.py --config configs/default.yaml \
       --umlip mace-mp-0 --workers 4

# (~3–5 days on 1 GPU for N≈150, ×3 uMLIPs)
python scripts/04_compute_T2.py --config configs/default.yaml \
       --umlip mace-mp-0 --workers 4

# (~hours on 16 CPU cores)
python scripts/05_run_sisso.py --config configs/default.yaml \
       --tier T2 --umlip mace-mp-0 --max-complexity 3

# (~minutes)
python scripts/06_validate_softening.py --config configs/default.yaml
python scripts/07_make_figures.py --config configs/default.yaml
```

Every script is **checkpointed per-material**: re-launching a killed run
resumes from the last successful material. Intermediate artefacts live in
`outputs/<material_id>/{relax.cif, phonons.yaml, gruneisen.npz, sigmaA.txt}`.

---

## Reproducing the paper figures

```bash
make figures   # runs 07_make_figures.py with the production config
```

The figures are written to `figs/` and match the labelling in `main.tex`.

---

## Citing

If you use this code or the discovered descriptor, please cite the
manuscript and the underlying tools (PySR, SISSO, MACE-MP-0, MatterSim,
ORB, Phonopy, matminer).
