# Simulation-based inference tutorial

An introduction to simulation-based inference (SBI) for astrophysics.

- **Slides:** <https://christopherlovell.github.io/sbi_tutorial/>: an interactive
  introduction to Bayesian inference, SBI and its NPE / NLE / NRE flavours.
- **Notebook:** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/christopherlovell/sbi_tutorial/blob/main/CAMELS_SBI_tutorial.ipynb)
  infers cosmological and feedback parameters from the galaxy stellar mass
  function and stellar–halo mass relation of the
  [CAMELS](https://camels.readthedocs.io) SIMBA or Swift-EAGLE LH suites, using
  [LtU-ILI](https://github.com/maho3/ltu-ili). It covers NPE, NLE, NRE and a
  comparison with emulator + MCMC.

## Contents

| Path | |
|---|---|
| `CAMELS_SBI_tutorial.ipynb` | Tutorial notebook |
| `data/camels_{SIMBA,Swift-EAGLE}_LH_z090.hdf5` | Precomputed summaries: parameters (with prior ranges), GSMF and SHMR for 1000 sims per suite |
| `scripts/download_camels.py` | Regenerates the summaries from the public CAMELS catalogues |
| `slides/` | [Slidev](https://sli.dev) source for the slides |

## Running locally

Notebook:

```bash
uv venv .venv
uv pip install --python .venv/bin/python \
    "ltu-ili[pytorch] @ git+https://github.com/maho3/ltu-ili.git" \
    corner h5py scikit-learn jupyterlab
.venv/bin/jupyter lab CAMELS_SBI_tutorial.ipynb
```

Skip the two Colab-only cells (`pip install` and `wget`).

Outputs are stripped from notebooks on commit by an
[nbstripout](https://github.com/kynan/nbstripout) git filter (see
`.gitattributes`), so you can save a run locally and GitHub stays clean. Set
it up once per clone:

```bash
uv tool install nbstripout   # or: pip install nbstripout
nbstripout --install
```

Slides:

```bash
cd slides && npm install && npm run dev
```

The slides are rebuilt and deployed to GitHub Pages on every push to `main`
that touches `slides/`.

Regenerating the data (about 3 min per suite of 1000 LH simulations):

```bash
python scripts/download_camels.py --suite SIMBA --out data/camels_SIMBA_LH_z090.hdf5
python scripts/download_camels.py --suite Swift-EAGLE --out data/camels_Swift-EAGLE_LH_z090.hdf5
```
