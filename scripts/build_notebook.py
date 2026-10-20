"""Build CAMELS_SBI_tutorial.ipynb from the cell list below.

Edit the cells here and re-run, rather than editing the .ipynb by hand.
Cells tagged "colab" (installs, downloads) can be skipped when testing locally.
"""

import sys

import nbformat as nbf

REPO_RAW = "https://raw.githubusercontent.com/christopherlovell/sbi_tutorial/main"

cells = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src.strip()))


def code(src, colab=False):
    c = nbf.v4.new_code_cell(src.strip())
    if colab:
        c.metadata["tags"] = ["colab"]
    cells.append(c)


# ---------------------------------------------------------------------------
md(r"""
# Simulation-based inference with CAMELS

In this notebook we use **simulation-based inference** (SBI) to constrain cosmological and galaxy formation parameters from two simple galaxy population statistics: the **galaxy stellar mass function** (GSMF) and the **stellar-to-halo mass relation** (SHMR).

Our "simulator" is the [CAMELS](https://camels.readthedocs.io) suite of cosmological hydrodynamic simulations. The *Latin Hypercube* (LH) sets contain 1000 simulations, each of a $(25 \, h^{-1}\,\mathrm{Mpc})^3$ volume, run with a different combination of six parameters:

| Parameter | Description | LH range |
|---|---|---|
| $\Omega_m$ | matter density | 0.1 – 0.5 |
| $\sigma_8$ | amplitude of matter fluctuations | 0.6 – 1.0 |
| $A_{\rm SN1}$, $A_{\rm SN2}$ | supernova feedback (energy / wind speed) | 0.25 – 4, 0.5 – 2 |
| $A_{\rm AGN1}$, $A_{\rm AGN2}$ | AGN feedback (energy / ejection speed or burstiness) | 0.25 – 4, 0.5 – 2 |

Each simulation also has a different random seed for its initial conditions, so even at fixed parameters the GSMF would vary (*cosmic variance*). This stochasticity is exactly what SBI is designed to handle: rather than writing down a likelihood $p(x \,|\, \theta)$, we learn it (or the posterior $p(\theta \,|\, x)$ directly) from pairs of simulated parameters $\theta$ and data $x$.

We will use the [LtU-ILI](https://github.com/maho3/ltu-ili) package to demonstrate three flavours of neural SBI:

* **NPE** – Neural Posterior Estimation: learn $p(\theta \,|\, x)$ directly.
* **NLE** – Neural Likelihood Estimation: learn $p(x \,|\, \theta)$, then sample the posterior with MCMC.
* **NRE** – Neural Ratio Estimation: learn the ratio $p(x \,|\, \theta) / p(x)$ with a classifier, then sample with MCMC.

Finally, we compare to a "traditional" approach – an emulator plus a Gaussian likelihood, sampled with MCMC – to highlight what SBI, and NPE in particular, buys us.
""")

md(r"""
## Installation

LtU-ILI is installed directly from GitHub with its PyTorch backend (which pulls in [`sbi`](https://github.com/sbi-dev/sbi)). We also install [`corner`](https://corner.readthedocs.io) for plotting posteriors. `emcee` comes with LtU-ILI.
""")

code(r"""
!pip install -q "ltu-ili[pytorch] @ git+https://github.com/maho3/ltu-ili.git" corner
""", colab=True)

md(r"""
## Getting the data

Running CAMELS is (very!) expensive, so here we use the publicly available outputs. The script [`scripts/download_camels.py`](%s/scripts/download_camels.py) reads just three arrays from the z = 0 Subfind group catalogue of each LH simulation (subhalo stellar masses, halo masses $M_{200c}$, and the index of each halo's central galaxy), and reduces them immediately to

* the **GSMF**, $\phi$ [Mpc$^{-3}$ dex$^{-1}$], in bins of $\log_{10} M_\star$, and
* the **SHMR**, the median $\log_{10} M_\star$ of central galaxies in bins of $\log_{10} M_{200c}$.

This only needs a few MB per simulation thanks to HTTP range requests, and takes a few minutes for the full LH set. To save time we download the precomputed summaries; uncomment the second line to regenerate them yourself.
""" % REPO_RAW)

code(r"""
SUITE = "SIMBA"  # "Swift-EAGLE" once its LH catalogues are public
DATA_FILE = f"data/camels_{SUITE}_LH_z090.hdf5"
""")

code(r"""
!wget -q -nc -P data %s/{DATA_FILE}
# !wget -q -nc %s/scripts/download_camels.py && mkdir -p data && python download_camels.py --suite {SUITE} --out {DATA_FILE}
""" % (REPO_RAW, REPO_RAW), colab=True)

code(r"""
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

with h5py.File(DATA_FILE, "r") as h:
    params = h["params"][:]
    param_names = list(h["params"].attrs["names"])
    gsmf = h["gsmf"][:]
    gsmf_edges = h["gsmf"].attrs["bin_edges"]
    shmr = h["shmr"][:]
    shmr_edges = h["shmr"].attrs["bin_edges"]

gsmf_bins = 0.5 * (gsmf_edges[1:] + gsmf_edges[:-1])
shmr_bins = 0.5 * (shmr_edges[1:] + shmr_edges[:-1])

print(param_names)
print("params:", params.shape, " gsmf:", gsmf.shape, " shmr:", shmr.shape)
""")

# ---------------------------------------------------------------------------
md(r"""
## Exploring the simulations

Let's look at how the GSMF and SHMR vary across the Latin hypercube. Below we colour each simulation by the value of one of its parameters; try changing `ip` to see which parameters each statistic is sensitive to.
""")

code(r"""
def plot_stats(ip, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    cmap = plt.get_cmap("viridis")
    p = params[:, ip]
    c = cmap((p - p.min()) / (p.max() - p.min()))

    for i in range(len(params)):
        with np.errstate(divide="ignore"):  # empty bins
            ax[0].plot(gsmf_bins, np.log10(gsmf[i]), color=c[i], alpha=0.3, lw=0.8)
        ax[1].plot(shmr_bins, shmr[i], color=c[i], alpha=0.3, lw=0.8)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(p.min(), p.max()))
    plt.colorbar(sm, ax=ax[1], label=param_names[ip])
    ax[0].set_xlabel(r"$\log_{10}(M_\star \,/\, \mathrm{M_\odot})$")
    ax[0].set_ylabel(r"$\log_{10}(\phi \,/\, \mathrm{Mpc^{-3}\,dex^{-1}})$")
    ax[1].set_xlabel(r"$\log_{10}(M_{200c} \,/\, \mathrm{M_\odot})$")
    ax[1].set_ylabel(r"median $\log_{10}(M_{\star,\rm cen} \,/\, \mathrm{M_\odot})$")
    return ax

ip = 0  # 0: Omega_m, 1: sigma_8, 2: A_SN1, 3: A_AGN1, 4: A_SN2, 5: A_AGN2
plot_stats(ip)
plt.show()
""")

code(r"""
plot_stats(2)  # A_SN1
plt.show()
""")

md(r"""
Note the jagged high-mass end of the GSMF: in a $(25\,h^{-1}\,\mathrm{Mpc})^3$ box there are only a handful of massive galaxies, and some bins are empty ($\phi = 0$). Similarly, the most massive halo-mass bins of the SHMR are empty in some simulations (these are stored as `NaN`).
""")

# ---------------------------------------------------------------------------
md(r"""
## Building the data and parameter vectors

Neural density estimators work best with well-behaved inputs, so we

* take $\log_{10}\phi$ for the GSMF, replacing empty bins with a floor value corresponding to half a galaxy per bin,
* keep only the SHMR halo-mass bins that are populated in *every* simulation,
* concatenate the two into a single data vector $x$.

For the parameters $\theta$, the feedback amplitudes were sampled uniformly in $\log A$, so we infer $\log_{10} A$; this makes the prior a simple uniform box.
""")

code(r"""
h = 0.6711  # fixed in CAMELS
volume = (25 / h) ** 3  # Mpc^3
dlogm = np.diff(gsmf_edges)[0]
phi_floor = 0.5 / volume / dlogm

log_phi = np.log10(np.maximum(gsmf, phi_floor))
shmr_keep = ~np.isnan(shmr).any(axis=0)
print("SHMR bins kept:", shmr_bins[shmr_keep])

x_all = np.hstack([log_phi, shmr[:, shmr_keep]]).astype(np.float32)

theta_all = params.copy()
theta_all[:, 2:] = np.log10(theta_all[:, 2:])
theta_all = theta_all.astype(np.float32)
labels = [r"$\Omega_m$", r"$\sigma_8$", r"$\log A_{\rm SN1}$",
          r"$\log A_{\rm AGN1}$", r"$\log A_{\rm SN2}$", r"$\log A_{\rm AGN2}$"]

print("x:", x_all.shape, " theta:", theta_all.shape)
""")

md(r"""
We hold back 100 simulations as a **test set**, which the networks never see during training. We'll use these to validate our posteriors.
""")

code(r"""
rng = np.random.default_rng(42)
idx = rng.permutation(len(x_all))
test, train = idx[:100], idx[100:]

x_train, theta_train = x_all[train], theta_all[train]
x_test, theta_test = x_all[test], theta_all[test]
""")

md(r"""
Our **prior** is uniform over the region spanned by the Latin hypercube.
""")

code(r"""
import ili
from ili.dataloaders import NumpyLoader
from ili.inference import InferenceRunner
from ili.validation.metrics import PlotSinglePosterior, PosteriorCoverage

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

prior_low = np.array([0.1, 0.6, np.log10(0.25), np.log10(0.25), np.log10(0.5), np.log10(0.5)])
prior_high = np.array([0.5, 1.0, np.log10(4.0), np.log10(4.0), np.log10(2.0), np.log10(2.0)])
assert np.all(theta_all >= prior_low - 1e-4) and np.all(theta_all <= prior_high + 1e-4)

prior = ili.utils.Uniform(low=prior_low, high=prior_high, device=device)
loader = NumpyLoader(x=x_train, theta=theta_train)
""")

# ---------------------------------------------------------------------------
md(r"""
## Neural Posterior Estimation (NPE)

NPE trains a conditional density estimator $q_\phi(\theta \,|\, x)$ by maximising $\sum_i \log q_\phi(\theta_i \,|\, x_i)$ over the training pairs. Here we use an ensemble of two **Masked Autoregressive Flows** (MAFs). LtU-ILI combines ensemble members weighted by their validation loss.

Once trained, the posterior is **amortised**: we can sample it for *any* new $x$ with a single forward pass, no further training or MCMC needed.
""")

code(r"""
train_args = {"training_batch_size": 32, "learning_rate": 5e-4, "stop_after_epochs": 20}

nets_npe = [
    ili.utils.load_nde_sbi(engine="NPE", model="maf", hidden_features=50, num_transforms=5, repeats=2),
]

runner_npe = InferenceRunner.load(
    backend="sbi", engine="NPE", prior=prior, nets=nets_npe,
    device=device, train_args=train_args, out_dir=None,
)

t0 = time.perf_counter()
posterior_npe, summaries_npe = runner_npe(loader=loader)
print(f"NPE training: {time.perf_counter() - t0:.1f} s")
""")

md(r"""
Let's check the training curves to make sure we're not overfitting: the validation loss should flatten out, and training stops automatically once it hasn't improved for `stop_after_epochs` epochs.
""")

code(r"""
def plot_losses(summaries, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, m in enumerate(summaries):
        ax.plot(m["training_log_probs"], ls="-", c=f"C{i}", label=f"net {i} train")
        ax.plot(m["validation_log_probs"], ls="--", c=f"C{i}", label=f"net {i} val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Log probability")
    ax.set_title(title)
    ax.legend()
    plt.show()

plot_losses(summaries_npe, "NPE")
""")

md(r"""
Now let's look at the posterior for one of our test simulations. The red lines show the true parameters.
""")

code(r"""
ind = 0
x_obs, theta_obs = x_test[ind], theta_test[ind]

metric = PlotSinglePosterior(num_samples=2000, sample_method="direct", labels=labels)
fig = metric(posterior=posterior_npe, x_obs=x_obs, theta_fid=theta_obs,
             lower=prior_low, upper=prior_high, plot_kws=dict(fill=True))
""")

md(r"""
Typically $\Omega_m$ is very tightly constrained: it sets the overall abundance of haloes, and hence galaxies. The supernova feedback parameters are also constrained, but notice the strong **degeneracy** between $\sigma_8$ and $A_{\rm SN1}$ – both shift the normalisation of the GSMF, so $\sigma_8$ on its own is only weakly constrained. The AGN parameters are poorly constrained, as AGN feedback mostly affects the most massive galaxies, of which there are very few in a $(25\,h^{-1}\,\mathrm{Mpc})^3$ box.

A single posterior can't tell us whether our NPE is *calibrated*. Because NPE is amortised, it is cheap to sample posteriors for every simulation in the test set and check:

* **coverage**: does the true value fall in the $X\%$ credible interval $X\%$ of the time?
* **rank histograms**: the rank of the true value among the posterior samples should be uniform,
* **TARP** ([Lemos et al. 2023](https://arxiv.org/abs/2302.03026)): a multi-dimensional coverage test.

Curves above (below) the diagonal indicate under-confident (over-confident) posteriors. Note that the per-parameter coverage tests only check the 1D *marginals*; TARP tests the *joint* posterior, and can reveal miscalibration (e.g. over-confidence) that the marginals hide. Larger training sets, bigger ensembles and more careful hyperparameter choices all help here.
""")

code(r"""
metric = PosteriorCoverage(
    num_samples=1000, sample_method="direct", labels=labels,
    plot_list=["coverage", "histogram", "predictions", "tarp"], out_dir=None,
)

t0 = time.perf_counter()
fig = metric(posterior=posterior_npe, x=x_test, theta=theta_test)
print(f"Sampled {len(x_test)} posteriors in {time.perf_counter() - t0:.1f} s")
""")

# ---------------------------------------------------------------------------
md(r"""
## Neural Likelihood Estimation (NLE)

NLE instead learns the **likelihood** $q_\phi(x \,|\, \theta)$, a density over the (higher-dimensional) data. To get the posterior we must combine it with the prior and sample $p(\theta \,|\, x) \propto q_\phi(x \,|\, \theta)\, p(\theta)$ using MCMC, *separately for every observation*.

Learning the likelihood has some advantages: it is independent of the prior, and likelihoods from independent observations can simply be multiplied together. But sampling is much slower.
""")

code(r"""
nets_nle = [
    ili.utils.load_nde_sbi(engine="NLE", model="maf", hidden_features=50, num_transforms=5, repeats=2),
]

runner_nle = InferenceRunner.load(
    backend="sbi", engine="NLE", prior=prior, nets=nets_nle,
    device=device, train_args=train_args, out_dir=None,
)

t0 = time.perf_counter()
posterior_nle, summaries_nle = runner_nle(loader=loader)
print(f"NLE training: {time.perf_counter() - t0:.1f} s")
plot_losses(summaries_nle, "NLE")
""")

md(r"""
We sample with the vectorised slice sampler from `sbi`, running several chains in parallel. We'll collect the samples from each method in a dictionary, along with how long they took, and plot them with [`corner`](https://corner.readthedocs.io).
""")

code(r"""
import corner
from ili.utils.samplers import DirectSampler, PyroSampler

def plot_corner(samples, truth, colors=("C0", "C1", "C2", "C3")):
    fig = None
    ranges = list(zip(prior_low, prior_high))
    for (name, s), c in zip(samples.items(), colors):
        fig = corner.corner(
            s, fig=fig, color=c, labels=labels, range=ranges, bins=30, smooth=1.0,
            levels=(0.68, 0.95), plot_datapoints=False, plot_density=False,
            hist_kwargs=dict(density=True),
        )
    corner.overplot_lines(fig, truth, color="k", ls="--")
    for (name, _), c in zip(samples.items(), colors):
        fig.axes[1].plot([], [], color=c, label=name)
    fig.axes[1].legend(loc="upper left", frameon=False)
    return fig

samples, timings = {}, {}
nsamp = 5000

t0 = time.perf_counter()
samples["NPE"] = DirectSampler(posterior_npe).sample(nsamp, x=x_obs)
timings["NPE"] = time.perf_counter() - t0

mcmc_kwargs = dict(num_chains=20, burn_in=200, thin=5)

t0 = time.perf_counter()
samples["NLE"] = PyroSampler(posterior_nle, **mcmc_kwargs).sample(nsamp, x=x_obs)
timings["NLE"] = time.perf_counter() - t0
print(f"NLE sampling (one observation): {timings['NLE']:.1f} s")

plot_corner({"NLE": samples["NLE"]}, theta_obs, colors=["C1"])
plt.show()
""")

# ---------------------------------------------------------------------------
md(r"""
## Neural Ratio Estimation (NRE)

NRE trains a **classifier** to distinguish jointly-drawn pairs $(\theta, x) \sim p(\theta, x)$ from shuffled pairs $(\theta, x) \sim p(\theta)p(x)$. The optimal classifier's logit is the likelihood-to-evidence ratio

$$ r(x, \theta) = \frac{p(x \,|\, \theta)}{p(x)} = \frac{p(\theta \,|\, x)}{p(\theta)}, $$

so the posterior is $p(\theta \,|\, x) = r(x,\theta)\,p(\theta)$, which again we sample with MCMC. Classifiers are simple and flexible, but we never get a normalised density, and sampling is again per-observation.
""")

code(r"""
nets_nre = [
    ili.utils.load_nde_sbi(engine="NRE", model="resnet", hidden_features=64, repeats=2),
]

runner_nre = InferenceRunner.load(
    backend="sbi", engine="NRE", prior=prior, nets=nets_nre,
    device=device, train_args=train_args, out_dir=None,
)

t0 = time.perf_counter()
posterior_nre, summaries_nre = runner_nre(loader=loader)
print(f"NRE training: {time.perf_counter() - t0:.1f} s")
""")

code(r"""
t0 = time.perf_counter()
samples["NRE"] = PyroSampler(posterior_nre, **mcmc_kwargs).sample(nsamp, x=x_obs)
timings["NRE"] = time.perf_counter() - t0
print(f"NRE sampling (one observation): {timings['NRE']:.1f} s")

plot_corner({"NRE": samples["NRE"]}, theta_obs, colors=["C2"])
plt.show()
""")

# ---------------------------------------------------------------------------
md(r"""
## Comparing NPE, NLE and NRE

Let's overplot the samples from all three for the same test simulation. With a perfect simulator and infinite training data all three would converge to the same, true posterior; with ~900 simulations they will differ somewhat.
""")

code(r"""
for k, v in timings.items():
    print(f"{k}: {v:6.2f} s for {nsamp} samples")

plot_corner(samples, theta_obs)
plt.show()
""")

# ---------------------------------------------------------------------------
md(r"""
## A "traditional" approach: emulator + Gaussian likelihood + MCMC

How would we tackle this problem without SBI? A common approach is to

1. build an **emulator**, a cheap model for the *mean* data vector $\mu(\theta)$, trained on the simulations,
2. **assume** a Gaussian likelihood, $\log p(x \,|\, \theta) = -\tfrac{1}{2}\,[x - \mu(\theta)]^{\rm T} C^{-1} [x - \mu(\theta)] + \mathrm{const}$, with some covariance $C$,
3. sample the posterior with MCMC, here using [`emcee`](https://emcee.readthedocs.io).

For the emulator we use a simple cubic polynomial regression. For the covariance we use the scatter of the training simulations around the emulator prediction, which (roughly) captures both cosmic variance and emulator error. Note that we assume it is *constant* across parameter space.
""")

code(r"""
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

emulator = make_pipeline(StandardScaler(), PolynomialFeatures(degree=3), Ridge(alpha=1e-2))
emulator.fit(theta_train, x_train)

resid = x_train - emulator.predict(theta_train)
cov = np.cov(resid, rowvar=False)
icov = np.linalg.inv(cov)

# how well does the emulator do on the unseen test set?
resid_test = x_test - emulator.predict(theta_test)
print("RMS residual per bin (train):", np.sqrt((resid**2).mean(0)).round(2))
print("RMS residual per bin (test): ", np.sqrt((resid_test**2).mean(0)).round(2))
""")

code(r"""
import emcee

def log_prob(theta, x):
    # theta has shape (nwalkers, ndim): we evaluate all walkers at once
    inside = np.all((theta > prior_low) & (theta < prior_high), axis=1)
    r = x - emulator.predict(theta)
    lp = -0.5 * np.einsum("ij,jk,ik->i", r, icov, r)
    return np.where(inside, lp, -np.inf)

ndim, nwalkers = theta_all.shape[1], 32
p0 = rng.uniform(prior_low, prior_high, size=(nwalkers, ndim))

t0 = time.perf_counter()
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=(x_obs,), vectorize=True)
sampler.run_mcmc(p0, 4000, progress=True)
timings["MCMC"] = time.perf_counter() - t0

print(f"Mean acceptance fraction: {sampler.acceptance_fraction.mean():.2f}")
print(f"Autocorrelation times: {sampler.get_autocorr_time(quiet=True).round(0)}")
samples["MCMC"] = sampler.get_chain(discard=1000, thin=10, flat=True)
print(f"MCMC: {timings['MCMC']:.1f} s")
""")

code(r"""
plot_corner({k: samples[k] for k in ["NPE", "MCMC"]}, theta_obs)
plt.show()
""")

md(r"""
### So why bother with SBI?

For a single observation and a simple data vector like ours, the emulator + MCMC approach can do a perfectly reasonable job. But note what it required, and what NPE gave us for free:

1. **An explicit likelihood.** We *assumed* the data are Gaussian with a fixed covariance. In reality the GSMF counts at the high-mass end are Poisson-like, our floor value for empty bins is decidedly non-Gaussian, and the scatter depends on $\theta$. NPE makes none of these assumptions – it learns the full distribution, including non-Gaussianity and parameter-dependent noise, directly from the simulations.
2. **An emulator.** We had to choose a functional form for $\mu(\theta)$ and hope it is accurate everywhere. With NPE there is no separate emulation step.
3. **Cost per observation.** MCMC must be re-run from scratch for every new $x$. Here that is cheap, because our polynomial emulator takes microseconds to evaluate; with a neural likelihood (NLE) it was already ~minutes per observation, and with a likelihood requiring a real model evaluation (e.g. forward-modelling spectra with `synthesizer`) it can be hours. NPE is *amortised*: after training once, posteriors for new data are essentially free. That is what made the coverage tests above – sampling 100 independent posteriors – trivial.
4. **Scaling.** MCMC cost grows quickly with the number of parameters, and the Gaussian approach needs a well-estimated (and invertible) covariance, which becomes hard for long data vectors. Neural SBI methods can use embedding networks to compress high-dimensional data (images, fields, spectra) automatically.

The comparison below makes point 3 concrete.
""")

code(r"""
t0 = time.perf_counter()
_ = [DirectSampler(posterior_npe).sample(1000, x=x) for x in x_test]
t_npe_all = time.perf_counter() - t0

print(f"NPE:  {t_npe_all:.1f} s for posteriors of all {len(x_test)} test simulations")
print(f"MCMC: ~{timings['MCMC'] * len(x_test):.0f} s (estimated) for the same")
print(f"NLE:  ~{timings['NLE'] * len(x_test):.0f} s (estimated) for the same")
""")

# ---------------------------------------------------------------------------
md(r"""
## Exercises

1. **Which statistic carries the information?** Re-train NPE using only the GSMF (or only the SHMR). Which parameters lose their constraints?
2. **Is our NPE well calibrated?** Try a smaller training set (e.g. 200 simulations). How do the coverage and TARP plots change? Try an ensemble with an MDN (`model="mdn"`) as well as MAFs.
3. **Model misspecification.** Train on SIMBA and apply the posterior to the Swift-EAGLE LH suite (`python download_camels.py --suite Swift-EAGLE`). The astrophysical parameters mean different things in the two models, but $\Omega_m$ and $\sigma_8$ do not. Are they still recovered? This is a key concern when applying simulation-trained models to real data.
4. **Coverage for the MCMC approach.** Run the emulator + MCMC on 20 test simulations and compute the fraction for which the truth lies within the 68% credible interval. Is the Gaussian likelihood calibrated?
5. **Synthetic observables.** Following the earlier part of this tutorial, replace the stellar mass function with a forward-modelled luminosity function computed with `synthesizer`.
""")

# ---------------------------------------------------------------------------
nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "accelerator": "GPU",
    "colab": {"provenance": []},
}
out = sys.argv[1] if len(sys.argv) > 1 else "CAMELS_SBI_tutorial.ipynb"
nbf.write(nb, out)
print(f"Wrote {out} ({len(cells)} cells)")
