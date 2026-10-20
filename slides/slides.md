---
theme: default
title: Simulation-Based Inference
info: An introduction to simulation-based inference; companion to the CAMELS × LtU-ILI tutorial.
colorSchema: light
fonts:
  sans: Inter
  serif: Source Serif 4
  mono: JetBrains Mono
transition: fade
mdc: true
layout: cover
class: cover
---

<div class="kicker">Tutorial</div>

# Simulation-Based Inference

<div class="abs-bl m-12 text-sm muted">
Chris Lovell
</div>

---

# Bayesian inference

<div class="grid grid-cols-[1.1fr_1fr] gap-10 items-start text-[0.95rem]">
<div>

$$
\underbrace{P(\boldsymbol{\theta} \mid \mathbf{x})}_{\text{posterior}}
= \frac{
  \overbrace{P(\mathbf{x} \mid \boldsymbol{\theta})}^{\text{likelihood}}\;
  \overbrace{P(\boldsymbol{\theta})}^{\text{prior}}
}{
  \underbrace{P(\mathbf{x})}_{\text{evidence}}
}
$$

<v-clicks>

- **Prior** – what we believe about the parameters $\boldsymbol\theta$ before seeing data
- **Likelihood** – how probable the data $\mathbf{x}$ are, given $\boldsymbol\theta$
- **Posterior** – what we believe *after* seeing the data
- **Evidence** – normalisation; useful for comparing models

</v-clicks>

</div>
<div class="text-[0.8rem] leading-snug">

<v-click>

<div class="card">
<div class="kicker">The traditional recipe</div>

1. Write down the likelihood, e.g. Gaussian:<br>
   $-2 \ln P(\mathbf{x} \mid \boldsymbol\theta) = \chi^2 + \text{const}$
2. Explore the posterior with MCMC or nested sampling

</div>

</v-click>

<v-click>

<div class="card mt-4" style="border-color: var(--like)">
<div class="kicker" style="color: var(--like)">The catch</div>

For realistic models the likelihood is often **unknown or intractable**:

- stochastic forward models (galaxy formation, cosmic variance)
- latent variables to marginalise: $\int P(\mathbf{x} \mid \mathbf{z}, \boldsymbol\theta)\, P(\mathbf{z} \mid \boldsymbol\theta)\, \mathrm{d}\mathbf{z}$
- complex noise, selection effects, systematics

…but we can often **simulate** $\mathbf{x} \sim P(\mathbf{x} \mid \boldsymbol\theta)$.

</div>

</v-click>

</div>
</div>

---
clicks: 3
---

# Simulation and inference

<SimInferDiagram :step="$clicks" />

<div class="credit">Adapted from Matt Ho</div>

---

# The SBI idea

<p class="lead">If you can simulate it, you can learn to invert it.</p>

<div class="pipeline mt-8">
  <div v-click="1" class="node">
    <div class="kicker">1 · Sample</div>
    <Tex expr="\boldsymbol\theta_i \sim P(\boldsymbol\theta)" />
  </div>
  <div v-click="2" class="arrow">→</div>
  <div v-click="2" class="node sim">
    <div class="kicker">2 · Simulate</div>
    <Tex expr="\mathbf{x}_i \sim P(\mathbf{x} \mid \boldsymbol\theta_i)" />
  </div>
  <div v-click="3" class="arrow">→</div>
  <div v-click="3" class="node">
    <div class="kicker">3 · Train</div>
    <span>neural density estimator on <Tex expr="\{(\boldsymbol\theta_i, \mathbf{x}_i)\}_{i=1}^{N}" /></span>
  </div>
  <div v-click="4" class="arrow">→</div>
  <div v-click="4" class="node out">
    <div class="kicker">4 · Infer</div>
    <Tex expr="q(\boldsymbol\theta \mid \mathbf{x}_{\rm obs})" />
  </div>
</div>

<div v-click="5" class="grid grid-cols-3 gap-6 mt-12 text-sm">
  <div>
    <b>The likelihood is never written down</b>
    <p class="muted">It is implicit in the simulator, noise and all.</p>
  </div>
  <div>
    <b>Pairs sample the joint</b>
    <p class="muted"><Tex expr="(\boldsymbol\theta_i, \mathbf{x}_i) \sim P(\mathbf{x}, \boldsymbol\theta)" />: everything we need is in there.</p>
  </div>
  <div>
    <b>Simulations are reusable</b>
    <p class="muted">Run once, then train, validate and infer many times.</p>
  </div>
</div>

<style>
.pipeline {
  display: flex;
  align-items: stretch;
  gap: 12px;
}
.pipeline .node {
  flex: 1;
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 14px 16px;
  font-size: 0.95rem;
}
.pipeline .node.sim {
  border-color: var(--sim);
  background: color-mix(in srgb, var(--sim) 8%, white);
}
.pipeline .node.out {
  border-color: var(--post);
  background: color-mix(in srgb, var(--post) 6%, white);
}
.pipeline .arrow {
  align-self: center;
  color: var(--muted);
  font-size: 1.4rem;
}
</style>

---
clicks: 2
---

# Posteriors and likelihoods are slices of the joint

<div class="-mt-4 mb-1 text-center text-sm">

$P(\theta \mid x) \propto P(x \mid \theta)\, P(\theta)$ · a toy simulator: $x = \mathrm{softplus}(\theta) + \mathcal{N}(0, \sigma^2)$

</div>

<JointExplorer :step="$clicks" />

<div class="credit">Adapted from Matt Ho</div>

---
clicks: 2
---

# From samples to posteriors

<div class="-mt-3 mb-3 text-sm muted min-h-[2.6em]">
<v-switch>
<template #0>

**Simulations sample the joint:** draw $\theta \sim P(\theta)$, then run the simulator, $x \sim P(x \mid \theta)$. Each dot is one simulation.

</template>
<template #1>

**Rejection ABC** (Rubin 1984; Pritchard et al. 1999) keeps the simulations with $|x - x_0| < \epsilon$: the slice from the previous slide, built from samples. Most simulations are thrown away.

</template>
<template #2>

**Neural SBI** instead fits a conditional density $q(\theta \mid x)$ to *all* the simulations: here a small mixture density network, trained live. Drag $x_0$: no retraining needed.

</template>
</v-switch>
</div>

<JointExplorer controls :stage="$clicks" />

<!--
Click 1 – ABC: draw from the prior, simulate, keep theta if the simulated x is within epsilon of the observation. The kept thetas sample P(theta | d(x, x_obs) < epsilon), which tends to the true posterior as epsilon -> 0.

Problems to point out with the sliders:
- Wasteful: almost every simulation is rejected; the acceptance rate falls as epsilon^d with data dimension d, so real data must be compressed to a few summary statistics.
- Bias vs noise: large epsilon blurs the posterior (try 0.3); small epsilon leaves few samples (try 0.01 with N = 500).
- Not amortised: move x0 and the accepted set changes; a new observation needs a new acceptance step.

Click 2 – neural SBI keeps the recipe (prior, simulate) but replaces accept/reject with density estimation on all simulations. The orange curve is a 3-component mixture density network trained in the browser. Dragging x0 just re-evaluates it (amortised). Switch to the informative prior and drag x0 to ~0 or ~1.2: no simulations there, so the network extrapolates badly – NPE is only as good as the training set.
-->

---

# Three flavours of neural SBI

<p class="lead -mt-2">Assuming a perfect simulator, we have (data, parameter) pairs. What should we learn?</p>

<FlavourCards />

<div class="credit">Adapted from Matt Ho</div>

---

# Why use SBI?

<div class="grid grid-cols-3 gap-4 mt-2 text-[0.85rem] leading-snug">

<div v-click class="card">
<div class="kicker">No likelihood needed</div>

Any stochastic simulator works: cosmological simulations, galaxy formation models, gravitational-wave signals in detector noise.

</div>

<div v-click class="card">
<div class="kicker">Amortised</div>

Train once, then infer in milliseconds per object: catalogue-scale inference for surveys like Euclid and Rubin, or rapid follow-up alerts.

</div>

<div v-click class="card">
<div class="kicker">Implicit marginalisation</div>

Nuisance parameters (baryonic physics, instrument systematics, initial conditions) are marginalised for free, simply by not inferring them.

</div>

<div v-click class="card">
<div class="kicker">High-dimensional data</div>

Embedding networks learn compact summaries of fields, images, spectra and time series, beyond hand-picked statistics.

</div>

<div v-click class="card">
<div class="kicker">Calibration at scale</div>

Coverage tests over thousands of mock observations become cheap, so uncertainties can be checked.

</div>

<div v-click class="card">
<div class="kicker">Model comparison</div>

Compare physical models directly, e.g. competing feedback prescriptions, dark matter models or stellar population codes.

</div>

</div>

---

# Amortisation: SED fitting at survey scale

<SpeedChart class="mt-4" />

<div class="credit">Synference: Harvey, Lovell et al. 2026 · timed on 3000 JADES galaxies with spectroscopic redshifts</div>

---

# Caveats

<div class="grid grid-cols-2 gap-x-12 gap-y-6 mt-6">

<div v-click>

### The simulator *is* the model
SBI is only as good as the forward model. **Model misspecification** – real data unlike any simulation – gives confident, wrong answers.

</div>

<div v-click>

### Always validate
Check calibration on held-out simulations: coverage, rank statistics, TARP. Ensembles help.

</div>

<div v-click>

### Priors are baked in
NPE learns the posterior *for the training prior*. Changing prior means re-training (or using NLE / NRE).

</div>

<div v-click>

### Simulation budget & summaries
Expensive simulators limit training sets; the choice of summary statistic sets how much information survives.

</div>

</div>

---

# Hands-on

<div class="grid grid-cols-[1.2fr_1fr] gap-12 mt-4">
<div>

<div class="card">
<div class="kicker">Notebook</div>

**Inferring cosmology and feedback from CAMELS**

GSMF + stellar–halo mass relation from 1000 SIMBA simulations → NPE, NLE and NRE with LtU-ILI, and a comparison to emulator + MCMC.

[Open in Colab](https://colab.research.google.com/github/christopherlovell/sbi_tutorial/blob/main/CAMELS_SBI_tutorial.ipynb) · [GitHub](https://github.com/christopherlovell/sbi_tutorial)

</div>

</div>
<div class="text-sm">

**Tools**
- [LtU-ILI](https://github.com/maho3/ltu-ili) – Ho et al. 2024
- [sbi](https://github.com/sbi-dev/sbi) – Tejero-Cantero et al. 2020
- [Synthesizer](https://synthesizer-project.github.io/synthesizer/) – Lovell et al. 2025; Roper et al. 2025
- [CAMELS](https://camels.readthedocs.io) – Villaescusa-Navarro et al. 2021

**Further reading**
- Cranmer, Brehmer & Louppe 2020, *The frontier of simulation-based inference*, PNAS

</div>
</div>
