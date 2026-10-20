<script setup>
// Bayes' theorem with the term learnt by each SBI flavour highlighted on click.
import { ref } from 'vue'

const active = ref(null)

const METHODS = [
  {
    key: 'npe',
    name: ['N', 'eural ', 'P', 'osterior ', 'E', 'stimation'],
    term: 'post',
    learns: 'q_\\phi(\\theta \\mid x)',
    how: 'Conditional density estimator (normalising flow, mixture density network)',
    pros: ['Amortised: sample any new x instantly', 'Cheap validation (coverage, TARP)'],
    cons: ['Tied to the training prior', 'Can extrapolate badly outside the training set'],
  },
  {
    key: 'nle',
    name: ['N', 'eural ', 'L', 'ikelihood ', 'E', 'stimation'],
    term: 'like',
    learns: 'q_\\phi(x \\mid \\theta)',
    how: 'Density estimator over the data, conditioned on the parameters',
    pros: ['Swap priors without retraining', 'Multiply likelihoods of independent data'],
    cons: ['MCMC needed for every observation', 'Harder for high-dimensional x'],
  },
  {
    key: 'nre',
    name: ['N', 'eural ', 'R', 'atio ', 'E', 'stimation'],
    term: 'ratio',
    learns: 'r_\\phi(x, \\theta) \\approx \\dfrac{P(x \\mid \\theta)}{P(x)}',
    how: 'Classifier separating joint pairs (θ, x) from shuffled pairs',
    pros: ['Simple, flexible classifiers', 'Swap priors without retraining'],
    cons: ['MCMC needed for every observation', 'Unnormalised: no direct density'],
  },
]

const toggle = (k) => (active.value = active.value === k ? null : k)
const on = (term) => METHODS.find((m) => m.key === active.value)?.term === term
</script>

<template>
  <div class="flavours">
    <div class="bayes">
      <div class="term" :class="{ hl: on('post'), npe: true }">
        <Tex expr="P(\boldsymbol{\theta} \mid \mathbf{x})" />
        <div class="cap">posterior</div>
      </div>
      <div class="op"><Tex expr="\propto" /></div>
      <div class="term" :class="{ hl: on('like') || on('ratio'), nle: on('like'), nre: on('ratio') }">
        <Tex expr="P(\mathbf{x} \mid \boldsymbol{\theta})" />
        <div class="cap">likelihood</div>
      </div>
      <div class="term prior">
        <Tex expr="P(\boldsymbol{\theta})" />
        <div class="cap">prior</div>
      </div>
    </div>

    <div class="cards">
      <button
        v-for="m in METHODS"
        :key="m.key"
        class="card"
        :class="[m.key, { active: active === m.key, dim: active && active !== m.key }]"
        @click="toggle(m.key)"
      >
        <div class="name">
          <template v-for="(s, i) in m.name" :key="i">
            <b v-if="i % 2 === 0">{{ s }}</b><span v-else>{{ s }}</span>
          </template>
        </div>
        <div class="detail" :class="{ open: active === m.key }">
          <div class="learns">learns <Tex :expr="m.learns" /></div>
          <div class="how">{{ m.how }}</div>
          <ul>
            <li v-for="p in m.pros" :key="p" class="pro">{{ p }}</li>
            <li v-for="c in m.cons" :key="c" class="con">{{ c }}</li>
          </ul>
        </div>
      </button>
    </div>
    <div class="hint" :class="{ gone: active }">Click a method</div>
  </div>
</template>

<style scoped>
.flavours {
  --npe: #d1495b;
  --nle: #2e5fd9;
  --nre: #2f8f46;
}
.bayes {
  display: flex;
  justify-content: center;
  align-items: flex-start;
  gap: 18px;
  font-size: 1.9rem;
  margin: 10px 0 26px;
}
.term {
  padding: 4px 14px 2px;
  border-radius: 10px;
  border: 2px solid transparent;
  text-align: center;
  transition: all 0.3s;
}
.term .cap {
  font-size: 0.85rem;
  color: var(--muted);
  border-top: 1.5px solid var(--line);
  margin-top: 2px;
  padding-top: 2px;
}
.op {
  padding-top: 4px;
}
.term.hl.npe {
  border-color: var(--npe);
  background: color-mix(in srgb, var(--npe) 8%, white);
}
.term.hl.nle {
  border-color: var(--nle);
  background: color-mix(in srgb, var(--nle) 8%, white);
}
.term.hl.nre {
  border-color: var(--nre);
  background: color-mix(in srgb, var(--nre) 8%, white);
}
.cards {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 18px;
  align-items: start;
}
.card {
  text-align: left;
  border: 1.5px solid var(--c);
  border-radius: 12px;
  padding: 12px 16px;
  background: white;
  transition: all 0.3s;
  cursor: pointer;
}
.card.npe {
  --c: var(--npe);
}
.card.nle {
  --c: var(--nle);
}
.card.nre {
  --c: var(--nre);
}
.card.active {
  background: color-mix(in srgb, var(--c) 6%, white);
  box-shadow: 0 6px 20px -10px var(--c);
}
.card.dim {
  opacity: 0.45;
}
.name {
  font-size: 1.05rem;
  color: var(--c);
}
.name b {
  font-weight: 700;
}
.detail {
  max-height: 0;
  overflow: hidden;
  transition: max-height 0.4s ease;
  font-size: 0.78rem;
  color: var(--ink);
}
.detail.open {
  max-height: 260px;
}
.learns {
  margin-top: 10px;
  font-size: 0.9rem;
}
.how {
  color: var(--muted);
  margin: 6px 0 4px;
}
ul {
  margin: 0;
  padding: 0;
  list-style: none;
}
li {
  margin: 3px 0;
  padding-left: 18px;
  position: relative;
}
li::before {
  position: absolute;
  left: 0;
  font-weight: 700;
}
li.pro::before {
  content: '+';
  color: #2f8f46;
}
li.con::before {
  content: '−';
  color: #d1495b;
}
.hint {
  text-align: center;
  font-size: 0.75rem;
  color: var(--muted);
  margin-top: 14px;
  transition: opacity 0.3s;
}
.hint.gone {
  opacity: 0;
}
</style>
