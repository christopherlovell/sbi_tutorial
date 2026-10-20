<script setup>
// The toy simulator's prior, joint P(x, theta) and conditional slices, then
// samples, rejection ABC and a neural posterior estimate. Drag on the joint
// to choose (theta_0, x_0). Clicks (step):
//   0: prior + joint            1: slices at an informative point
//   2: slices where the simulator is flat (posterior ~ prior)
//   3: simulations (+ controls)  4: rejection ABC  5: NPE fit to all sims
import { computed, onBeforeUnmount, onMounted, ref, toRaw, watch } from 'vue'
import { scaleLinear } from 'd3-scale'
import {
  THETA_RANGE, X_RANGE, PRIORS, abcPosterior, jointGrid, likelihoodCurve,
  linspace, posteriorCurve, priorCurve, simulate,
} from '../lib/toy.js'
import { evalMDN } from '../lib/mdn.js'

const props = defineProps({ step: { type: Number, default: 99 } })

const prior = ref('bimodal')
const sigma = ref(0.1)
const simsOn = ref(true)
const nSims = ref(2000)
const eps = ref(0.05)

const theta0 = ref(1.0)
const x0 = ref(0.75)

const showSlices = computed(() => props.step >= 1)
const sampling = computed(() => props.step >= 3)
const showSims = computed(() => sampling.value && simsOn.value)
const showABC = computed(() => showSims.value && props.step >= 4)
const showNPE = computed(() => showSims.value && props.step >= 5)

// ---- move (theta_0, x_0) between presets when stepping through clicks
const INFORMATIVE = [1.0, 0.75]
const FLAT = [-0.1, 0.05]
const preset = (s) => (s === 2 ? FLAT : s <= 3 ? INFORMATIVE : null)
let raf = 0
function moveTo([t1, x1]) {
  cancelAnimationFrame(raf)
  const [ta, xa] = [theta0.value, x0.value]
  const start = performance.now()
  const tick = (now) => {
    const u = Math.min(1, (now - start) / 700)
    const e = u < 0.5 ? 2 * u * u : 1 - (-2 * u + 2) ** 2 / 2
    theta0.value = ta + (t1 - ta) * e
    x0.value = xa + (x1 - xa) * e
    if (u < 1) raf = requestAnimationFrame(tick)
  }
  raf = requestAnimationFrame(tick)
}
watch(
  () => props.step,
  (s) => preset(s) && moveTo(preset(s)),
)

// ---- layout: fixed sizes so nothing moves between clicks
const W = 400
const H = 236
const PRIOR_H = 74
const SLICE_W = 340
const SLICE_H = 128
const m = { t: 8, r: 10, b: 32, l: 40 }
const sx = scaleLinear().domain(THETA_RANGE).range([m.l, W - m.r])
const sy = scaleLinear().domain(X_RANGE).range([H - m.b, m.t])
const ticks = sx.ticks(5)
const yticks = sy.ticks(5)

// ---- curves
const ts = linspace(...THETA_RANGE, 400)
const xs = linspace(...X_RANGE, 400)
const priorY = computed(() => priorCurve(prior.value, ts))
const postY = computed(() => posteriorCurve(x0.value, sigma.value, prior.value, ts))
const likeY = computed(() => likelihoodCurve(theta0.value, sigma.value, xs))

const sims = computed(() => (showSims.value ? simulate(nSims.value, sigma.value, prior.value, 7) : []))
const abc = computed(() => (showABC.value ? abcPosterior(sims.value, x0.value, eps.value) : null))

// ---- neural posterior estimate: trained in a worker whenever the simulations
// change, then evaluated at any x0 without retraining (amortised)
const fit = ref(null)
const training = ref(false)
let worker = null
let job = 0
let timer = 0
function retrain() {
  clearTimeout(timer)
  if (!worker || !showSims.value) return
  training.value = true
  timer = setTimeout(() => worker.postMessage({ id: ++job, sims: toRaw(sims.value) }), 250)
}
const npeY = computed(() => (showNPE.value && fit.value ? evalMDN(fit.value, x0.value, ts) : null))

// ---- joint density as a banded heatmap on a canvas
const canvas = ref(null)
const simCanvas = ref(null)
const NT = 200
const NX = 140
const LEVELS = [0.01, 0.05, 0.15, 0.35, 0.6]
const SHADES = ['#eef4fc', '#cfe0f6', '#9fc2ec', '#6a9fdf', '#3f7fcf', '#215fae']

function drawJoint() {
  const c = canvas.value
  if (!c) return
  const { values, max } = jointGrid(sigma.value, prior.value, NT, NX)
  const ctx = c.getContext('2d')
  const img = ctx.createImageData(NT, NX)
  for (let j = 0; j < NX; j++)
    for (let i = 0; i < NT; i++) {
      const v = values[j * NT + i] / max
      let k = 0
      while (k < LEVELS.length && v > LEVELS[k]) k++
      const hex = k === 0 ? '#ffffff' : SHADES[k]
      const p = ((NX - 1 - j) * NT + i) * 4 // flip: row 0 is the top (largest x)
      img.data[p] = parseInt(hex.slice(1, 3), 16)
      img.data[p + 1] = parseInt(hex.slice(3, 5), 16)
      img.data[p + 2] = parseInt(hex.slice(5, 7), 16)
      img.data[p + 3] = 255
    }
  ctx.putImageData(img, 0, 0)
}

function drawSims() {
  const c = simCanvas.value
  if (!c) return
  const ctx = c.getContext('2d')
  const dpr = 2
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, W, H)
  if (!showSims.value) return
  const r = nSims.value > 5000 ? 0.7 : 1.2
  for (const [t, x] of sims.value) {
    const inBand = showABC.value && Math.abs(x - x0.value) < eps.value
    ctx.fillStyle = inBand ? 'rgba(37, 99, 235, 0.9)' : 'rgba(30, 41, 59, 0.28)'
    ctx.beginPath()
    ctx.arc(sx(t), sy(x), inBand ? r + 0.5 : r, 0, 2 * Math.PI)
    ctx.fill()
  }
}

onMounted(() => {
  const p = preset(props.step)
  if (p) [theta0.value, x0.value] = p
  drawJoint()
  drawSims()
  worker = new Worker(new URL('../lib/mdn.worker.js', import.meta.url), { type: 'module' })
  worker.onmessage = ({ data }) => {
    if (data.id !== job) return // superseded by a newer request
    fit.value = data.model
    training.value = false
  }
  retrain()
})
onBeforeUnmount(() => worker?.terminate())
watch([sigma, prior], drawJoint)
watch([sims, x0, eps, showABC], drawSims)
watch(sims, retrain)

// ---- dragging (convert client coords through Slidev's scaling transform)
const overlay = ref(null)
let dragging = false
function setFromEvent(ev) {
  const r = overlay.value.getBoundingClientRect()
  const px = ((ev.clientX - r.left) / r.width) * W
  const py = ((ev.clientY - r.top) / r.height) * H
  const clamp = (v, [a, b]) => Math.min(b, Math.max(a, v))
  theta0.value = clamp(sx.invert(px), THETA_RANGE)
  x0.value = clamp(sy.invert(py), X_RANGE)
}
const down = (ev) => {
  if (!showSlices.value) return
  cancelAnimationFrame(raf)
  dragging = true
  ev.target.setPointerCapture?.(ev.pointerId)
  setFromEvent(ev)
}
const move = (ev) => dragging && setFromEvent(ev)
const up = () => (dragging = false)

const fmt = (v) => +v.toFixed(2)
</script>

<template>
  <div class="explorer">
    <div class="col-left">
      <MiniPlot
        :width="W"
        :height="PRIOR_H"
        :x-domain="THETA_RANGE"
        :curves="[{ x: ts, y: priorY, color: 'var(--prior)', fill: true }]"
        title="\text{Prior } P(\theta)"
        hide-x-ticks
      />
      <div class="joint-title"><Tex expr="\text{Joint } P(x, \theta)" /></div>
      <div class="joint" :style="{ width: `${W}px`, height: `${H}px` }">
        <canvas
          ref="canvas"
          :width="NT"
          :height="NX"
          :style="{ left: `${m.l}px`, top: `${m.t}px`, width: `${W - m.l - m.r}px`, height: `${H - m.t - m.b}px` }"
        />
        <canvas ref="simCanvas" :width="W * 2" :height="H * 2" :style="{ width: `${W}px`, height: `${H}px` }" />
        <svg
          ref="overlay"
          :width="W"
          :height="H"
          :class="{ draggable: showSlices }"
          @pointerdown="down"
          @pointermove="move"
          @pointerup="up"
          @pointerleave="up"
        >
          <g class="gridlines">
            <line v-for="t in ticks" :key="`gx${t}`" :x1="sx(t)" :x2="sx(t)" :y1="m.t" :y2="H - m.b" />
            <line v-for="t in yticks" :key="`gy${t}`" :x1="m.l" :x2="W - m.r" :y1="sy(t)" :y2="sy(t)" />
          </g>
          <rect
            v-if="showABC"
            :x="m.l"
            :width="W - m.l - m.r"
            :y="sy(x0 + eps)"
            :height="sy(x0 - eps) - sy(x0 + eps)"
            class="band"
          />
          <g v-if="showSlices">
            <line :x1="m.l" :x2="W - m.r" :y1="sy(x0)" :y2="sy(x0)" class="xline" />
            <line :x1="sx(theta0)" :x2="sx(theta0)" :y1="m.t" :y2="H - m.b" class="tline" />
            <circle :cx="sx(theta0)" :cy="sy(x0)" r="6" class="handle" />
          </g>
          <g class="axis">
            <rect :x="m.l" :y="m.t" :width="W - m.l - m.r" :height="H - m.t - m.b" fill="none" />
            <text v-for="t in ticks" :key="`tx${t}`" :x="sx(t)" :y="H - m.b + 14" text-anchor="middle">{{ fmt(t) }}</text>
            <text v-for="t in yticks" :key="`ty${t}`" :x="m.l - 6" :y="sy(t) + 3.5" text-anchor="end">{{ fmt(t) }}</text>
          </g>
        </svg>
        <div class="lab-x"><Tex expr="\text{Parameters } \theta" /></div>
        <div class="lab-y"><Tex expr="\text{Data } x" /></div>
      </div>
    </div>

    <div class="col-right" :class="{ hidden: !showSlices }">
      <MiniPlot
        :width="SLICE_W"
        :height="SLICE_H"
        :x-domain="THETA_RANGE"
        :curves="[
          { x: ts, y: priorY, color: 'var(--prior)', dash: true, width: 1.5 },
          { x: ts, y: postY, color: 'var(--post)', fill: true },
          ...(npeY ? [{ x: ts, y: npeY, color: 'var(--npe)', width: 2.5 }] : []),
        ]"
        :bars="abc ? [{ edges: abc.edges, density: abc.density, color: 'var(--post)' }] : []"
        :vlines="[{ x: theta0 }]"
        :title="`\\text{Posterior } P(\\theta \\mid x_0 = ${fmt(x0)})`"
        xlabel="\theta"
      />
      <div class="legend">
        <span><i class="sw post" /> exact</span>
        <span><i class="sw prior dashed" /> prior</span>
        <span v-if="abc"><i class="sw hist" /> ABC: {{ abc.kept }} sims</span>
        <span v-if="showNPE"><i class="sw npe" /> {{ training ? 'NPE: training…' : `NPE: ${Math.min(nSims, 3000)} sims` }}</span>
      </div>

      <!-- fixed-height slot: likelihood before sampling, controls after -->
      <div class="slot">
        <template v-if="!sampling">
          <MiniPlot
            :width="SLICE_W"
            :height="SLICE_H"
            :x-domain="X_RANGE"
            :curves="[{ x: xs, y: likeY, color: 'var(--like)', fill: true }]"
            :vlines="[{ x: x0 }]"
            :title="`\\text{Likelihood } P(x \\mid \\theta_0 = ${fmt(theta0)})`"
            xlabel="x"
          />
          <div class="hint">Drag the point to move the slices</div>
        </template>
        <template v-else>
          <div class="controls">
            <div class="row">
              <span class="lbl">Prior</span>
              <div class="seg">
                <button v-for="(p, k) in PRIORS" :key="k" :class="{ on: prior === k }" @click="prior = k">
                  {{ p.label }}
                </button>
              </div>
            </div>
            <div class="row">
              <span class="lbl"><Tex expr="\sigma" /> noise</span>
              <input v-model.number="sigma" type="range" min="0.02" max="0.3" step="0.01" />
              <span class="val">{{ sigma.toFixed(2) }}</span>
            </div>
            <div class="row">
              <label class="lbl"><input v-model="simsOn" type="checkbox" /> <Tex expr="N" /> sims</label>
              <input v-model.number="nSims" type="range" min="100" max="20000" step="100" :disabled="!simsOn" />
              <span class="val">{{ nSims }}</span>
            </div>
            <div class="row" :class="{ off: !showABC }">
              <span class="lbl"><Tex expr="\epsilon" /> ABC</span>
              <input v-model.number="eps" type="range" min="0.01" max="0.3" step="0.01" :disabled="!showABC" />
              <span class="val">{{ eps.toFixed(2) }}</span>
            </div>
          </div>
          <div class="note">Simulations are drawn from the prior: change it and the training set changes.</div>
        </template>
      </div>
    </div>
  </div>
</template>

<style scoped>
.explorer {
  display: flex;
  gap: 36px;
  justify-content: center;
  align-items: flex-start;
  user-select: none;
}
.col-right {
  width: 340px;
  display: flex;
  flex-direction: column;
  gap: 6px;
  transition: opacity 0.4s;
}
.col-right.hidden {
  opacity: 0;
  pointer-events: none;
}
.joint-title {
  text-align: center;
  font-size: 0.8rem;
  margin: 4px 0 2px;
  padding-left: 30px;
}
.joint {
  position: relative;
}
.joint canvas,
.joint svg {
  position: absolute;
  left: 0;
  top: 0;
}
svg.draggable {
  cursor: crosshair;
}
.gridlines line {
  stroke: rgba(15, 23, 42, 0.08);
}
.axis rect {
  stroke: var(--muted);
}
.axis text {
  font-size: 9px;
  fill: var(--muted);
  font-variant-numeric: tabular-nums;
}
.xline {
  stroke: var(--post);
  stroke-width: 2;
}
.tline {
  stroke: var(--like);
  stroke-width: 2;
}
.handle {
  fill: var(--ink);
  stroke: white;
  stroke-width: 2;
  cursor: grab;
}
.band {
  fill: var(--post);
  opacity: 0.12;
}
.lab-x {
  position: absolute;
  bottom: -6px;
  left: 40px;
  right: 10px;
  text-align: center;
  font-size: 0.75rem;
}
.lab-y {
  position: absolute;
  left: -8px;
  top: 42%;
  transform: translate(-50%, 0) rotate(-90deg);
  font-size: 0.75rem;
}
.legend {
  display: flex;
  gap: 12px;
  white-space: nowrap;
  justify-content: center;
  font-size: 0.65rem;
  color: var(--muted);
  height: 16px;
  margin: 4px 0 2px;
}
.sw {
  display: inline-block;
  width: 16px;
  height: 0;
  border-top: 2px solid;
  vertical-align: middle;
  margin-right: 3px;
}
.sw.post {
  border-color: var(--post);
}
.sw.prior {
  border-color: var(--prior);
}
.sw.dashed {
  border-top-style: dashed;
}
.sw.npe {
  border-color: var(--npe);
  border-top-width: 2.5px;
}
.sw.hist {
  height: 8px;
  border: none;
  background: var(--post);
  opacity: 0.35;
}
.slot {
  height: 196px;
}
.hint {
  font-size: 0.7rem;
  color: var(--muted);
  text-align: center;
  margin-top: 10px;
}
.controls {
  margin-top: 14px;
  display: grid;
  gap: 8px;
  font-size: 0.68rem;
}
.row {
  display: grid;
  grid-template-columns: 74px 1fr 40px;
  align-items: center;
  gap: 8px;
  transition: opacity 0.3s;
}
.row:first-child {
  grid-template-columns: 74px 1fr;
}
.row.off {
  opacity: 0.35;
}
.lbl {
  color: var(--muted);
  display: flex;
  align-items: center;
  gap: 4px;
}
.val {
  font-variant-numeric: tabular-nums;
  text-align: right;
}
.note {
  font-size: 0.62rem;
  color: var(--muted);
  font-style: italic;
  margin-top: 10px;
}
input[type='range'] {
  accent-color: var(--accent);
  width: 100%;
}
input[type='checkbox'] {
  accent-color: var(--accent);
}
.seg {
  display: inline-flex;
  border: 1px solid var(--line);
  border-radius: 6px;
  overflow: hidden;
  justify-self: start;
}
.seg button {
  padding: 2px 9px;
  font-size: 0.68rem;
  color: var(--muted);
  border-right: 1px solid var(--line);
}
.seg button:last-child {
  border-right: none;
}
.seg button.on {
  background: var(--accent);
  color: white;
}
</style>
