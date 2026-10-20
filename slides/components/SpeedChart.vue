<script setup>
// Inference time per object (Harvey, Lovell et al. 2026), and the total cost
// for a survey of N objects on a single core.
import { computed, ref } from 'vue'
import { scaleLog } from 'd3-scale'

const gpu = ref(false)
const logN = ref(Math.log10(3000))

const SURVEYS = [
  { label: 'JADES spec-z', n: 3e3 },
  { label: 'DESI', n: 3e7 },
  { label: 'Euclid', n: 1.5e9 },
  { label: 'Rubin', n: 2e10 },
]

const rows = computed(() => [
  { name: 'Synference MDN', t: gpu.value ? 0.0018 : 0.0037, sbi: true },
  { name: 'Synference NSF', t: gpu.value ? 0.047 : 0.069, sbi: true },
  { name: 'EAZY-py', note: 'photo-z only', t: 0.018 },
  { name: 'Bagpipes', t: 101.4 },
  { name: 'Parrot', note: 'Prospector emulator', t: 1800, approx: true },
  { name: 'Prospector', t: 36000, lower: true },
])

const W = 430
const ROW = 34
const L = 150
const sx = scaleLog().domain([1e-3, 1e5]).range([L, W - 10])
const ticks = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5]
const tickLabel = (t) => (t >= 1 && t < 1e3 ? `${t}` : sup(t))
function sup(t) {
  const e = Math.round(Math.log10(t))
  const map = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵' }
  return `10${String(e).split('').map((c) => map[c]).join('')}`
}

const N = computed(() => 10 ** logN.value)
function human(sec) {
  const units = [
    [3.156e7, 'years'],
    [86400, 'days'],
    [3600, 'hours'],
    [60, 'min'],
    [1, 's'],
  ]
  for (const [s, u] of units)
    if (sec >= s) {
      const v = sec / s
      if (u === 'years' && v >= 1e4) return `${v.toExponential(1)} years`
      return `${v < 10 ? v.toFixed(1) : Math.round(v).toLocaleString()} ${u}`
    }
  return `${sec.toFixed(2)} s`
}
const fmtN = (n) => (n >= 1e9 ? `${+(n / 1e9).toFixed(1)} billion` : n >= 1e6 ? `${+(n / 1e6).toFixed(1)} million` : Math.round(n).toLocaleString())
const fmtT = (t) => (t < 1 ? `${t} s` : human(t))
</script>

<template>
  <div class="speed">
    <div class="chart">
      <div class="head">
        <span>Inference time per object</span>
        <label class="tog"><input v-model="gpu" type="checkbox" /> GPU for SBI</label>
      </div>
      <svg :width="W" :height="rows.length * ROW + 30">
        <g v-for="t in ticks" :key="t">
          <line :x1="sx(t)" :x2="sx(t)" y1="0" :y2="rows.length * ROW" class="gridline" />
          <text :x="sx(t)" :y="rows.length * ROW + 14" class="tick">{{ tickLabel(t) }}</text>
        </g>
        <g v-for="(r, i) in rows" :key="r.name" :transform="`translate(0, ${i * ROW})`">
          <text x="0" :y="ROW / 2 - (r.note ? 2 : -4)" class="name" :class="{ sbi: r.sbi }">{{ r.name }}</text>
          <text v-if="r.note" x="0" :y="ROW / 2 + 11" class="note">{{ r.note }}</text>
          <rect :x="L" :y="7" :width="sx(r.t) - L" :height="ROW - 14" rx="3" :class="['bar', { sbi: r.sbi }]" />
          <text :x="sx(r.t) + 5" :y="ROW / 2 + 4" class="val">{{ r.lower ? '> ' : r.approx ? '~ ' : '' }}{{ fmtT(r.t) }}</text>
        </g>
        <text :x="(L + W) / 2" :y="rows.length * ROW + 28" class="axis-lbl">seconds (log scale)</text>
      </svg>
    </div>

    <div class="survey">
      <div class="head">Total time{{ gpu ? ' (SBI on GPU)' : ' (single core)' }} for</div>
      <div class="bign">{{ fmtN(N) }} <span>objects</span></div>
      <input v-model.number="logN" type="range" min="2" max="10.5" step="0.01" />
      <div class="presets">
        <button v-for="s in SURVEYS" :key="s.label" @click="logN = Math.log10(s.n)">{{ s.label }}</button>
      </div>
      <table>
        <tr v-for="r in rows" :key="r.name" :class="{ sbi: r.sbi }">
          <td>{{ r.name }}</td>
          <td class="num">{{ r.lower ? '> ' : '' }}{{ human(r.t * N) }}</td>
        </tr>
      </table>
    </div>
  </div>
</template>

<style scoped>
.speed {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 40px;
  align-items: start;
}
.head {
  display: flex;
  justify-content: space-between;
  font-size: 0.8rem;
  color: var(--muted);
  margin-bottom: 8px;
}
.tog {
  display: flex;
  gap: 5px;
  align-items: center;
}
input[type='checkbox'],
input[type='range'] {
  accent-color: var(--accent);
}
input[type='range'] {
  width: 100%;
}
svg {
  overflow: visible;
}
.gridline {
  stroke: var(--grid);
}
.tick {
  font-size: 9px;
  fill: var(--muted);
  text-anchor: middle;
}
.axis-lbl {
  font-size: 10px;
  fill: var(--muted);
  text-anchor: middle;
}
.name {
  font-size: 12px;
  fill: var(--ink);
}
.name.sbi {
  font-weight: 600;
  fill: var(--accent);
}
.note {
  font-size: 9px;
  fill: var(--muted);
}
.bar {
  fill: #cbd5e1;
  transition: width 0.4s;
}
.bar.sbi {
  fill: var(--accent);
}
.val {
  font-size: 10px;
  fill: var(--ink);
  font-variant-numeric: tabular-nums;
}
.bign {
  font-size: 1.6rem;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}
.bign span {
  font-size: 0.9rem;
  font-weight: 400;
  color: var(--muted);
}
.presets {
  display: flex;
  gap: 6px;
  margin: 4px 0 10px;
}
.presets button {
  font-size: 0.65rem;
  padding: 1px 8px;
  border: 1px solid var(--line);
  border-radius: 999px;
  color: var(--muted);
}
.presets button:hover {
  border-color: var(--accent);
  color: var(--accent);
}
table {
  width: 100%;
  font-size: 0.75rem;
  border-collapse: collapse;
}
td {
  padding: 3px 0;
  border-bottom: 1px solid var(--grid);
}
tr.sbi td {
  color: var(--accent);
  font-weight: 600;
}
.num {
  text-align: right;
  font-variant-numeric: tabular-nums;
}
</style>
