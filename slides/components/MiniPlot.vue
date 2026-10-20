<script setup>
// Lightweight SVG line/histogram plot used by the interactive figures.
import { computed } from 'vue'
import { scaleLinear } from 'd3-scale'
import { area, line } from 'd3-shape'

const props = defineProps({
  width: { type: Number, default: 300 },
  height: { type: Number, default: 150 },
  xDomain: { type: Array, required: true },
  yDomain: { type: Array, default: null }, // null: 0 .. 1.15 * max
  curves: { type: Array, default: () => [] }, // { x, y, color, dash, fill, width }
  bars: { type: Array, default: () => [] }, // { edges, density, color }
  vlines: { type: Array, default: () => [] }, // { x, color, dash }
  title: { type: String, default: '' },
  xlabel: { type: String, default: '' },
  ylabel: { type: String, default: 'PDF' },
  xticks: { type: Number, default: 5 },
  hideXTicks: Boolean,
})

const m = { t: 6, r: 8, b: 28, l: 34 }

const yMax = computed(() => {
  let mx = 0
  for (const c of props.curves) for (const v of c.y) if (v > mx) mx = v
  for (const b of props.bars) for (const v of b.density) if (v > mx) mx = v
  return mx > 0 ? mx * 1.15 : 1
})

const sx = computed(() => scaleLinear().domain(props.xDomain).range([m.l, props.width - m.r]))
const sy = computed(() =>
  scaleLinear()
    .domain(props.yDomain ?? [0, yMax.value])
    .range([props.height - m.b, m.t]),
)

const paths = computed(() =>
  props.curves.map((c) => {
    const pts = c.x.map((x, i) => [x, c.y[i]])
    const l = line()
      .x((d) => sx.value(d[0]))
      .y((d) => sy.value(d[1]))
    const a = area()
      .x((d) => sx.value(d[0]))
      .y0(sy.value(0))
      .y1((d) => sy.value(d[1]))
    return { ...c, d: l(pts), area: c.fill ? a(pts) : null }
  }),
)

const rects = computed(() =>
  props.bars.flatMap((b) =>
    b.density.map((v, i) => ({
      x: sx.value(b.edges[i]),
      w: Math.max(0, sx.value(b.edges[i + 1]) - sx.value(b.edges[i]) - 0.5),
      y: sy.value(v),
      h: sy.value(0) - sy.value(v),
      color: b.color,
    })),
  ),
)

const xt = computed(() => sx.value.ticks(props.xticks))
const yt = computed(() => sy.value.ticks(3))
const fmt = (v) => +v.toFixed(2)
</script>

<template>
  <div class="miniplot" :style="{ width: `${width}px` }">
    <div v-if="title" class="mp-title"><Tex :expr="title" /></div>
    <svg :width="width" :height="height">
      <g class="gridlines">
        <line v-for="t in xt" :key="`gx${t}`" :x1="sx(t)" :x2="sx(t)" :y1="m.t" :y2="height - m.b" />
        <line v-for="t in yt" :key="`gy${t}`" :x1="m.l" :x2="width - m.r" :y1="sy(t)" :y2="sy(t)" />
      </g>
      <rect
        v-for="(r, i) in rects"
        :key="`r${i}`"
        :x="r.x"
        :y="r.y"
        :width="r.w"
        :height="r.h"
        :fill="r.color"
        opacity="0.35"
      />
      <g v-for="(p, i) in paths" :key="`c${i}`">
        <path v-if="p.area" :d="p.area" :fill="p.color" opacity="0.12" />
        <path
          :d="p.d"
          fill="none"
          :stroke="p.color"
          :stroke-width="p.width ?? 2"
          :stroke-dasharray="p.dash ? '5 4' : null"
        />
      </g>
      <line
        v-for="(v, i) in vlines"
        :key="`v${i}`"
        :x1="sx(v.x)"
        :x2="sx(v.x)"
        :y1="m.t"
        :y2="height - m.b"
        :stroke="v.color ?? 'var(--ink)'"
        stroke-width="1.5"
        :stroke-dasharray="v.dash ? '4 3' : null"
      />
      <g class="axis">
        <line :x1="m.l" :x2="width - m.r" :y1="height - m.b" :y2="height - m.b" />
        <line :x1="m.l" :x2="m.l" :y1="m.t" :y2="height - m.b" />
        <template v-if="!hideXTicks">
          <text v-for="t in xt" :key="`tx${t}`" :x="sx(t)" :y="height - m.b + 13" text-anchor="middle">
            {{ fmt(t) }}
          </text>
        </template>
        <text v-for="t in yt" :key="`ty${t}`" :x="m.l - 5" :y="sy(t) + 3.5" text-anchor="end">
          {{ fmt(t) }}
        </text>
      </g>
    </svg>
    <div v-if="xlabel" class="mp-xlabel"><Tex :expr="xlabel" /></div>
    <div v-if="ylabel" class="mp-ylabel" :style="{ top: `${(height - m.b) / 2 + (title ? 20 : 0)}px` }">
      {{ ylabel }}
    </div>
  </div>
</template>

<style scoped>
.miniplot {
  position: relative;
}
.mp-title {
  text-align: center;
  font-size: 0.8rem;
  color: var(--ink);
  height: 20px;
}
.mp-xlabel {
  position: absolute;
  bottom: -4px;
  width: 100%;
  padding-left: 34px;
  text-align: center;
  font-size: 0.75rem;
}
.mp-ylabel {
  position: absolute;
  left: -2px;
  transform: translate(-50%, -50%) rotate(-90deg);
  font-size: 0.65rem;
  color: var(--muted);
}
svg {
  display: block;
  overflow: visible;
}
.gridlines line {
  stroke: var(--grid);
  stroke-width: 1;
}
.axis line {
  stroke: var(--muted);
  stroke-width: 1;
}
.axis text {
  font-size: 9px;
  fill: var(--muted);
  font-variant-numeric: tabular-nums;
}
</style>
