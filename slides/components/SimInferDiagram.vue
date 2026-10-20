<script setup>
// Simulation vs inference diagram series (adapted from Matt Ho).
//   0: forward model   1: simulator gives (x, theta) pairs
//   2: inference + model comparison   3: inference box -> machine learning
defineProps({ step: { type: Number, default: 99 } })

// trapezoid used for "parameters" nodes (centre cx, top y)
const trap = (cx, y, w = 140, h = 50, inset = 9) =>
  `${cx - w / 2 + inset},${y} ${cx + w / 2 - inset},${y} ${cx + w / 2},${y + h} ${cx - w / 2},${y + h}`
</script>

<template>
  <svg viewBox="0 0 900 400" class="diagram">
    <defs>
      <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b" />
      </marker>
      <linearGradient id="ml" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0" stop-color="#fff7d6" />
        <stop offset="1" stop-color="#fcd34d" />
      </linearGradient>
    </defs>

    <!-- forward model -->
    <g class="part" :class="{ on: step >= 0 }">
      <polygon :points="trap(110, 30)" class="param" />
      <text x="110" y="61" class="lbl">Parameters <tspan class="sym">θ</tspan></text>
      <line x1="110" y1="80" x2="110" y2="128" class="edge" marker-end="url(#arr)" />
      <rect x="55" y="130" width="110" height="100" class="sim" />
      <text x="110" y="185" class="lbl light">Simulator</text>
      <line x1="110" y1="230" x2="110" y2="278" class="edge" marker-end="url(#arr)" />
      <rect x="40" y="280" width="140" height="56" rx="12" class="data" />
      <text x="110" y="304" class="lbl">Observed</text>
      <text x="110" y="324" class="lbl">data <tspan class="sym">x</tspan></text>
    </g>

    <!-- joint samples -->
    <g class="part" :class="{ on: step >= 1 }">
      <rect x="255" y="70" width="110" height="100" class="sim" />
      <text x="310" y="125" class="lbl light">Simulator</text>
      <path d="M310 170 V 205 M245 205 H 375" class="edge" />
      <line x1="245" y1="205" x2="245" y2="238" class="edge" marker-end="url(#arr)" />
      <line x1="375" y1="205" x2="375" y2="238" class="edge" marker-end="url(#arr)" />
      <rect x="185" y="240" width="120" height="56" rx="12" class="data" />
      <text x="245" y="264" class="lbl">Observed</text>
      <text x="245" y="284" class="lbl">data <tspan class="sym">x</tspan></text>
      <polygon :points="trap(375, 243, 128, 50)" class="param" />
      <text x="375" y="274" class="lbl">Parameters <tspan class="sym">θ</tspan></text>
      <text x="310" y="330" class="note">sample the joint <tspan class="sym">P(x, θ)</tspan></text>
    </g>
    <text x="215" y="385" class="caption part" :class="{ on: step >= 0 }">Simulation</text>

    <line x1="465" y1="20" x2="465" y2="390" class="divider part" :class="{ on: step >= 2 }" />

    <!-- inference -->
    <g class="part" :class="{ on: step >= 2 }">
      <rect x="500" y="30" width="130" height="56" rx="12" class="data" />
      <text x="565" y="54" class="lbl">Observed</text>
      <text x="565" y="74" class="lbl">data <tspan class="sym">x</tspan></text>
      <line x1="565" y1="86" x2="565" y2="133" class="edge" marker-end="url(#arr)" />

      <g class="swap" :class="{ out: step >= 3 }">
        <rect x="505" y="135" width="120" height="90" class="infer" />
        <text x="565" y="175" class="lbl">Bayesian</text>
        <text x="565" y="195" class="lbl">Inference</text>
      </g>
      <g class="swap" :class="{ out: step < 3 }">
        <rect x="500" y="130" width="130" height="100" class="ml" />
        <text x="565" y="175" class="lbl ml-txt">Machine</text>
        <text x="565" y="198" class="lbl ml-txt">Learning</text>
      </g>

      <line x1="565" y1="230" x2="565" y2="278" class="edge" marker-end="url(#arr)" />
      <polygon :points="trap(565, 280)" class="param" />
      <text x="565" y="311" class="lbl">Parameters <tspan class="sym">θ</tspan></text>

      <rect x="650" y="30" width="245" height="90" class="bracket" />
      <rect x="660" y="45" width="100" height="56" rx="12" class="data" />
      <text x="710" y="69" class="lbl">Observed</text>
      <text x="710" y="89" class="lbl">data <tspan class="sym">x</tspan></text>
      <polygon :points="trap(828, 48, 124, 50)" class="param" />
      <text x="828" y="78" class="lbl">Parameters <tspan class="sym">θ</tspan></text>
      <path d="M745 120 L 745 150 L 725 150 L 772 195 L 819 150 L 799 150 L 799 120" class="funnel" />
      <rect x="670" y="205" width="80" height="60" class="infer" />
      <text x="710" y="230" class="lbl">Model</text>
      <text x="710" y="250" class="lbl">1</text>
      <text x="772" y="242" class="q">?</text>
      <rect x="795" y="205" width="80" height="60" class="infer" />
      <text x="835" y="230" class="lbl">Model</text>
      <text x="835" y="250" class="lbl">2</text>
      <text x="772" y="300" class="note">model comparison</text>
    </g>
    <text x="680" y="385" class="caption part" :class="{ on: step >= 2 }">Inference</text>
  </svg>
</template>

<style scoped>
.diagram {
  width: 100%;
  max-height: 400px;
  font-family: var(--slidev-theme-font-sans, Inter, sans-serif);
}
.part {
  opacity: 0;
  transition: opacity 0.5s;
}
.part.on {
  opacity: 1;
}
.swap {
  transition: opacity 0.6s;
}
.swap.out {
  opacity: 0;
}
.param,
.data {
  fill: #f1f5f9;
  stroke: #94a3b8;
  stroke-width: 1.2;
}
.sim {
  fill: var(--sim);
  stroke: none;
}
.infer {
  fill: var(--infer);
  stroke: none;
}
.ml {
  fill: url(#ml);
  stroke: #d946ef;
  stroke-width: 3;
}
.ml-txt {
  font-family: 'Caveat', cursive;
  font-size: 22px;
  font-weight: 700;
}
.bracket {
  fill: none;
  stroke: #475569;
  stroke-width: 1.2;
}
.funnel {
  fill: white;
  stroke: #475569;
  stroke-width: 1.2;
}
.edge {
  stroke: #64748b;
  stroke-width: 1.4;
  fill: none;
}
.divider {
  stroke: #94a3b8;
  stroke-width: 1.5;
  stroke-dasharray: 6 6;
}
.lbl {
  font-size: 14px;
  text-anchor: middle;
  fill: #1e293b;
}
.lbl.light {
  fill: white;
  font-weight: 600;
}
.sym {
  font-weight: 700;
  font-style: italic;
  font-family: 'Source Serif 4', serif;
}
.note {
  font-size: 13px;
  text-anchor: middle;
  fill: var(--muted);
}
.q {
  font-size: 22px;
  font-weight: 700;
  text-anchor: middle;
  fill: #1e293b;
}
.caption {
  font-size: 22px;
  text-anchor: middle;
  fill: #0f172a;
}
</style>
