// A tiny mixture density network, q(theta | x) = sum_k pi_k N(theta; mu_k, s_k),
// with pi, mu, log s output by a one-hidden-layer tanh network of x.
// Trained full-batch with Adam on all simulations: neural posterior
// estimation in miniature, small enough to run live in the browser.
// trainMDN returns a plain (structured-cloneable) model so it can be trained
// in a Web Worker; evalMDN evaluates q(theta | x0) on a theta grid.

import { mulberry32 } from './toy.js'

const K = 3 // mixture components
const H = 24 // hidden units
const NOUT = 3 * K

export function trainMDN(sims, { iters = 500, lr = 0.02, seed = 3, maxTrain = 3000 } = {}) {
  const rand = mulberry32(seed)
  const data = sims.length > maxTrain ? sims.slice(0, maxTrain) : sims
  const n = data.length

  // standardise inputs and targets
  const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length
  const std = (a, m) => Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length) || 1
  const ts = data.map((d) => d[0])
  const xs = data.map((d) => d[1])
  const tm = mean(ts)
  const tsd = std(ts, tm)
  const xm = mean(xs)
  const xsd = std(xs, xm)
  const z = ts.map((t) => (t - tm) / tsd)
  const u = xs.map((x) => (x - xm) / xsd)

  // parameters: W1 (H), b1 (H), W2 (NOUT x H), b2 (NOUT)
  const gauss = () => Math.sqrt(-2 * Math.log(rand() || 1e-12)) * Math.cos(2 * Math.PI * rand())
  const P = {
    W1: Float64Array.from({ length: H }, () => gauss() * 1.0),
    b1: Float64Array.from({ length: H }, () => gauss() * 0.5),
    W2: Float64Array.from({ length: NOUT * H }, () => gauss() * 0.1),
    b2: new Float64Array(NOUT),
  }
  // spread the initial component means
  for (let k = 0; k < K; k++) P.b2[K + k] = -1 + (2 * k) / (K - 1)

  const names = Object.keys(P)
  const m1 = Object.fromEntries(names.map((k) => [k, new Float64Array(P[k].length)]))
  const m2 = Object.fromEntries(names.map((k) => [k, new Float64Array(P[k].length)]))
  const G = Object.fromEntries(names.map((k) => [k, new Float64Array(P[k].length)]))
  const h = new Float64Array(H)
  const o = new Float64Array(NOUT)
  const dout = new Float64Array(NOUT)
  const dh = new Float64Array(H)
  const b1 = 0.9
  const b2 = 0.999

  for (let it = 1; it <= iters; it++) {
    for (const k of names) G[k].fill(0)
    for (let i = 0; i < n; i++) {
      forward(P, u[i], h, o)
      // log-sum-exp over components
      let amax = -Infinity
      for (let k = 0; k < K; k++) amax = Math.max(amax, o[k])
      let asum = 0
      for (let k = 0; k < K; k++) asum += Math.exp(o[k] - amax)
      const logw = []
      let lmax = -Infinity
      for (let k = 0; k < K; k++) {
        const ls = clampLogS(o[2 * K + k])
        const r = (z[i] - o[K + k]) / Math.exp(ls)
        const lw = o[k] - amax - Math.log(asum) - ls - 0.5 * r * r
        logw.push(lw)
        lmax = Math.max(lmax, lw)
      }
      let wsum = 0
      for (let k = 0; k < K; k++) wsum += Math.exp(logw[k] - lmax)
      // gradients of the negative log-likelihood w.r.t. network outputs
      for (let k = 0; k < K; k++) {
        const gamma = Math.exp(logw[k] - lmax) / wsum
        const pi = Math.exp(o[k] - amax) / asum
        const ls = clampLogS(o[2 * K + k])
        const s2 = Math.exp(2 * ls)
        const d = z[i] - o[K + k]
        dout[k] = pi - gamma
        dout[K + k] = (-gamma * d) / s2
        dout[2 * K + k] = gamma * (1 - (d * d) / s2)
      }
      // backprop
      dh.fill(0)
      for (let j = 0; j < NOUT; j++) {
        G.b2[j] += dout[j]
        for (let q = 0; q < H; q++) {
          G.W2[j * H + q] += dout[j] * h[q]
          dh[q] += dout[j] * P.W2[j * H + q]
        }
      }
      for (let q = 0; q < H; q++) {
        const da = dh[q] * (1 - h[q] * h[q])
        G.W1[q] += da * u[i]
        G.b1[q] += da
      }
    }
    // Adam step on the mean gradient
    for (const k of names) {
      const p = P[k]
      for (let j = 0; j < p.length; j++) {
        const g = G[k][j] / n
        m1[k][j] = b1 * m1[k][j] + (1 - b1) * g
        m2[k][j] = b2 * m2[k][j] + (1 - b2) * g * g
        const mh = m1[k][j] / (1 - b1 ** it)
        const vh = m2[k][j] / (1 - b2 ** it)
        p[j] -= (lr * mh) / (Math.sqrt(vh) + 1e-8)
      }
    }
  }

  return { P, tm, tsd, xm, xsd, n }
}

// q(theta | x0) on a theta grid, in the original units
export function evalMDN({ P, tm, tsd, xm, xsd }, x0, thetaGrid) {
  const h = new Float64Array(H)
  const o = new Float64Array(NOUT)
  forward(P, (x0 - xm) / xsd, h, o)
  let amax = -Infinity
  for (let k = 0; k < K; k++) amax = Math.max(amax, o[k])
  let asum = 0
  for (let k = 0; k < K; k++) asum += Math.exp(o[k] - amax)
  return thetaGrid.map((t) => {
    const zz = (t - tm) / tsd
    let p = 0
    for (let k = 0; k < K; k++) {
      const s = Math.exp(clampLogS(o[2 * K + k]))
      const r = (zz - o[K + k]) / s
      p += ((Math.exp(o[k] - amax) / asum) * Math.exp(-0.5 * r * r)) / (s * Math.sqrt(2 * Math.PI))
    }
    return p / tsd
  })
}

const clampLogS = (v) => Math.min(2, Math.max(-4, v))

function forward(P, x, h, o) {
  for (let q = 0; q < H; q++) h[q] = Math.tanh(P.W1[q] * x + P.b1[q])
  for (let j = 0; j < NOUT; j++) {
    let s = P.b2[j]
    for (let q = 0; q < H; q++) s += P.W2[j * H + q] * h[q]
    o[j] = s
  }
}
