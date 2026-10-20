// Toy simulator used by the interactive figures:
//   theta ~ prior,   x = f(theta) + N(0, sigma),
// with f a softplus, so x is insensitive to theta below ~0.1 and linear above.

export const THETA_RANGE = [-0.5, 1.5]
export const X_RANGE = [-0.5, 1.5]

const SQRT2PI = Math.sqrt(2 * Math.PI)
const sigmoid = (z) => 1 / (1 + Math.exp(-z))

export const gauss = (z, mu, s) => Math.exp(-0.5 * ((z - mu) / s) ** 2) / (s * SQRT2PI)

// box between a and b with logistic edges of width w, normalised to ~1
const smoothBox = (z, a, b, w) => (sigmoid((z - a) / w) - sigmoid((z - b) / w)) / (b - a)

export const PRIORS = {
  bimodal: {
    label: 'Bimodal',
    pdf: (t) => 0.35 * gauss(t, 0, 0.08) + 0.65 * smoothBox(t, -0.15, 1.25, 0.04),
  },
  uniform: {
    label: 'Uniform',
    pdf: (t) => smoothBox(t, -0.45, 1.45, 0.015),
  },
  informative: {
    label: 'Informative',
    pdf: (t) => gauss(t, 0.6, 0.12),
  },
}

const S = 0.07
const T0 = 0.05
export const f = (t) => {
  const z = (t - T0) / S
  return S * (z > 30 ? z : Math.log1p(Math.exp(z)))
}

export const linspace = (a, b, n) => Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1))

const normalise = (xs, ys) => {
  const dx = xs[1] - xs[0]
  const total = ys.reduce((s, y) => s + y, 0) * dx
  return total > 0 ? ys.map((y) => y / total) : ys
}

export const priorCurve = (prior, ts) => normalise(ts, ts.map(PRIORS[prior].pdf))

export const likelihoodCurve = (theta0, sigma, xs) => xs.map((x) => gauss(x, f(theta0), sigma))

export const posteriorCurve = (x0, sigma, prior, ts) =>
  normalise(ts, ts.map((t) => gauss(x0, f(t), sigma) * PRIORS[prior].pdf(t)))

// joint density p(theta, x) on an (nt x nx) grid, row-major in x
export function jointGrid(sigma, prior, nt, nx) {
  const ts = linspace(...THETA_RANGE, nt)
  const xs = linspace(...X_RANGE, nx)
  const p = ts.map(PRIORS[prior].pdf)
  const fx = ts.map(f)
  const out = new Float32Array(nt * nx)
  let max = 0
  for (let j = 0; j < nx; j++)
    for (let i = 0; i < nt; i++) {
      const v = p[i] * gauss(xs[j], fx[i], sigma)
      out[j * nt + i] = v
      if (v > max) max = v
    }
  return { values: out, max, nt, nx }
}

// small seeded RNG so the "simulations" are reproducible between renders
export function mulberry32(seed) {
  return () => {
    seed |= 0
    seed = (seed + 0x6d2b79f5) | 0
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

// draw n (theta, x) pairs: theta by inverse-CDF on a fine grid, then simulate x
export function simulate(n, sigma, prior, seed = 1) {
  const rand = mulberry32(seed)
  const ts = linspace(...THETA_RANGE, 2000)
  const cdf = []
  let c = 0
  for (const t of ts) cdf.push((c += PRIORS[prior].pdf(t)))
  const sims = []
  for (let k = 0; k < n; k++) {
    const u = rand() * c
    let lo = 0
    let hi = cdf.length - 1
    while (lo < hi) {
      const mid = (lo + hi) >> 1
      if (cdf[mid] < u) lo = mid + 1
      else hi = mid
    }
    const theta = ts[lo]
    const g = Math.sqrt(-2 * Math.log(rand() || 1e-12)) * Math.cos(2 * Math.PI * rand())
    sims.push([theta, f(theta) + sigma * g])
  }
  return sims
}

// histogram of theta for simulations with |x - x0| < eps (rejection ABC)
export function abcPosterior(sims, x0, eps, nbins = 40) {
  const [a, b] = THETA_RANGE
  const w = (b - a) / nbins
  const counts = new Array(nbins).fill(0)
  let kept = 0
  for (const [t, x] of sims)
    if (Math.abs(x - x0) < eps) {
      counts[Math.min(nbins - 1, Math.floor((t - a) / w))]++
      kept++
    }
  const edges = linspace(a, b, nbins + 1)
  const density = counts.map((c) => (kept ? c / (kept * w) : 0))
  return { edges, density, kept }
}
