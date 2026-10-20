import { trainMDN } from './mdn.js'

self.onmessage = ({ data }) => {
  const t0 = performance.now()
  const model = trainMDN(data.sims)
  self.postMessage({ id: data.id, model, ms: performance.now() - t0 })
}
