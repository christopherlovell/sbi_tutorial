# SBI introduction slides

Browser-based [Slidev](https://sli.dev) deck introducing simulation-based
inference; companion to `../CAMELS_SBI_tutorial.ipynb`.

```bash
npm install
npm run dev      # live at http://localhost:3030 (presenter view: /presenter)
npm run build    # static site in dist/, e.g. for GitHub Pages
npm run export   # PDF (needs playwright-chromium)
```

Interactive figures are Vue components in `components/`; the toy simulator
behind them is `lib/toy.js`.
