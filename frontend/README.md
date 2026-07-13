# LogicSLM — frontend

An editorial landing page for the critical-reasoning SLM, with a live **compare** tool that runs your
premises + conclusion through LogicSLM, Claude Opus 4.8, and Claude Sonnet 4.6 side by side. Vite +
React, plain CSS, no backend of its own — it talks to the [Colab server](../server).

## Run

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173
```

1. Start the [server notebook](../server/logicslm_server.ipynb) in Colab; it prints an ngrok URL.
2. Paste that URL into the **Server URL** field on the page and click **Connect** (the dot turns green
   when `/health` responds).
3. Pick an example or type your own premises (one per line) + a conclusion, and hit **Compare**.

Optionally set a default server URL so you don't paste it each time:

```bash
cp .env.example .env      # then set VITE_API_URL=https://xxxx.ngrok-free.app
```

## Build

```bash
npm run build             # static bundle in dist/
npm run preview           # serve the build locally
```

`dist/` is a static site — host it anywhere (GitHub Pages, Netlify, etc.). Because the ngrok URL
changes each Colab session, the in-page **Server URL** field is the intended way to point it at a live
server; `VITE_API_URL` just sets the default.

## Structure

```
src/
  App.jsx                 page composition
  index.css               editorial theme (Fraunces/Newsreader, oxblood accent)
  api.js                  fetch client: getApiUrl/setApiUrl, health(), compare()
  data/examples.js        school/debate presets for the compare tool
  data/results.js         held-out accuracy numbers (from ../../data-synthesis.md)
  components/
    Hero.jsx  UseCases.jsx  HowItWorks.jsx
    Compare.jsx  ApiConnect.jsx  ModelCard.jsx  Footer.jsx
```

Every model receives the identical prompt; the compare tool exposes it under
"Show the exact prompt sent to all three models" so the comparison is verifiably apples-to-apples.
