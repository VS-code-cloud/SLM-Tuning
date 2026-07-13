# LogicSLM server (Colab)

A single self-contained notebook, **`logicslm_server.ipynb`**, that serves the critical-reasoning SLM
and proxies Claude Opus 4.8 / Sonnet 4.6 through the project gateway. The React app in
[`../frontend`](../frontend) calls one endpoint and shows all three side by side. Secrets stay on the
server — the browser never sees them.

## Run it (GPU)

1. Open `logicslm_server.ipynb` in Google Colab, set the runtime to **GPU**.
2. Run the cells top to bottom. They: install deps → mount Drive → load secrets → inline the small
   `slm_core` helpers → load `Qwen/Qwen3.5-4B` + the LoRA adapter → start Flask behind a public tunnel.
3. The last cell prints a public URL. **Copy it into the frontend's "Server URL" box.** That cell
   blocks while the server runs — leave it running; interrupt to restart.

### Tunnel choice (`TUNNEL` in the config cell)

- **`"cloudflare"` (default)** — a Cloudflare quick tunnel (`https://xxxx.trycloudflare.com`). No
  signup, **no browser interstitial**, and CORS passes straight through — the right choice for a
  browser frontend. Downloads the `cloudflared` binary on first run.
- **`"ngrok"`** — uses `NGROK_AUTH_TOKEN`, prints `https://xxxx.ngrok-free.app`. The free tier serves a
  one-time interstitial page; the frontend bypasses it by sending an `ngrok-skip-browser-warning`
  header, so the app works — but a direct browser visit to the URL still shows the interstitial once.

The model is loaded from the same Drive location as `colab_standalone.py`:
`Qwen/Qwen3.5-4B` + adapter at `/content/drive/MyDrive/SLM/runs/colab/adapter`. Edit `ADAPTER_DIR` in
the config cell if yours differs.

## Secrets

`.env` is gitignored, so it is **not** in the repo. Provide the four keys one of two ways (the notebook
checks both, Drive first):

- **Drive `.env`** — put a `KEY=VALUE` file at `/content/drive/MyDrive/SLM/.env` (edit `DRIVE_ENV` to
  change the path).
- **Colab Secrets** — the key icon in Colab's left sidebar; add each name and enable notebook access.

Keys used: `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN` (gateway, Bearer), `ANTHROPIC_API_KEY`
(fallback), `NGROK_AUTH_TOKEN` (tunnel). The gateway path (`BASE_URL` + `AUTH_TOKEN`) matches
`slm_core._client()`.

## API

- `GET /health` → `{ok, model_loaded, device, mock}`
- `POST /compare` → body `{ "premises": [str] | "line\nline", "conclusion": str }`
  ```json
  {
    "prompt_used": "…the identical prompt sent to all three…",
    "question": "Based only on the premises, is the conclusion True, False, or Unknown …",
    "results": [
      {"model": "LogicSLM (Qwen3.5-4B)", "kind": "slm",    "ok": true, "label": "True",  "reasoning": "…", "latency_ms": 812},
      {"model": "Claude Opus 4.8",       "kind": "opus",   "ok": true, "label": "True",  "reasoning": "…", "latency_ms": 1503},
      {"model": "Claude Sonnet 4.6",     "kind": "sonnet", "ok": true, "label": "True",  "reasoning": "…", "latency_ms": 1104}
    ]
  }
  ```
  All three models receive the **same** `prompt_used` (built with `slm_core.frq_prompt`), so the
  comparison is apples-to-apples. A frontier failure returns `{"ok": false, "error": "unavailable"}`
  for that model instead of failing the whole request.

## Fast wiring check (no GPU)

Set `MOCK_SLM = True` in the config cell and run on a CPU runtime. LogicSLM returns a stub answer while
Opus/Sonnet are still called for real — enough to exercise the frontend, secrets, gateway, and ngrok
end to end without loading the 4B.

## Notes

- The `slm_core` helpers (`frq_prompt`, `parse_label`, `_client`, `call_agent`) are inlined verbatim
  from [`../slm_core.py`](../slm_core.py) (source lines cited in the cell) so the notebook needs nothing
  from the repo. If you change those in `slm_core.py`, update the inlined copies.
- Requires the trained adapter to already exist at `ADAPTER_DIR` (training writes it there).
- ngrok free tier allows one tunnel; the launch cell calls `ngrok.kill()` first to clear stale ones.
