// Tiny client for the LogicSLM Flask server. The base URL is resolved at call
// time from localStorage (set via the "Server URL" field) or VITE_API_URL.

const LS_KEY = 'logicslm_api_url'

export function getApiUrl() {
  const stored = (localStorage.getItem(LS_KEY) || '').trim()
  const fallback = (import.meta.env.VITE_API_URL || '').trim()
  return (stored || fallback).replace(/\/+$/, '')
}

export function setApiUrl(url) {
  localStorage.setItem(LS_KEY, (url || '').trim())
}

async function req(path, opts = {}, timeoutMs = 90000) {
  const base = getApiUrl()
  if (!base) throw new Error('No server URL set. Paste the tunnel URL from the Colab notebook.')
  const ctrl = new AbortController()
  const t = setTimeout(() => ctrl.abort(), timeoutMs)
  // `ngrok-skip-browser-warning` bypasses the ngrok free-tier interstitial (which is
  // served without CORS headers and would otherwise block the request). Harmless on
  // Cloudflare tunnels. Only send Content-Type when there's a body, to keep preflight lean.
  const headers = { 'ngrok-skip-browser-warning': 'true', ...(opts.headers || {}) }
  if (opts.body) headers['Content-Type'] = 'application/json'
  try {
    const res = await fetch(base + path, { ...opts, signal: ctrl.signal, headers })
    const data = await res.json().catch(() => ({}))
    if (!res.ok) throw new Error(data.error || `Server error (${res.status})`)
    return data
  } catch (e) {
    if (e.name === 'AbortError') throw new Error('Request timed out.')
    // A network/CORS failure here usually means the URL is wrong or the tunnel is down.
    if (e instanceof TypeError) throw new Error('Could not reach the server (URL wrong, tunnel down, or blocked).')
    throw e
  } finally {
    clearTimeout(t)
  }
}

// Returns {ok, model_loaded, device, mock}. Short timeout — it's just a ping.
export async function health() {
  const d = await req('/health', { method: 'GET' }, 8000)
  if (!d || d.ok !== true) throw new Error('Unexpected response — is this a LogicSLM server URL?')
  return d
}

// body: { premises: string[], conclusion: string } -> { prompt_used, question, results[] }
export function compare(body) {
  return req('/compare', { method: 'POST', body: JSON.stringify(body) })
}
