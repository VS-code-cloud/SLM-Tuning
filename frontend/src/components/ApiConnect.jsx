import { useEffect, useState } from 'react'
import { getApiUrl, setApiUrl, health } from '../api.js'

// status: 'unknown' | 'checking' | 'ok' | 'down'
export default function ApiConnect({ onStatus }) {
  const [url, setUrl] = useState(getApiUrl())
  const [status, setStatus] = useState('unknown')
  const [detail, setDetail] = useState('')

  async function check() {
    if (!getApiUrl()) { setStatus('unknown'); setDetail(''); onStatus?.('unknown'); return }
    setStatus('checking'); onStatus?.('checking')
    try {
      const h = await health()
      setStatus('ok'); onStatus?.('ok')
      setDetail(h.mock ? 'connected · mock model' : `connected · ${h.device}${h.model_loaded ? '' : ' · no model'}`)
    } catch {
      setStatus('down'); onStatus?.('down'); setDetail('unreachable')
    }
  }

  useEffect(() => { check() /* on mount */ /* eslint-disable-next-line */ }, [])

  function save() {
    setApiUrl(url)
    check()
  }

  return (
    <div className="api-connect">
      <label className="api-connect__label" htmlFor="server-url">Server URL</label>
      <div className="api-connect__row">
        <span className={'status-dot status-dot--' + status} title={status} />
        <input
          id="server-url"
          className="api-connect__input"
          type="text"
          placeholder="https://xxxx.ngrok-free.app  (from the Colab notebook)"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && save()}
          spellCheck={false}
        />
        <button className="btn btn--ghost" onClick={save}>Connect</button>
      </div>
      <p className="api-connect__detail">
        {status === 'ok' && <span className="ok-text">● {detail}</span>}
        {status === 'down' && <span className="down-text">● {detail} — check the URL and that the notebook cell is running</span>}
        {status === 'checking' && <span>● checking…</span>}
        {status === 'unknown' && <span>Paste the ngrok or trycloudflare URL printed by the server notebook, then Connect.</span>}
      </p>
    </div>
  )
}
