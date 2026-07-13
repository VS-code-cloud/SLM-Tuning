function labelClass(label) {
  const l = (label || '').toLowerCase()
  if (l === 'true') return 'label-chip label-chip--true'
  if (l === 'false') return 'label-chip label-chip--false'
  if (l === 'unknown' || l === 'uncertain') return 'label-chip label-chip--unknown'
  return 'label-chip label-chip--none'
}

const KIND_TAG = { slm: 'This project', opus: 'Frontier', sonnet: 'Frontier' }

export default function ModelCard({ result, loading }) {
  const isSlm = result?.kind === 'slm'
  return (
    <article className={'model-card' + (isSlm ? ' model-card--slm' : '')}>
      <header className="model-card__head">
        <div>
          <h3 className="model-card__name">{result?.model || '—'}</h3>
          {result?.kind && <span className="model-card__tag">{KIND_TAG[result.kind]}</span>}
        </div>
        {!loading && result?.ok && (
          <span className={labelClass(result.label)}>{result.label || 'no answer'}</span>
        )}
        {!loading && result && !result.ok && <span className="label-chip label-chip--none">unavailable</span>}
      </header>

      <div className="model-card__body">
        {loading && (
          <div className="model-card__loading">
            <span className="dots"><i /><i /><i /></span> reasoning…
          </div>
        )}
        {!loading && result?.ok && (
          <p className="model-card__reasoning">{result.reasoning}</p>
        )}
        {!loading && result && !result.ok && (
          <p className="model-card__reasoning model-card__reasoning--muted">
            {result.error === 'unavailable'
              ? 'Model did not respond (gateway/credentials). It is excluded from this comparison.'
              : (result.error || 'No response.')}
          </p>
        )}
      </div>

      {!loading && result?.ok && (
        <footer className="model-card__meta">
          {typeof result.latency_ms === 'number' ? `${(result.latency_ms / 1000).toFixed(1)}s` : ''}
        </footer>
      )}
    </article>
  )
}
