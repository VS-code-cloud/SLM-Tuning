import { useState } from 'react'
import { compare } from '../api.js'
import { EXAMPLES } from '../data/examples.js'
import ModelCard from './ModelCard.jsx'
import ApiConnect from './ApiConnect.jsx'

const PLACEHOLDERS = [
  { model: 'LogicSLM (Qwen3.5-4B)', kind: 'slm' },
  { model: 'Claude Opus 4.8', kind: 'opus' },
  { model: 'Claude Sonnet 4.6', kind: 'sonnet' },
]

export default function Compare() {
  const [premisesText, setPremisesText] = useState(EXAMPLES[0].premises.join('\n'))
  const [conclusion, setConclusion] = useState(EXAMPLES[0].conclusion)
  const [status, setStatus] = useState('unknown')
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState(null)
  const [promptUsed, setPromptUsed] = useState('')
  const [error, setError] = useState('')

  function loadExample(ex) {
    setPremisesText(ex.premises.join('\n'))
    setConclusion(ex.conclusion)
    setResults(null)
    setError('')
  }

  async function run() {
    setError('')
    const premises = premisesText.split('\n').map((s) => s.trim()).filter(Boolean)
    if (!premises.length || !conclusion.trim()) {
      setError('Enter at least one premise (one per line) and a conclusion.')
      return
    }
    setLoading(true)
    setResults(null)
    try {
      const data = await compare({ premises, conclusion })
      setResults(data.results || [])
      setPromptUsed(data.prompt_used || '')
    } catch (e) {
      setError(e.message || 'Request failed.')
    } finally {
      setLoading(false)
    }
  }

  const canRun = status === 'ok' && !loading

  return (
    <section className="section compare" id="compare">
      <div className="section__head">
        <p className="eyebrow">Try it</p>
        <h2 className="section__title">Compare LogicSLM to frontier models</h2>
        <p className="section__intro">
          Enter the premises of an argument and the conclusion it's meant to support. LogicSLM,
          Claude&nbsp;Opus&nbsp;4.8, and Claude&nbsp;Sonnet&nbsp;4.6 each get the <em>identical</em> prompt
          and decide whether the conclusion is <strong>True</strong>, <strong>False</strong>, or
          <strong> Unknown</strong> (does not follow either way).
        </p>
      </div>

      <ApiConnect onStatus={setStatus} />

      <div className="presets">
        <span className="presets__label">Examples:</span>
        {EXAMPLES.map((ex) => (
          <button key={ex.id} className="preset-chip" onClick={() => loadExample(ex)} title={ex.demonstrates}>
            {ex.title} <span className="preset-chip__ctx">· {ex.context}</span>
          </button>
        ))}
      </div>

      <div className="compare__form">
        <div className="field">
          <label className="field__label" htmlFor="premises">Premises <span className="field__hint">one per line</span></label>
          <textarea
            id="premises"
            className="field__input field__input--area"
            rows={5}
            value={premisesText}
            onChange={(e) => setPremisesText(e.target.value)}
            placeholder={'All whales are mammals.\nAll mammals are warm-blooded.'}
          />
        </div>
        <div className="field">
          <label className="field__label" htmlFor="conclusion">Conclusion</label>
          <textarea
            id="conclusion"
            className="field__input field__input--area"
            rows={2}
            value={conclusion}
            onChange={(e) => setConclusion(e.target.value)}
            placeholder={'All whales are warm-blooded.'}
          />
        </div>
        <div className="compare__actions">
          <button className="btn btn--primary" onClick={run} disabled={!canRun}>
            {loading ? 'Comparing…' : 'Compare'}
          </button>
          {status !== 'ok' && (
            <span className="compare__hint">Connect a server URL above to run the comparison.</span>
          )}
        </div>
        {error && <p className="compare__error">{error}</p>}
      </div>

      {(loading || results) && (
        <>
          <div className="results-grid">
            {(loading ? PLACEHOLDERS : results).map((r, i) => (
              <ModelCard key={i} result={r} loading={loading} />
            ))}
          </div>
          {!loading && promptUsed && (
            <details className="prompt-reveal">
              <summary>Show the exact prompt sent to all three models</summary>
              <pre className="prompt-reveal__pre">{promptUsed}</pre>
            </details>
          )}
        </>
      )}
    </section>
  )
}
