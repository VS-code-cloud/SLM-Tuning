import { RESULTS, RESULTS_NOTE, NEUTRAL, NEUTRAL_NOTE, ROBUSTNESS, ROBUSTNESS_NOTE } from '../data/results.js'

export default function HowItWorks() {
  const max = Math.max(...RESULTS.map((r) => r.acc))
  return (
    <section className="section how" id="how">
      <div className="section__head">
        <p className="eyebrow">How it works</p>
        <h2 className="section__title">Trained to reason, then commit</h2>
      </div>

      <div className="how__grid">
        <div className="how__text">
          <p>
            LogicSLM is a LoRA fine-tune of a 4B open model. It learns from formal-logic entailment
            sets (FOLIO, LogicNLI, ProverQA) and argument-analysis sets (LSAT-style reasoning,
            adversarial warrant tasks, deductive multiple-choice).
          </p>
          <p>
            The training style is deliberate: reason briefly from the premises, then state exactly one
            verdict — <strong>True</strong>, <strong>False</strong>, or <strong>Unknown</strong>. The
            reasoning never leaks the answer word early, so the model has to actually decide rather
            than hedge across possibilities.
          </p>
          <p className="how__note">{RESULTS_NOTE}</p>
        </div>

        <figure className="results">
          <figcaption className="results__cap">Held-out accuracy</figcaption>
          <table className="results__table">
            <tbody>
              {RESULTS.map((r) => (
                <tr key={r.model} className={r.highlight ? 'is-highlight' : ''}>
                  <th scope="row">
                    <span className="results__model">{r.model}</span>
                    <span className="results__params">{r.params}</span>
                  </th>
                  <td className="results__barcell">
                    <span className="results__bar" style={{ width: `${(r.acc / max) * 100}%` }} />
                  </td>
                  <td className="results__acc">{r.acc.toFixed(1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </figure>
      </div>

      <div className="neutral">
        <div className="neutral__head">
          <h3 className="neutral__title">Where it beats the frontier: recognizing what doesn’t follow</h3>
          <p className="neutral__note">{NEUTRAL_NOTE}</p>
        </div>
        <table className="neutral__table">
          <thead>
            <tr>
              <th scope="col">Neutral class, by family</th>
              <th scope="col">LogicSLM</th>
              <th scope="col">Opus 4.8</th>
            </tr>
          </thead>
          <tbody>
            {NEUTRAL.map((r) => (
              <tr key={r.family} className={r.slm > r.opus ? 'beats' : ''}>
                <th scope="row">{r.family}</th>
                <td className="neutral__slm">
                  {r.slm.toFixed(1)}
                  {r.slm > r.opus && <span className="pill">+{(r.slm - r.opus).toFixed(1)}</span>}
                  {r.slm === r.opus && <span className="pill pill--tie">tie</span>}
                </td>
                <td className="neutral__opus">{r.opus.toFixed(1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="neutral">
        <div className="neutral__head">
          <h3 className="neutral__title">…and staying consistent when a problem is reworded</h3>
          <p className="neutral__note">{ROBUSTNESS_NOTE}</p>
        </div>
        <table className="neutral__table">
          <thead>
            <tr>
              <th scope="col">Answer flip-rate (lower is better)</th>
              <th scope="col">MVR</th>
            </tr>
          </thead>
          <tbody>
            {ROBUSTNESS.map((r) => (
              <tr key={r.model} className={r.highlight ? 'beats' : ''}>
                <th scope="row">{r.model}</th>
                <td className="neutral__slm">
                  {r.mvr.toFixed(1)}%
                  {r.highlight && <span className="pill">most robust</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}
