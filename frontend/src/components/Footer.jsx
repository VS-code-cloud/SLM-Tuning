export default function Footer() {
  return (
    <footer className="footer">
      <div className="footer__inner">
        <div>
          <span className="footer__brand">LogicSLM</span>
          <p className="footer__tag">A critical-reasoning small language model.</p>
        </div>
        <div className="footer__links">
          {/* Replace with your real links when published */}
          <a href="#" rel="noreferrer">Model card ↗</a>
          <a href="#" rel="noreferrer">Source ↗</a>
          <a href="#compare">Try the comparison</a>
        </div>
      </div>
      <p className="footer__fine">
        The comparison runs live: LogicSLM from a Colab-hosted server, Opus and Sonnet through the
        project's gateway. Every model receives the identical prompt.
      </p>
    </footer>
  )
}
