export default function Hero() {
  return (
    <header className="hero">
      <nav className="nav">
        <span className="nav__brand">
          <svg className="nav__mark" viewBox="0 0 64 64" width="20" height="20" aria-hidden="true">
            <circle cx="32" cy="20" r="7" />
            <circle cx="20" cy="44" r="7" />
            <circle cx="44" cy="44" r="7" />
          </svg>
          LogicSLM
        </span>
        <div className="nav__links">
          <a href="#use">For the classroom</a>
          <a href="#how">How it works</a>
          <a href="#compare">Try it</a>
        </div>
      </nav>

      <div className="hero__inner">
        <p className="eyebrow">A small language model for logical reasoning</p>
        <h1 className="hero__title">
          Reason from the premises. Commit to <em>one</em> answer.
        </h1>
        <p className="hero__lede">
          LogicSLM is a compact, fine-tuned model that judges whether a conclusion actually
          follows from its premises — and, crucially, says so plainly when it <em>doesn't</em>.
          It's built to make language models more reliable at the logical work students do:
          argumentative essays and debate.
        </p>
        <div className="hero__actions">
          <a className="btn btn--primary" href="#compare">Compare it to frontier models</a>
          <a className="btn btn--ghost" href="#how">See the results</a>
        </div>
      </div>
    </header>
  )
}
