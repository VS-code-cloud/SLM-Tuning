const CASES = [
  {
    n: '01',
    title: 'Argumentative essays',
    body:
      'History and language-arts essays live or die on whether the evidence actually supports the ' +
      'thesis. LogicSLM checks the step from premises to claim, so a student can see where an ' +
      'argument is airtight and where it only looks that way.',
  },
  {
    n: '02',
    title: 'Debate clubs',
    body:
      'A good rebuttal turns on whether the opponent’s conclusion truly follows from what they ' +
      'granted. LogicSLM flags valid moves, contradictions, and inferences that simply don’t ' +
      'follow — practice for spotting them under time pressure.',
  },
  {
    n: '03',
    title: 'Recognizing what doesn’t follow',
    body:
      'The hardest call is “this is under-determined.” Frontier models often overreach here. ' +
      'LogicSLM is trained specifically to hold the line and answer Unknown when the premises don’t ' +
      'settle the question.',
  },
]

export default function UseCases() {
  return (
    <section className="section usecases" id="use">
      <div className="section__head">
        <p className="eyebrow">Where it helps</p>
        <h2 className="section__title">Logical rigor, for school settings</h2>
        <p className="section__intro">
          The same skill underlies all of these: reason carefully from what's given, then commit to a
          single, defensible verdict.
        </p>
      </div>
      <div className="usecases__grid">
        {CASES.map((c) => (
          <article className="usecase" key={c.n}>
            <span className="usecase__n">{c.n}</span>
            <h3 className="usecase__title">{c.title}</h3>
            <p className="usecase__body">{c.body}</p>
          </article>
        ))}
      </div>
    </section>
  )
}
