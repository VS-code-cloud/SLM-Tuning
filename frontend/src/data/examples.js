// Compare-tool presets. Each is a logically/semantically structured EQUIVALENT of a
// held-out eval item where Claude Opus 4.8 answered incorrectly but the tuned LogicSLM
// answered correctly (sources: ProverQA multi-step, and the neutral "does-not-follow"
// class across FOLIO/LogicNLI/ProverQA). The surface content is re-themed for school /
// debate; the logical form and gold label are preserved. `demonstrates` describes the
// form and the typical frontier failure mode — not a guarantee of any model's output.
export const EXAMPLES = [
  {
    id: 'elimination-chain',
    title: 'Elimination chain',
    context: 'Policy debate',
    demonstrates: 'Stacked exclusive-ors resolve by elimination → True. Frontier models often retreat to Unknown.',
    premises: [
      'Every proposal is either funded or shelved, but not both.',
      'Any funded proposal is either piloted this year or deferred to the next review, but not both.',
      'The transit proposal is not shelved.',
      'The transit proposal is not deferred to the next review.',
    ],
    conclusion: 'The transit proposal is piloted this year.',
  },
  {
    id: 'similar-terms',
    title: 'Similar-sounding terms',
    context: 'History essay',
    demonstrates: 'One term is fixed; the other is never linked to the actor → Unknown. Frontier models often over-commit to False.',
    premises: [
      'The historian Alcott popularized the term "oppidum" for a large fortified Iron Age town.',
      'A "castellum" is not a town but a small Roman fort.',
    ],
    conclusion: 'Alcott popularized the term "castellum".',
  },
  {
    id: 'entailed-opposite',
    title: 'Entailed opposite',
    context: 'Source analysis',
    demonstrates: 'A short chain entails the negation of the claim → False, not merely "unclear".',
    premises: [
      'Every primary source in the collection was written before 1850.',
      'Anything written before 1850 predates the railway era.',
      'The Fenwick diary is a primary source in the collection.',
    ],
    conclusion: 'The Fenwick diary was written during the railway era.',
  },
  {
    id: 'unstated-category',
    title: 'A claim about the other category',
    context: 'Archival research',
    demonstrates: 'The premises pin down one item; the claim asserts something about a category that is never established → Unknown. Frontier models over-commit to False here.',
    premises: [
      'The Verel Codex was produced in either the northern or the southern scriptorium.',
      'Brother Anselm studied the Verel Codex.',
      'The Verel Codex is written in gold ink.',
      'No manuscript from the southern scriptorium is written in gold ink.',
    ],
    conclusion: 'Brother Anselm studied a manuscript from the southern scriptorium.',
  },
]
