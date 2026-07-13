// Compare-tool presets, verified live against the deployed model.
//   1-2 are straightforward cases (increasing difficulty) that ALL THREE models get right —
//       baseline competence, so the tool isn't only cherry-picked wins.
//   3-4 are held-out eval items where LogicSLM answers correctly but BOTH Claude Opus 4.8 and
//       Sonnet 4.6 miss. (Hand-written "clean" equivalents of these don't hold up — the frontier
//       solves the tidied version — so 3-4 use the real items verbatim.)
export const EXAMPLES = [
  {
    id: 'clear-cut',
    title: 'A clear-cut deduction',
    context: 'Essay sourcing',
    demonstrates: 'A straightforward valid inference — all three models agree it is True. Not every question is a hard case; on clear ones LogicSLM simply matches the frontier.',
    premises: [
      'Every source cited in the essay is a peer-reviewed study.',
      'The Lindqvist paper is cited in the essay.',
    ],
    conclusion: 'The Lindqvist paper is a peer-reviewed study.',
  },
  {
    id: 'dense-constraints',
    title: 'A dense web of constraints',
    context: 'Constraint reasoning',
    demonstrates: 'Several interlocking rules that resolve only by elimination — harder than a one-step inference, but all three models still reach it (True).',
    premises: [
      'Each team is seeded either high or low, but not both.',
      'Every high-seeded team receives a first-round bye.',
      'No team with a first-round bye plays on opening night.',
      'The Foxes are seeded high or low.',
      'The Foxes play on opening night.',
    ],
    conclusion: 'The Foxes are seeded low.',
  },
  {
    id: 'unstated-region',
    title: 'Only one case is pinned down',
    context: 'Does it follow?',
    demonstrates: 'The premises fix the Picuris range to New Mexico, but say nothing about whether Juan visited any Texas range — so it does not follow (Unknown). Opus and Sonnet over-commit to False; LogicSLM holds the line.',
    premises: [
      'The Picuris Mountains are a mountain range in New Mexico or Texas.',
      'Juan de Onate visited the Picuris Mountains.',
      'The Harding Pegmatite Mine, located in the Picuris Mountains, was donated.',
      'There are no mountain ranges in Texas that have mines which have been donated.',
    ],
    conclusion: 'Juan de Onate visited a mountain range in Texas.',
  },
  {
    id: 'conditional-chain',
    title: 'Follow the conditionals through',
    context: 'Deductive chain',
    demonstrates: 'A chain of conditionals actually settles it — the conclusion is True. Opus and Sonnet retreat to "Unknown"; LogicSLM commits.',
    premises: [
      'If people own at least one pet, then they do not have tidy houses.',
      'If people grew up with childhood pets, then they own at least one pet.',
      'If people hire a maid or cleaning service, then they have tidy houses.',
      'If people live in the suburbs, then they have tidy houses.',
      'Jack either does not hire a maid or cleaning service or, if he does, then he does not own at least one pet.',
    ],
    conclusion: 'Jack does not live in the suburbs.',
  },
]
