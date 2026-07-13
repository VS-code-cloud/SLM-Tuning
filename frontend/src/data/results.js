// Held-out clean-eval numbers (eval_items_clean.json, bf16 Qwen3.5-4B, n=60/family).
// LogicSLM = v7 (LGMT full-taxonomy rebalance); Opus/Sonnet = the same clean eval.
// Mean accuracy across six logical-reasoning families.
export const RESULTS = [
  { model: 'Claude Opus 4.8', kind: 'opus', params: 'frontier', acc: 87.2 },
  { model: 'LogicSLM', kind: 'slm', params: '4B params', acc: 85.0, highlight: true },
  { model: 'Claude Sonnet 4.6', kind: 'sonnet', params: 'frontier', acc: 79.6 },
]

export const RESULTS_NOTE =
  'Mean accuracy over six logical-reasoning families (FOLIO, LogicNLI, ProverQA, ' +
  'ReClor/LSAT, adversarial ARCT, LogiQA), n=60 per family, on a held-out set. ' +
  'A 4B open model lands between Sonnet and Opus.'

// Neutral / "does-not-follow" recall on the entailment families (same clean eval).
// This is the class frontier models most often get wrong by over-committing — and
// where LogicSLM matches or beats Opus. slm/opus are % of neutral-gold items recalled.
export const NEUTRAL = [
  { family: 'FOLIO', slm: 86.7, opus: 80.0 },
  { family: 'ProverQA', slm: 96.7, opus: 90.0 },
  { family: 'LogicNLI', slm: 90.0, opus: 90.0 },
]

export const NEUTRAL_NOTE =
  'The hardest judgment is "the premises don’t settle this." LogicSLM leads Opus on that ' +
  'class in FOLIO and ProverQA (+6.7 each) and ties on LogicNLI — and it beats Opus on ' +
  'ProverQA overall (90.0 vs 80.0).'
