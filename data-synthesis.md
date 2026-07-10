# Data synthesis — critical-reasoning SLM corpus

How the SFT corpus (`data/sft_train.jsonl` + held-out `data/eval_items.json`) was assembled:
where each dataset came from, **why** it was included (and at what difficulty), **why**
it is FRQ vs MCQ, and every transformation applied from raw scrape to SFT-ready rows —
including the symbolic **solver-derived working paths** (with countermodel witnesses for
the neutral class). Rebuild with `python data_construction/build_corpus.py` (deterministic
given seed+caps). The FOL/corpus-build code lives in `colab/data_construction/`; it reads
`colab/data/_raw/` and writes `colab/data/` (shared with the trainer), importing `slm_core`
from the parent. Needs `z3-solver` for the solver working-paths (absent → template fallback).

## 1. The one behavior, and two regimes that are never merged

Goal: instil **one** reliable behavior — given a self-contained logic problem, reason
briefly from the premises and **commit to exactly one credited answer**, *including*
recognizing under-determination (the neutral "does-not-follow" class), where even
frontier models are weak (RESULTS-DOK: Opus 4.8 on LogicNLI-Unknown = **31.7%**).

Data splits into two regimes, reported/trained separately (never averaged):
- **Entailment core** — FOLIO, LogicNLI, ProverQA → **FRQ label** `True/False/Uncertain|Unknown`.
  Deterministically gradable; carries the hard neutral class.
- **Argument satellite** — ReClor (LSAT LR), adversarial ARCT, LogiQA → **MCQ** option letter.

Rationale: the reasoning weakness lives in *sub-dimensions* (neutral recognition, multi-step
depth, consistency under reformulation, open-ended generation), not headline accuracy — so
we curate to expose and train those, and keep the regimes separate because recognition (MCQ)
and generation/entailment (FRQ) are different skills.

## 2. Sources, difficulty rationale, and regime

| Family | Dataset (file) | n (train) | Why included / difficulty | Regime & why |
|---|---|--:|---|---|
| `folio` | FOLIO (Han et al. 2022) train+val (`folio_{train,validation}.jsonl`) | 1284 | FOL NLI; ships FOL (val: premises+conclusion; train: premises only). The FOL lets us **solve** it. Neutral (Uncertain) is the target weak class. | FRQ — 3-way entailment label, deterministically gradable |
| `logicnli` | LogicNLI (Tian et al. 2021) dev via logi_glue (`logicnli_dev.jsonl`) | 1615 | Largest entailment family; the **flagship neutral weakness** (Unknown 31.7% for Opus). Balanced True/False/Unknown. | FRQ — 3-way; `self_contradiction` dropped (as in the LGMT paper) |
| `proverqa` | ProverQA (Qi et al. 2025) dev easy/med/hard (`proverqa_*.json`) | 1193 | Multi-step FOL with a **clean depth gradient**; ships a real Prover9 `reasoning` chain. **Easy dropped** (96.7% ceiling teaches nothing); medium+hard only. | FRQ — 3-way; the shipped chain is the model trace |
| `lsat_lr` | ReClor (Yu et al. 2020) train+val (`reclor_data.zip`) | 1216 | LSAT argument analysis. Kept **only flaw / assumption / weaken** (the gap-analysis types) — the discriminating argument-reasoning subset. | MCQ — recognition of the credited option |
| `arct` | **Adversarial** ARCT (Niven & Kao 2019) neg splits (`arct_*-adv-negated.csv`) | 1023 | Implicit-warrant task on the **de-biased** set (negation-balanced) so lexical cues can't be gamed — genuine warrant reasoning. | MCQ — 2-way warrant choice |
| `logiqa` | LogiQA 2.0 test (`logiqa2_test.txt`) | 1138 | Deductive "what can be concluded" reasoning. | MCQ — deductive inference |
| `LGMT` | follow-ups built from the 3 entailment families (train only) | 1200 | Metamorphic consistency probes (P1 reorder / P2 duplicate / P3 irrelevant), MRs varied, P3 capped — trains answer-stability under logic-preserving edits. | FRQ — inherit the source label |

(Per-family sourcing matches the sibling eval `critical-reasoning-eval` exactly — same files,
splits, task-type filters, label maps, and the adversarial-ARCT variant — so the SLM trains on
the same *kind* of data the frontier eval measures.)

## 3. Pipeline: raw scrape → SFT rows

1. **Parse + filter** (`build_corpus.py` `parse_*`): task-type filter for ReClor
   (flaw/assumption/weaken); label maps (`entailment/contradiction/neutral → True/False/Unknown`,
   drop `self_contradiction`; ProverQA `A/B/C → True/False/Uncertain`); FOLIO 3-way; ProverQA
   easy dropped. Per-family caps (`CAPS`) equalize format types (small pools high-%, huge pools low-%).
2. **LGMT follow-ups** from *train* entailment items only (no leakage), ~1,200 split evenly across
   folio/logicnli/proverqa with varied MRs (P3 ≤ ½).
3. **Hashed 80/20 split** (`slm_core.train_eval_split`, `sha1(seed:group_id)`) — an order-independent
   held-out slice of **every** family, stable across rebuilds (an item never moves train↔eval). The
   hash key is a **`group_id` (shared-stimulus key)**, not the row id: ARCT per-annotator duplicates,
   LogiQA/ReClor questions over one passage, and LogicNLI statements over one facts+rules block all
   carry a common `group_id` so a stimulus never straddles the split. (Splitting on id alone leaked
   ~4.8% of eval stimuli — 22.7% of ARCT — into training prompts; measured leakage is now **0**.)
   Row `id`s are separately made globally unique so distinct problems never collide.
4. **House-style verbalizer** (`verbalize`): natural prose that reasons, then a single committing
   final sentence (`…the statement is {label}.` / `…the correct option is (X).`). The reasoning avoids
   label words so the parser's last-label-hit rule always reads the committed answer. **0 / 7,469
   completions fail the grader** (verified every rebuild).
5. **Trace source, in priority order** (`to_row.trace_source`):
   - **`solver_path`** (NEW, §4) — a real symbolic derivation (FOLIO, LogicNLI).
   - **`prover_chain`** — ProverQA's shipped Prover9 chain, restyled.
   - **`grounded_template`** — grounded True/False/neutral template (fallback).

## 4. Solver-derived working paths (the FOLIO/LogicNLI upgrade)

Templates state the label without a genuine derivation — they teach *a* label, not *why it does
not follow* — so the neutral/under-determination class transfers poorly. So FOLIO and LogicNLI now
get **real derivations**, used exactly as ProverQA's chain is — baked into the deterministic corpus,
no frontier model:

- **`fol_solver.classify_path`** (z3): decides `True/False/Uncertain` from FOL and emits a path —
  the **unsat-core premises** that force True/False, and for **Uncertain a countermodel witness**
  (an atom the premises leave open, with one model where the statement holds and one where it fails).
  This is the first time the neutral class gets a *concrete* justification rather than a template.
- **FOLIO-val**: ships premises-FOL + conclusion-FOL → solved offline (self-test: solves 96%,
  agrees with gold 96%).
- **FOLIO-train**: ships premises-FOL but **no conclusion-FOL** → a cheap, cached, lexicon-constrained
  NL→FOL translation of the conclusion (`folio_concl_fol.py`, gateway Haiku), then solved like val
  (sample: ~75% solve-to-gold).
- **LogicNLI**: no FOL shipped → a **separate** best-effort NL→FOL parser (`logicnli_fol.py`) for its
  closed synthetic grammar, then grounded to finite-domain propositional and solved.
- **Safety net — solver-verification:** a solver path is kept **only if it decides AND its label equals
  the gold**; otherwise the template is used. A wrong parse/translation is dropped, never poisoning data.

**Solver-path coverage (this build):** FOLIO **797 / 1100** (val offline + train via cached
translation), LogicNLI **0 / 1500** (all blocks FOL-inconsistent under strict closure — see caveat;
all use templates). `trace_source` mix over 7,473 train rows:
**solver_path 976 · prover_chain 745 · grounded_template 5,752** (solver_path = 797 FOLIO + their
LGMT morphs). FOLIO carries the solver-path value.

### LogicNLI caveat (important)
LogicNLI's gold is computed under **bounded-depth** reasoning, not full logical closure: the *same*
facts+rules block yields entailment/neutral/contradiction/self_contradiction across its 20 hypotheses,
and blocks are routinely FOL-inconsistent under a strict reading yet aren't labelled self_contradiction.
A full-closure z3 solver therefore finds the premises **unsatisfiable** and declines: of the 100 dev
blocks, 76 fully parse and **all 76 are UNSAT** (0 consistent), so LogicNLI yields **0 solver paths**
and falls back to templates entirely. (Verified: every unsat core is a legitimate fact-vs-rule
contradiction, e.g. *"Broderick is soft" + "Broderick is not scared" + "if soft then scared"* — no
parse artifacts. A stricter parser surfaces *more* of these real contradictions, which is why solver
yield dropped as parser coverage rose.) The parser is implemented and kept separate for a possible
bounded-depth solver later; the safety net guarantees no unsound path is ever emitted. FOLIO carries
the value.

## 5. Trace-fidelity tiers
1. **Deterministic (shipped, `sft_train.jsonl`)** — templates + ProverQA chain + **solver paths** (this doc). Cheap, offline (except FOLIO-train's cached translation), gold-correct by construction.
2. **Step B (deferred)** — frontier reword of a *real* derivation (never a template): solver emits the worked path/countermodel, a cheap model naturalizes the prose. See `[[slm-next-steps]]`; delayed by request.

## 6. Known limitations
- **FOLIO-train translation** is gateway-gated and paraphrase-limited (binary predicates, nested tails);
  ~25% of translations don't solve-to-gold and fall back to templates.
- **LogicNLI** bounded-depth mismatch (§4) → **0** solver-path yield (all fully-parsed blocks are
  FOL-inconsistent under strict closure); it ships template traces only.
- **No corpus-level eval numbers are quoted here, deliberately.** The earlier 4-model suite predates
  this build's train/eval split change (the `group_id` leakage fix) and the ProverQA/FOLIO solver-path
  updates, so those figures are stale — re-run the suite on the current corpus before citing anything.
  For frontier-model reference measurements (thinking, multi-run, balanced), see
  `critical-reasoning-eval/RESULTS-DOK.md`.
