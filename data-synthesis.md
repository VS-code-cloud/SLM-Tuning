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
| `logicnli` | LogicNLI (Tian et al. 2021) dev via logi_glue (`logicnli_dev.jsonl`); FOL joined from the **original structured release** (`logicnli_sim/`) by index | 1615 | Largest entailment family; the **flagship neutral weakness** (Unknown 31.7% for Opus). Balanced True/False/Unknown. Now **solver-verbalized** via forward-chaining (§4). | FRQ — 3-way; `self_contradiction` dropped (as in the LGMT paper) |
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

The two datasets need **different oracles** because their gold uses different semantics:

- **FOLIO — `fol_solver.classify_path`** (z3 global entailment; the theory is consistent): decides
  `True/False/Uncertain` from FOL and emits a path — the **unsat-core premises** that force True/False,
  and for **Uncertain a countermodel witness** (an atom the premises leave open, with one model where
  the statement holds and one where it fails). First concrete justification for the neutral class.
  - **FOLIO-val**: ships premises-FOL + conclusion-FOL → solved offline (self-test: solves 96%,
    agrees with gold 96%).
  - **FOLIO-train**: ships premises-FOL but **no conclusion-FOL** → a cheap, cached, lexicon-constrained
    NL→FOL translation of the conclusion (`folio_concl_fol.py`, gateway Haiku), then solved like val
    (sample: ~75% solve-to-gold).
- **LogicNLI — per-conclusion pruning to a classically-consistent subset, then `fol_solver.classify_path`
  (z3 classical)**. FOL comes from the **original structured release** (`logicnli_sim/*_logic.json`, exact
  machine form — no NL parsing), joined to the logi_glue items by the verified index (row `i` ⇒ block
  `i//20`, statement `i%20`, matched 2000/2000) and grounded to finite-domain propositional. Because a raw
  block is inconsistent by construction (see caveat), `logicnli_prune.py` rebuilds each non-paradox
  statement into a **classically consistent ⟨Γ, q⟩**: seed Γ with the premises the forward proof actually
  used (`fol_forward.support_indices`; facts for a neutral), then greedily add back every other premise as
  long as Γ stays satisfiable AND z3's verdict still equals the gold ("keep the irrelevant premises too").
  The **stimulus is rebuilt from just the kept premises** (avg ~21 of 24). z3 then emits the real classical
  derivation — entailing premises for True/False, a countermodel witness for Unknown. Coverage **1430/1500
  (95%)**; the ~5% that can't be made consistent-and-gold-matching are **dropped** (never emitted
  inconsistent). This also makes LogicNLI usable by LGMT (`data/lgmt_sources.json` export).
- **Safety net — solver-verification:** a solver path is kept **only if it decides AND its label equals
  the gold**; otherwise the template is used. A wrong parse/translation is dropped, never poisoning data.

**Solver-path coverage (this build):** FOLIO **797 / 1100** (val offline + train via cached
translation), LogicNLI **1430 / 1430** (100% of the pruned, consistent pool; unprunable statements are
dropped upstream). `trace_source` mix over 7,414 train rows:
**solver_path 2,532 · prover_chain 745 · grounded_template 4,137** (solver_path = FOLIO + LogicNLI + their
LGMT morphs). Both entailment families now carry real, classically-valid derivations.

### LogicNLI caveat (resolved by pruning)
LogicNLI's gold comes from a **bounded, cause-to-effect** reasoner and its blocks are *deliberately*
near-paradoxical — every 20-statement block carries exactly 5 `self_contradiction` (paradox) statements,
so the 12 facts + 12 rules are **jointly UNSAT** under classical FOL. Verified two ways: z3 finds
**0/1000 blocks** (all splits) consistent, matching LogicNLI's own 5-paradox-per-block labels; e.g.
*"Broderick is soft" + "if soft then scared" + "as long as soft, … not scared"*. A whole raw block is
therefore unusable for a classical oracle (z3 refuses) or for LGMT. The fix is **per-conclusion pruning**
(above): drop the paradox statements and, for each surviving statement, keep the largest satisfiable
premise subset that still yields the gold verdict — turning each into a genuinely classical ⟨Γ, q⟩. The
old whole-block NL→FOL parser (`logicnli_fol.py`) and the forward-chainer (`fol_forward.py`, now used only
to *seed* the prune) are kept as fallbacks.

## 5. Trace-fidelity tiers
1. **Deterministic (shipped, `sft_train.jsonl`)** — templates + ProverQA chain + **solver paths** (this doc). Cheap, offline (except FOLIO-train's cached translation), gold-correct by construction.
2. **Step B (done 2026-07-11)** — real, gold-verified frontier reasoning for the families that have no solver (the MCQ satellite), plus solver-verbalized entailment. Kept out of `build_corpus` proper and applied as build-time overlays (`*_stepb_review.jsonl`). See §7 and `[[slm-next-steps]]`.

## 6. Known limitations
- **FOLIO-train translation** is gateway-gated and paraphrase-limited (binary predicates, nested tails);
  ~25% of translations don't solve-to-gold and fall back to templates.
- **LogicNLI** forward-chaining (§4) recovers **96%** (1441/1500) as real derivations; the ~4% that
  don't match gold (usually disjunctive cases the unit-propagation reasoner under-derives) fall back to
  templates. The reasoner is a bounded forward closure, not a full theorem prover.
- **Current corpus-level eval numbers live in §7** (bf16 Qwen3.5-4B on the de-leaked held-out set,
  all variants on the identical eval). The *earlier 4-model suite is stale* — it predates the
  `group_id` leakage fix, the LogiQA↔ReClor dedup, and the solver-path updates — so don't cite it.
  For frontier-model reference measurements (thinking, multi-run, balanced), see
  `critical-reasoning-eval/RESULTS-DOK.md`.

## 7. Version history — 2026-07-11 (leakage fix, shortening, gap-closing variants)

Each variant is a **separate file** built on a common base; the trainer reads only
`{version}.jsonl` + `eval_items.json`. Evals are **bf16 Qwen3.5-4B**, greedy, `enable_thinking=False`,
n=60/family, on the identical de-leaked eval (6-family tuned-accuracy mean). Frontier reference on the
same eval: **Opus 4.8 = 88.5**, MVR 10. **Best so far: `sft_train_short.jsonl` (85.3).**

**Step B for the MCQ satellite (done).** lsat_lr/arct/logiqa previously had only shallow move-naming
templates. `stepb_verbalize.py` gives each a real, gold-verified trace via a
Sonnet→Opus→answer-conditioned-backfill cascade with a blind re-read gate; the committing option
*letter is isolated to the final sentence* (preserves the last-hit grade invariant). ~3,175 verified
traces in `mcq_stepb_review.jsonl`, applied as an overlay. Replaces the old "deferred" plan.

**base / v1-long (`sft_train.jsonl`) — leakage fix + rebuild.**
- *Change:* LogiQA-2.0-test and ReClor share **478 byte-identical items** (47 crossed the train/eval
  split = real leakage, inflating lsat_lr/logiqa). `parse_logiqa` now drops any LogiQA passage present
  in ReClor (ReClor owns shared passages); **478 clean replacements** pulled from LogiQA-2.0 *train*
  (verified not-in-corpus, not-in-ReClor, Step-B traced) keep the family volume; `parse_proverqa` given
  a stimulus-hashed `group_id`. Full rebuild on the group_id split.
- *Result:* 0 train↔eval leakage; eval **81.4**. The long MCQ frontier traces get high lsat_lr (90) but
  their verbosity makes MCQ **62%** of supervised tokens, crowding out the entailment core.

**shortened (`sft_train_short.jsonl`) — MCQ trace compression.  ← current best.**
- *Change:* `shorten_mcq_traces.py` compresses each verified MCQ frontier trace to ~4 sentences (Haiku),
  re-verified to commit to gold with the letter isolated. MCQ token-mass **62% → 44%** (entailment back
  to the majority); entailment traces untouched.
- *Result:* eval **85.3** (best) — recovers the entailment families the long traces had crowded.

**v2 (`sft_train_v2.jsonl`) — 4-phase gap-closer on the shortened base.**
- *Change:* P1 Opus-distilled distractor-elimination traces for lsat_lr+logiqa
  (`distill_mcq_elimination.py`); P2 folio-neutral up-sample (2×); P3 Haiku-deepened logicnli
  True/False (`deepen_logicnli_tf.py`); P4 irrelevant-premise robustness aug (borrow a sentence from a
  different entailment problem; `assemble_v2.py`).
- *Result:* eval **82.8**. Verdict: **P2 helps** (folio-neutral 66.7→86.7, +20); **P1 & P3 hurt**
  (lsat_lr 86.7→81.7, logicnli-neutral 96.7→86.7). Net regression vs shortened.

**v3 (`sft_train_v3.jsonl`) — P1(lsat_lr-elim, trimmed ~700) + P4.**
- *Change:* kept only lsat_lr elimination (trimmed via `trim_elim_traces.py`) + P4; dropped logiqa-elim,
  P2, P3. (Designed from an analysis mistakenly run on v1-long, not v2 — hence it kept a loser.)
- *Result:* eval **81.7**; MVR 5 (n=20). The retained elim-distill underperforms shortened on lsat_lr.

**v4 (`sft_train_v4.jsonl`) — P2(folio up-sample) + P4.  ← corrected best-candidate (untested).**
- *Change:* the corrected winners only — folio-neutral up-sample + irrelevant-premise aug on the
  shortened base (`assemble_v4.py`); MCQ/logicnli traces left exactly as shortened. MCQ token-mass 38.7%.
- *Goal:* carry shortened's 85.3 accuracy + v2's folio-neutral (+20) + P4 robustness together.

**Lessons.** (a) Verbose MCQ reasoning trades a little MCQ accuracy for entailment interference and a
higher LGMT flip-rate — shortening is the right default. (b) The lsat_lr→Opus gap is a **trace-length**
lever (long traces → lsat_lr 90), *not* the Opus-elimination distill (which underperformed). (c) MVR at
n=20 (±1 item = 5%) is too noisy to rank on — needs n≥40. Running decision log: `[[slm-next-steps]]`.

### Results — all variants + Opus (bf16 Qwen3.5-4B, tuned acc, same de-leaked eval, n=60/family)

Neutral recall (entailment-only) in parens; MVR/HDR from the n=20 LGMT probe (too noisy to rank on).

| Model | arct | folio (neu) | logicnli (neu) | logiqa | lsat_lr | proverqa (neu) | **mean** | MVR/HDR |
|---|---|---|---|---|---|---|---|---|
| base (untuned) | 75.0 | 61.7 (50.0) | 48.3 (26.7) | 65.0 | 81.7 | 50.0 (63.3) | 63.6 | 20/20 |
| **shortened** ← best tested | 93.3 | 73.3 (66.7) | 88.3 (96.7) | 78.3 | 86.7 | 91.7 (100) | **85.3** | 20/20 |
| v2 (4 phases) | 90.0 | 71.7 (**86.7**) | 83.3 (86.7) | 80.0 | 81.7 | 90.0 (100) | 82.8 | — (not captured) |
| v3 (lsat_lr-elim + P4) | 88.3 | 68.3 (66.7) | 85.0 (96.7) | 68.3 | 85.0 | 95.0 (93.3) | 81.7 | 5/5 |
| v1-long (non-shortened) | 91.7 | 65.0 (66.7) | 85.0 (96.7) | 68.3 | **90.0** | 88.3 (96.7) | 81.4 | 5/5 |
| v4 (P2 + P4) | *untested* | | | | | | *TBD* | *TBD* |
| Sonnet 4.6 (frontier ref) | 88.3 | 75.0 (76.7) | 63.3 (30.0) | 81.7 | 96.6 | 69.5 (73.3) | 79.1 | 15/10 |
| **Opus 4.8** (frontier ref) | 95.0 | 76.7 (76.7) | 95.0 (93.3) | 83.3 | 100 | 81.4 (93.3) | **88.5** | 10/10 |

Reads: **shortened (85.3) leads the tested corpora and beats Sonnet 4.6 (79.1) outright** — all four SLM
variants beat Sonnet. So the distilled 4B sits **between Sonnet and Opus**, within 3.2 of Opus (88.5).
The SLM **beats both frontier models on proverqa** (91.7 vs Opus 81.4 / Sonnet 69.5) and leads neutral
recall (Sonnet collapses on logicnli-neutral, 30.0). The residual gap to Opus is lsat_lr / logicnli /
logiqa (4B reasoning capacity). v4 is the untested candidate meant to add v2's folio-neutral + P4
robustness on top of shortened's accuracy.
