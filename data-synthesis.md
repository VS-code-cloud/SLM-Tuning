# Data synthesis — critical-reasoning SLM corpus

How the SFT corpus (`data/sft_train_v7.jsonl`) and held-out eval (`data/eval_items_clean.json`)
are built: where each dataset comes from, why it's FRQ vs MCQ, and every transform from raw
scrape to SFT row — including the symbolic **solver-derived working paths**. Rebuild the base
with `python data_construction/build_corpus.py` (deterministic given seed+caps); code lives in
`data_construction/`, reads `data/_raw/`, writes `data/`. Needs `z3-solver` for solver paths.

## 1. One behavior, two regimes (never merged)

Goal: one reliable behavior — reason briefly from the premises and **commit to exactly one
credited answer**, *including* recognizing under-determination (the neutral "does-not-follow"
class, where frontier models are weak — Opus 4.8 on LogicNLI-Unknown = **31.7%**, RESULTS-DOK).

- **Entailment core** — FOLIO, LogicNLI, ProverQA → **FRQ label** `True/False/Uncertain|Unknown`.
  Deterministically gradable; carries the hard neutral class.
- **Argument satellite** — ReClor (LSAT LR), adversarial ARCT, LogiQA → **MCQ** option letter.

Reported/trained separately: recognition (MCQ) and generation/entailment (FRQ) are different
skills, and the weakness lives in sub-dimensions (neutral recognition, multi-step depth,
consistency under reformulation), not headline accuracy.

## 2. Sources and what each trains

| Family | Dataset | Regime | Trains / why |
|---|---|---|---|
| `folio` | FOLIO (Han 2022) train+val | FRQ 3-way | FOL NLI; ships FOL → solvable; neutral (Uncertain) is the target weak class |
| `logicnli` | LogicNLI (Tian 2021) dev via logi_glue + structured release | FRQ 3-way | largest entailment family; flagship neutral weakness; solver-verbalized (§4); `self_contradiction` dropped |
| `proverqa` | ProverQA (Qi 2025) dev med+hard | FRQ 3-way | multi-step FOL depth gradient; ships a Prover9 chain (the trace); easy dropped (96.7% ceiling) |
| `lsat_lr` | ReClor (Yu 2020) train+val | MCQ | LSAT argument analysis; kept only flaw/assumption/weaken (gap-analysis subset) |
| `arct` | **Adversarial** ARCT (Niven & Kao 2019) neg splits | MCQ 2-way | implicit-warrant on the de-biased set (negation-balanced) → genuine warrant reasoning |
| `logiqa` | LogiQA 2.0 test+train/dev | MCQ | deductive "what follows"; reading-comprehension items filtered out (§7) |
| `LGMT` | follow-ups from the 3 entailment families (train only) | FRQ | metamorphic consistency across the eval's full MR taxonomy (MR-P/C/S/E) → answer-stability under logic-preserving edits |

Per-family sourcing matches the sibling `critical-reasoning-eval` exactly, so the SLM trains on
the same *kind* of data the frontier eval measures.

### Sizes (current, v7)

- **Train `sft_train_v7.jsonl`: 10,067 rows** — folio 2,401 · logicnli 2,356 · proverqa 1,701 ·
  logiqa 1,366 · lsat_lr 1,218 · arct 1,025. **LGMT: 3,300, balanced** — MR-P 900 · MR-C 900 ·
  MR-S 900 · MR-E 600 (300 per family×category; MR-E from FOL, folio+logicnli only). Base
  (`sft_train_short.jsonl`): 7,414.
- **Test `eval_items_clean.json`: 1,516 held-out** — lsat_lr 282 · logiqa 278 · arct 275 ·
  logicnli 274 · folio 208 · proverqa 199. Scored: **60/family** (even-stride), neutral
  over-sampled to ~25%, plus a **20-item P3 LGMT** consistency probe (MVR/HDR).

## 3. Pipeline: raw → SFT rows

1. **Parse + filter** (`build_corpus.py`): ReClor task-type filter (flaw/assumption/weaken);
   label maps (→ True/False/Unknown, drop `self_contradiction`; ProverQA A/B/C → T/F/Uncertain);
   ProverQA easy dropped. Per-family caps (`CAPS`) equalize format types.
2. **LGMT follow-ups** from *train* entailment items only, ~3,300 across folio/logicnli/proverqa,
   using the eval's exact MR transforms (MR-P/C/S deterministic; MR-E from FOL — `lgmt_augment.py`).
3. **Hashed 80/20 split** on a **`group_id`** (shared-stimulus key), not row id — so ARCT
   per-annotator dupes, LogiQA/ReClor questions over one passage, and LogicNLI statements over
   one block never straddle the split. (id-only splitting leaked ~4.8% of eval stimuli; now **0**.)
4. **House-style verbalizer**: prose that reasons, then one committing final sentence
   (`…the statement is {label}.` / `…the correct option is (X).`). Reasoning avoids label/letter
   words so the parser's last-hit rule reads the committed answer. **0 grade-mismatches** (every build).
5. **Trace source priority**: `solver_path` (§4) → `prover_chain` (ProverQA) → `grounded_template`
   (fallback) → frontier/Step-B (§5, §7 for MCQ).

## 4. Solver-derived working paths (FOLIO / LogicNLI)

Templates teach *a* label, not *why it doesn't follow*, so the neutral class transfers poorly.
FOLIO and LogicNLI instead get **real derivations** baked in (no frontier model), with different
oracles because their gold uses different semantics:

- **FOLIO — z3 global entailment** (`fol_solver.classify_path`): emits the unsat-core premises for
  True/False, and for **Uncertain a countermodel witness** (an open atom + models both ways).
  FOLIO-val solves offline; FOLIO-train has no conclusion-FOL → cached Haiku NL→FOL translation.
- **LogicNLI — per-conclusion prune to a classically-consistent ⟨Γ, q⟩, then z3** (classical). FOL
  from the original structured release (no NL parsing), joined by verified index. Raw blocks are
  deliberately near-paradoxical (5 `self_contradiction`/block → jointly UNSAT), so `logicnli_prune.py`
  keeps the largest satisfiable premise subset that still yields the gold, then z3 emits the real
  derivation / countermodel.
- **Safety net:** keep a solver path only if it decides **and** its label == gold; else template.
  A wrong parse is dropped, never poisons data.

Coverage: FOLIO **797/1100**, LogicNLI **1430/1430** (of the pruned pool; ~5% unprunable dropped).

## 5. Trace-fidelity tiers

1. **Deterministic** — templates + ProverQA chain + solver paths. Offline, gold-correct by construction.
2. **Step-B (frontier)** — real, gold-verified reasoning for families with no solver (the MCQ
   satellite) via a Sonnet→Opus→answer-conditioned-backfill cascade with a blind re-read gate; the
   committing letter is isolated to the final sentence. Applied as build overlays (see §7).

## 6. Known limitations

- **FOLIO-train translation** is gateway-gated; ~25% don't solve-to-gold → template fallback.
- **LogicNLI** forward-chaining recovers ~96%; the ~4% (disjunctive under-derivation) fall back.
- **Current eval numbers live in §7.** The earlier 4-model suite is stale (predates the leakage
  fix, LogiQA↔ReClor dedup, solver paths). Frontier reference: `critical-reasoning-eval/RESULTS-DOK.md`.

## 7. Version history

Each variant is a separate file on a common base; the trainer reads `{version}.jsonl` + the eval.
Evals: **bf16 Qwen3.5-4B**, greedy, `enable_thinking=False`, n=60/family. **v1-long…v6 ran on the
old `eval_items.json`; v7 targets the cleaned `eval_items_clean.json`** (§ eval cleanup below), so
v7 isn't directly comparable to the earlier numbers.

**Step-B (done):** lsat_lr/arct/logiqa got real gold-verified traces (`stepb_verbalize.py`),
replacing shallow move-naming templates; ~3,175 in `mcq_stepb_review.jsonl`.

| Variant | Change | eval | notes |
|---|---|--:|---|
| base / v1-long | leakage fix + rebuild; 478 ReClor-shared LogiQA replaced | 81.4 | long MCQ traces → MCQ 62% of tokens, crowds entailment |
| **shortened** | MCQ traces compressed to ~4 sentences (MCQ 62→44%) | **85.3** | best on the old eval; robustness worst (MVR 20) |
| v2 | 4-phase gap-closer (P1 elim, P2 folio-up, P3 deepen, P4 irr-aug) | 82.8 | P2 helps (folio-neu +20); P1/P3 hurt |
| v3 | lsat_lr-elim (trimmed) + P4 | 81.7 | elim underperforms shortened on lsat_lr |
| v4 | P2 + P4 on shortened | 83.9 | logiqa fell to 66.7 — P2/P4 dilute logiqa's share |
| v5 | v4 + logiqa Opus-elim swap | 82.8 | elim did **not** recover logiqa (68.3) → decline is mix-share dilution, not trace content |
| v6 | +300 fresh logiqa, ensemble bad-gold drop (92), P4 600→300 | 83.3 | logiqa recovered 71.7 (share 13.2→15.9%); P4 cut → MVR 15/HDR 10 |
| **v7** | v6-recipe (P4=600) + full MCQ cleanup (below), on the clean eval | **85.3** | ties peak accuracy **and** best-ever robustness (**MVR 0/0**); best logiqa 80.0, lsat_lr 91.7, proverqa 93.3 |

### Results — old eval (bf16 Qwen3.5-4B, tuned acc, n=60/family; neutral-recall in parens; MVR/HDR from n=20 P3 probe)

| Model | arct | folio (neu) | logicnli (neu) | logiqa | lsat_lr | proverqa (neu) | **mean** | MVR/HDR |
|---|---|---|---|---|---|---|---|---|
| base (untuned) | 75.0 | 61.7 (50.0) | 48.3 (26.7) | 65.0 | 81.7 | 50.0 (63.3) | 63.6 | 20/20 |
| **shortened** | 93.3 | 73.3 (66.7) | 88.3 (96.7) | 78.3 | 86.7 | 91.7 (100) | **85.3** | 20/20 |
| v2 | 90.0 | 71.7 (**86.7**) | 83.3 (86.7) | 80.0 | 81.7 | 90.0 (100) | 82.8 | **0/0** |
| v3 | 88.3 | 68.3 (66.7) | 85.0 (96.7) | 68.3 | 85.0 | 95.0 (93.3) | 81.7 | 5/5 |
| v1-long | 91.7 | 65.0 (66.7) | 85.0 (96.7) | 68.3 | **90.0** | 88.3 (96.7) | 81.4 | 5/5 |
| v4 | 93.3 | 75.0 (**83.3**) | 85.0 (86.7) | **66.7** | **90.0** | 93.3 (96.7) | 83.9 | **5/0** |
| v5 | 91.7 | 75.0 (83.3) | 86.7 (96.7) | 68.3 | 85.0 | 90.0 (96.7) | 82.8 | **5/0** |
| v6 | 91.7 | 75.0 (86.7) | 86.7 (90.0) | **71.7** | 85.0 | 90.0 (96.7) | 83.3 | 15/10 |
| Sonnet 4.6 (ref) | 88.3 | 75.0 (76.7) | 63.3 (30.0) | 81.7 | 96.6 | 69.5 (73.3) | 79.1 | 15/10 |
| **Opus 4.8** (ref) | 95.0 | 76.7 (76.7) | 95.0 (93.3) | 83.3 | 100 | 81.4 (93.3) | **88.5** | 10/10 |

Reads: every tuned SLM beats Sonnet (79.1); the 4B sits between Sonnet and Opus (88.5), within 3.2.
It **beats both frontier models on proverqa** and on neutral recall (Sonnet collapses on
logicnli-neutral, 30.0). Residual gap to Opus is lsat_lr / logicnli / logiqa (4B capacity).

**Lessons.** (a) Shortening MCQ traces is the right default — verbose traces trade entailment
interference + higher flip-rate for a little MCQ accuracy. (b) The lsat_lr→Opus gap is a
trace-*length* lever, not the elimination distill. (c) The logiqa decline is **mix-share
dilution**, not trace quality; restoring share recovers it, but the P4 cut that helps costs
robustness — **share and robustness trade off through P4**.

### Eval cleanup → `eval_items_clean.json`

Same ensemble + careful-Opus process applied to the held-out eval: **17 of 278 logiqa fixed** (10
bad-gold swaps + 7 RC swaps), replacements verified `n_agree==3` and leakage-free.

### v7 MCQ-quality cleanup (2026-07-12)

On the v6-recipe base (P4 restored to 600), all MCQ-quality fixes, verified 0 grade-mismatch /
0 leakage / hollow-templates 1:
- **Reading-comprehension removed** — ~2.5–3% of LogiQA are verbal-comprehension (title/main-idea/theme,
  not logic) with subjective gold; dropped from train (46) + eval (7), replaced in kind.
- **Bad-gold filter** — LogiQA gold is noisy (machine-translated civil-service exam). Careful blind +
  fair-steelman Opus re-adjudicated the low-confidence rows; **105 bad-gold dropped**, 100 hollow
  templates upgraded + 38 wrongly-dropped restored → real traces. Backfilled bad-gold volume (+46 arct,
  +16 logiqa) so no family shrank.
- **Elim traces shortened** — the 917 long Opus distractor-elimination traces (171 tok) compressed to
  match the family (~99 tok), giving one consistent logiqa length and MCQ token-mass 40.8%.
- **Prevarication QC** — new traces judged for hedging; 4 flagged (1.4%) dropped/replaced.

**Results — clean eval (v7 vs frontier, apples-to-apples, n=60/family):** all three run on
`eval_items_clean.json`; frontier via `opus_eval.py --eval-file`.

| Model | arct | folio (neu) | logicnli (neu) | logiqa | lsat_lr | proverqa (neu) | **mean** | MVR/HDR |
|---|--|--|--|--|--|--|--|--|
| Base 4B (untuned) | 75.0 | 61.7 (50.0) | 48.3 (26.7) | 66.7 | 81.7 | 50.0 (63.3) | 63.9 | 20/20 |
| **v7 SLM** | 91.7 | 71.7 (80.0) | 83.3 (90.0) | 80.0 | 91.7 | 93.3 (96.7) | **85.3** | **0/0** |
| Opus 4.8 | 90.0 | 78.3 (80.0) | 93.3 (90.0) | 83.3 | 98.3 | 80.0 (90.0) | **87.2** | 10/10 |
| Sonnet 4.6 | 88.3 | 80.0 (76.7) | 60.0 (26.7) | 85.0 | 94.9 | 69.5 (73.3) | 79.6 | 25/10 |

**v7 is a strong second to Opus and clears Sonnet by 5.7 — and, on this n=20 P3 probe, is the most
robust of all three (MVR 0 vs Opus 10, Sonnet 25).** It's the only run with peak accuracy *and* MVR 0 (shortened 85.3
but MVR 20; v2 MVR 0 but 82.8; v6 MVR 15) — the payoff of splitting the levers (P4=600 for robustness,
clean corpus for accuracy). **v7 beats Opus on proverqa (+13.3), arct, and robustness**; the residual
gap to Opus is **logicnli (−10), lsat_lr (−6.6), folio (−6.6)** — the 4B multi-step-depth ceiling.
(Opus 87.2 vs its old-eval 88.5 is single-run variance on unchanged items.)


