# Colab / local SFT — critical-reasoning SLM (Qwen3.5)

A **self-contained** package to fine-tune a small Qwen (default **Qwen3.5-0.8B**;
2B / 4B-QLoRA also supported) to do **one thing reliably**: given a self-contained
logic problem, reason briefly from the premises and **commit to exactly one credited
answer** — an option letter (MCQ) or an entailment label `True / False /
Uncertain|Unknown`, *including* the neutral "does-not-follow" class. No dependency on
the parent repo; the SFT data (traces already generated) and the prompt/grading
contract are bundled here.
Note: see data-synthesis.md for data making methods

## Contents

| File | Purpose |
|------|---------|
| `critical_reasoning_sft.ipynb` | Colab notebook — install → load data → QLoRA train → base-vs-tuned eval |
| `run_sft.py` | headless trainer+eval (GPU/CPU); optimizations below + `--resume`/`--skip-updates`, `--eval-only`, batched eval (`--eval-batch`) |
| `colab_standalone.py` | **single-file** SFT (slm_core+run_sft bundled) to paste into one Colab cell; regenerate via `build_standalone.py` |
| `opus_eval.py` | score a frontier API/gateway model (default Opus 4.8) on the **identical** held-out slice — the README reference row |
| `supervisor.sh` | self-healing overnight driver: 4B to a full epoch (resume-on-crash), then a best-effort bf16 2B |
| `slm_core.py` | prompt contract, deterministic grading, natural-answer parser, JSONL loaders, **Claude judge/eval** (API key or gateway) |
| `data_construction/` | **corpus-build / FOL cluster** (see below); writes `data/`, reads `data/_raw/`, imports `slm_core` from the parent |
| `data_construction/build_corpus.py` | rebuilds the corpus from `data/_raw/` — the **single house-style verbalizer** + LGMT + budget |
| `data_construction/stepb_verbalize.py` | optional **frontier-verbalizer** upgrade (Opus/Haiku routing + FOL solver), resumable |
| `data_construction/fol_solver.py` | z3 classical entailment solver (unsat-core + Uncertain countermodel witness) — used for **both** FOLIO and (post-pruning) LogicNLI |
| `data_construction/logicnli_logic.py` | LogicNLI **structured** `_logic.json` → FOL (exact, from the original release) |
| `data_construction/logicnli_prune.py` | **LogicNLI per-conclusion pruning** to a classically-consistent, z3-verified, gold-preserving premise subset (raw blocks are UNSAT by construction); also exports `data/lgmt_sources.json` |
| `data_construction/fol_forward.py` | bounded forward-chaining reasoner — now only **seeds** the prune (relevant premises) |
| `data_construction/fol.py` · `logicnli_fol.py` · `folio_concl_fol.py` | FOL parser · LogicNLI NL→FOL (legacy fallback) · FOLIO-train conclusion NL→FOL (cached) |
| `smoke.py` | end-to-end loop check on the real data (CPU, tiny model, few steps) |
| `.env` / `.env.example` | paste `ANTHROPIC_API_KEY=` here for the judge fallback + Step B (gitignored) |
| `data/sft_train.jsonl` | **pre-built** training rows `{prompt, completion, gold, …}` (~7.5k) |
| `data/eval_items.json` | held-out eval items (hashed 20% of **every** family, full fields) |
| `data/corpus_meta.json` | exact counts + provenance |
| `data/_raw/` | raw source datasets (for regenerating the corpus) |
| `requirements.txt` | pip deps (Colab already has CUDA torch) |

---

## Methodology (current)

### The behavior & the two regimes (never merged)
The thesis: control the *training data* to instil one reliable behavior — format-committed
logical reasoning, including recognizing under-determination (the neutral class), where
even frontier models are weak. Data is split into two regimes, reported/trained separately:

- **Entailment core** — FOLIO, LogicNLI, ProverQA → **FRQ label** (`True/False/neutral`),
  deterministically gradable; carries the neutral "does-not-follow" class (the hard part).
- **Argument satellite** — ReClor (flaw / necessary-or-sufficient assumption / weaken),
  adversarial ARCT, LogiQA → **MCQ** option letter.

### Natural completions (not a rigid tag)
Completions read as natural prose that reasons first, then commits in a final sentence —
e.g. *"…nothing settles it either way. Therefore, based only on the premises, the
statement is Uncertain."* / *"…So the correct option is (C)."* No `Reasoning:/Answer:`
scaffold. The parser (`slm_core.parse_label` / `parse_letter`) extracts the committed
answer from natural text (last label mention; `(X)`/"answer: X"/"option … X"). If a
response can't be parsed **and** a key is present, a **non-deterministic fallback** asks a
cheap model (Haiku, via `.env`) for the one-word answer (`--judge-model`, opt-in).

### Corpus (v2, ~7.5k train + ~1.5k eval)
Built by `build_corpus.py` from the full source datasets, capped per source so *format
types* stay balanced (small pools used at a high %, huge pools at a low %). Every family
contributes to both train and a hashed-20% held-out eval slice. Rebalanced (see PLAN.md
"Plan 1"): ProverQA trimmed + med/hard-weighted, FOLIO adds its validation split, LGMT
consistency follow-ups reduced to ~1,200 split **evenly** across the 3 entailment families
with **varied MRs** (P1 reorder / P2 duplicate / P3 irrelevant, P3 capped) — fixing v1's
FOLIO-heavy, P3-dominated skew. LGMT follow-ups are built from **train** sources only (no
leakage). Format mix ≈ 55% FRQ / 45% MCQ.

### Trace fidelity — two tiers
1. **Deterministic (shipped default, `data/sft_train.jsonl`).** One verbalizer:
   ProverQA restyles its real Prover9 chain; **FOLIO** gets z3 solver paths (entailing premises /
   countermodel witness) and **LogicNLI** gets forward-chained derivations (§ solver paths), with a
   grounded template only where the solver declines; MCQ cites the credited option. Every completion is
   gold-correct (0/7473 fail the grader). Cheap, instant, teaches the **primary** target
   (format + commitment + neutral).
2. **Step B frontier verbalizer (optional upgrade, `stepb_verbalize.py`).** Replaces the
   templates with *real, faithfully-verified* reasoning, routed by cost:
   - **ProverQA** → Haiku **rewords** the shipped chain.
   - **FOLIO** → `fol_solver.py` (z3) proves the entailment; if the solver agrees with
     gold, Haiku **rewords the verified verdict** (with a countermodel witness for neutral).
   - **LogicNLI** → `fol_forward.py` forward-chains the structured logic; Haiku **rewords the
     verified chain** (96% of items have one).
   - **Everything else (MCQ, no formal FOL)** → **Opus generates**: blind-solve; on a miss,
     answer-conditioned backfill gated by a **blind re-read** (a 2nd call sees only the
     reasoning and must re-commit to gold — catches rationalization).
   Every trace is gold-verified before use; failures keep the deterministic completion.
   Resumable cache; concurrent. FOLIO solver: solves 96% of FOLIO-val, agrees with gold on
   96% of those. Needs `ANTHROPIC_API_KEY` in `.env`.

### Training (`run_sft.py` / notebook)
Completion-only loss masking (loss on the trace+answer, prompt masked). Key mechanics,
tuned for an 8 GB laptop GPU:
- **Token-budget, length-sorted batching** — long ProverQA rows land in small batches →
  bounded peak VRAM (the fix for the batch-size OOM).
- **Token-weighted gradient accumulation** — each micro-batch contributes ∝ its supervised
  tokens; trailing window flushed (no dropped batches).
- **Gradient checkpointing with `preserve_rng_state=False`** (+ `lora_dropout=0`) — the RNG
  fork throws `cudaErrorUnknown` on this GPU; disabling it is safe and fixes it.
- **Cosine LR + warmup**, sized to real optimizer updates.
- **`flash-linear-attention`** — ~10× speedup for the Qwen3.5 (linear-attention) family.
- **`--save-every N`** — periodic adapter checkpoint (overwrites `runs/<out>/checkpoint`,
  ~0.3–1 s each) so a killed run leaves a usable adapter.
- **`--load-4bit`** — nf4 QLoRA so bigger bases (Qwen3.5-4B ≈ 2 GB in 4-bit) fit 8 GB
  (4-bit kernel verified working on sm_120).
- **`--train-file`** — point at `data/sft_train_stepb.jsonl` to train on Step B traces.

### Eval (`run_sft.py` / notebook cell 8)
Base-vs-tuned on the held-out slice, deterministic grading, **never merging regimes**:
- **Stratified sampling** — even stride across each family's ordered slice (not first-N,
  which was all-easy for ProverQA and inflated it).
- KV cache re-enabled + gradient checkpointing disabled for fast generation.
- Metrics: **parse rate** (format gate), **accuracy per family**, **neutral-class recall**
  (headline). Optional `--judge-model` fallback for unparseable outputs.
- **`--lgmt-eval N`** — LGMT consistency: append an irrelevant premise (P3 over-inference
  probe) and check the label survives. Reports **MVR** (flip rate) and **HDR** (flips among
  source-correct answers — the defects static accuracy hides), for base and tuned.

---

## How to run

**Colab (notebook):** upload the `colab/` folder, set Runtime → GPU (T4), run cells top to
bottom. `MODEL_ID` defaults to `Qwen/Qwen3.5-0.8B` (fallback `Qwen/Qwen3-0.6B`); the install
cell adds `flash-linear-attention` for the Qwen3.5 fast path.

**Colab (single file):** paste [`colab_standalone.py`](colab_standalone.py) into one cell —
it bundles `slm_core` + `run_sft` (config vars at the top), pip-installs, and prompts to
upload `sft_train.jsonl` + `eval_items.json`. Rebuild it after editing either module with
`python build_standalone.py`.

**Local / headless:**
```bash
# 2B in bf16 (tensor cores, no dequant → much faster than 4-bit): fits 8 GB, checkpoint + LGMT
python run_sft.py --model Qwen/Qwen3.5-2B --epochs 1 --save-every 100 --lgmt-eval 20 --eval-batch 8
# biggest that fits 8 GB: 4B via 4-bit QLoRA — ONLY batch-2/token-1024 is stable on sm_120
# (~20 s/update → ~11 h/epoch; checkpointed + resumable). Do NOT set expandable_segments.
python run_sft.py --model Qwen/Qwen3.5-4B --load-4bit --epochs 1 --batch-size 2 --max-tokens 1024 --grad-accum 2 --save-every 100 --lgmt-eval 20
# self-healing overnight (4B to a full epoch, then a best-effort bf16 2B):
bash supervisor.sh &                      # relaunches-with-resume on a crash
# resume a crashed/stopped run from its last checkpoint:
python run_sft.py --model Qwen/Qwen3.5-4B --load-4bit --resume runs/gpu_4b/checkpoint --skip-updates 400 ...
# eval a checkpoint WITHOUT training (base-vs-tuned + LGMT), batched:
python run_sft.py --model Qwen/Qwen3.5-4B --load-4bit --eval-only --resume runs/gpu_4b/checkpoint --out runs/eval_ckpt --eval-batch 8
# score a frontier API/gateway model on the identical slice (needs .env creds):
python opus_eval.py --model claude-opus-4-8 --eval-per-family 10 --lgmt-eval 8
# train on Step B (frontier) traces instead of the deterministic corpus:
python run_sft.py --train-file data/sft_train_stepb.jsonl ...
```

---

## Reference — the same eval on a frontier model and the untuned base

The SFT results below read best against two fixed goalposts, both run through the
**identical harness on the identical held-out slice** (60 items, 10/family,
deterministic grading; LGMT probe n=8): the **untuned base** (floor) and **Opus 4.8**
(frontier ceiling). Opus 4.8 answered via the Anthropic API (gateway), **no extended
thinking**, same prompts, scored by the same deterministic `grade()` (see
[`opus_eval.py`](opus_eval.py); `runs/eval_opus48/metrics.json`).

| Model — identical 60-item slice | Parse | Accuracy | Neutral recall | LGMT MVR ↓ | LGMT HDR ↓ |
|---|--:|--:|--:|--:|--:|
| **Opus 4.8** (frontier, no ext. thinking) | 100 | **95.0** | 100 | 37.5\* | 25.0\* |
| Qwen3.5-4B **base** (untuned) | 98.3 | 58.3 | 37.5 | 28.6\* | 14.3\* |

Per-family accuracy (Opus / base): arct 100/40 · folio 80/50 · logicnli 100/50 ·
logiqa 100/70 · lsat_lr 90/80 · proverqa 100/60. \*LGMT n=7–8 (tiny, noisy); the
frontier LGMT figure on the full 20-MR suite *with* thinking is ≈8% MVR (eval repo
[`RESULTS-DOK.md`](../../critical-reasoning-eval/RESULTS-DOK.md)). These goalposts are on
the **v2** slice; **v1 (0.6B) used a different, smaller slice** and is not directly comparable.

---

## Results at a glance — the two SFT runs

Base-vs-tuned, deterministic grading, two regimes never merged.
**v1** = Qwen3-0.6B (LoRA, 250 upd, 25 items/family, pre-stratified sampling).
**v2** = Qwen3.5-4B (4-bit QLoRA, **partial: 400/1943 updates ≈ 21% of epoch 1**, 10
items/family, stratified). Different bases *and* slices → read the **direction and size
of the lift**, not v1-vs-v2 absolutes.

| Run | Parse (b→t) | Accuracy (b→t) | Neutral recall (b→t) | LGMT MVR (b→t) ↓ |
|---|--:|--:|--:|--:|
| v1 · Qwen3-0.6B (250 upd) | 72.7 → **99.3** | 32.7 → **45.3** | 4.2 → **16.7** | — |
| v2 · Qwen3.5-4B (upd 400, ~21%) | 98.3 → **100** | 58.3 → **81.7** | 37.5 → **50.0** | 28.6 → **0.0** |

Per-family accuracy (base → tuned; Opus 4.8 shown as the ceiling):

| Family | v1 · 0.6B | v2 · 4B (~21%) | Opus 4.8 |
|---|--:|--:|--:|
| arct | 20 → 48 | 40 → **80** | 100 |
| folio | 32 → 20 | 50 → **70** | 80 |
| logicnli | 36 → 52 | 50 → **60** | 100 |
| logiqa | 32 → 28 | 70 → **100** | 100 |
| lsat_lr | 20 → 32 | 80 → **90** | 90 |
| proverqa | 56 → 92 | 60 → **90** | 100 |

**Findings.**
- **v2 closes ≈64% of the base→Opus accuracy gap at 21% of one epoch** (58.3 → 81.7 vs
  Opus 95.0). Every family improves; biggest lifts on the MCQ satellite (logiqa 70→100,
  arct 40→80) and proverqa 60→90.
- **On LGMT consistency the tuned SLM (MVR 0) beats non-thinking Opus 4.8 (37.5) and its
  own base (28.6)** on the P3 irrelevant-premise probe — the over-inference defect the
  project targets. *n=8, noisy*, and thinking-Opus is far more robust (≈8% on the full
  suite); the *direction* is the thesis: targeted data instils answer-stability that raw
  scale alone doesn't guarantee.
- **FOLIO neutral stays 0→0** (v2) — the one family the deterministic templates don't
  fix, and the standing motivation for **Step B** (solver/frontier-verbalized traces).
  Opus gets FOLIO-neutral 100% here (small n).
- v1's FOLIO **regressed** (32→20) on templated traces; v2 at least *holds* FOLIO accuracy
  (50→70) even if not its neutral class.

---

## Results — v1 (first real GPU run)

**Qwen3-0.6B**, LoRA r16, **250 updates** (loss 3.07 → 0.06), ~60 min on an RTX 5050 (8 GB).
Base-vs-tuned, 25 items/family. Adapter: `runs/gpu_v1/adapter/`.

| Metric | Base | Tuned | Δ |
|--------|-----:|------:|--:|
| Parse rate (format gate) | 72.7% | **99.3%** | +26.6 |
| Accuracy | 32.7% | **45.3%** | +12.6 |
| Neutral-class recall | 4.2% | **16.7%** | +12.5 (4×) |

Per family (base → tuned acc): proverqa 56→**92** (neutral 0→67), logicnli 36→52, arct
20→48, lsat_lr 20→32, logiqa 32→28, folio 32→20.

**Caveats:** reduced run (0.6B, ~⅓ epoch, deterministic templates, 25 items/family); the
92% ProverQA was measured *before* the stratified-sampling fix (so it's easy-only); FOLIO
regressed (templated traces) — the motivation for Step B. Treat per-family numbers as
indicative, not final.

---

## Results — v2 (Qwen3.5-4B QLoRA, partial checkpoint)

**Qwen3.5-4B**, 4-bit QLoRA (nf4), LoRA r16, **400 / 1943 updates (~21% of epoch 1)** on
an RTX 5050 (8 GB). Base-vs-tuned on the held-out slice, 10 items/family (n=60),
deterministic grading, **batched eval** (`--eval-batch 8`). The run was stopped
mid-epoch and is **resumable** (see Run 3); checkpoint at `runs/gpu_4b/checkpoint/`,
eval at `runs/eval_upd400/metrics.json`.

| Metric | Base | Tuned | Δ |
|--------|-----:|------:|--:|
| Parse rate (format gate) | 98.3% | **100%** | +1.7 |
| Accuracy | 58.3% | **81.7%** | +23.4 |
| Neutral-class recall | 37.5% | **50.0%** | +12.5 |
| LGMT MVR (label-flip rate) ↓ | 28.6% | **0.0%** | fully consistent (n=8) |
| LGMT HDR (hidden defects) ↓ | 14.3% | **0.0%** | — |

Per family (base → tuned acc): logiqa 70→**100**, proverqa 60→**90**, lsat_lr 80→**90**,
arct 40→**80**, folio 50→**70**, logicnli 50→**60**. Neutral: logicnli 0→**66.7**,
proverqa 100→66.7, folio 0→0.

**Caveats:** *partial* checkpoint (~21% of one epoch — numbers will still move); n=60
(+ LGMT n=8) is small; deterministic-template corpus (Step B not applied); the Opus 4.8
reference is non-thinking. A strong early signal, not a final result. (See the goalposts
and the v1-vs-v2 tables above.)

---

## Run history — what differed, what failed, what changed

**Environment:** RTX 5050 Laptop GPU, **8 GB VRAM, Blackwell / sm_120, WSL2**. The venv
first had **CPU-only torch**; a CUDA (cu128) build was installed so the GPU could be used at
all. Most of the hard-won fixes below came from getting a GPU + WSL2 to train
reliably.

### Run 1 — Qwen3-0.6B (the v1 result above)
Chosen as the first target: cached, dense-attention (no linear-attention surprises), small.
Failures found and the fixes each produced:

| Failure | Cause | Fix |
|---|---|---|
| Backward crashed intermittently (`cudaErrorUnknown` / "device not ready") | gradient checkpointing's `fork_rng`/`set_rng_state` on sm_120 | `preserve_rng_state=False` (+ `lora_dropout=0` so recompute stays deterministic) |
| Only survived with `CUDA_LAUNCH_BLOCKING=1` (~30 s/update) | serialized kernels masked the above | after the fork_rng fix, **non-blocking** runs (~2–5 s/update) |
| OOM at update 250 | a length-sorted batch of long ProverQA rows at batch-4 + allocator fragmentation | **token-budget batching** (long rows → small batches) |
| LR barely decayed | cosine scheduler mis-sized by `grad_accum²` | size the scheduler to **real optimizer updates** |
| Eval very slow | `use_cache=False` + gradient checkpointing left on during `generate()` | re-enable KV cache + disable GC for eval |
| ProverQA accuracy inflated | eval took the **first-N** items/family → all `easy` | **stratified** even-stride sampling |

Outcome: a clean 250-update run → the v1 numbers above.

### Run 2 — Qwen3.5-0.8B (attempted; not completed)
The intended target (Qwen3.5 small series). It surfaced a **new** class of problems because
Qwen3.5 uses **linear-attention** layers:

| Failure | Cause | Fix |
|---|---|---|
| **OOM on the very first backward** | Qwen3.5's fast path (`flash-linear-attention` + `causal-conv1d`) wasn't installed → memory-heavy torch fallback blew 8 GB even on a short batch | immediate: lower token budget/seq (token 1024, batch 4, max-len 768). proper: **install `flash-linear-attention`** → ~10× faster (~28 → ~2 s/update) *and* lower memory |
| Relaunch gave empty output / exit 1 | the launch command's `pkill -f run_sft.py` matched the **launcher shell's own** command line and killed itself before training | don't self-match in the kill |
| Trained fine to update 125/400, then **lost everything** | VSCode was closed; the adapter only saved at the **end** | **`--save-every N`** periodic checkpointing (overwrite `runs/<out>/checkpoint`) |

### Run 3 — Qwen3.5-4B QLoRA (4-bit; partial ~21% of epoch, resumable)
The biggest base that fits 8 GB, via nf4 QLoRA — the source of the **v2** numbers above.
New problems came from running at the **memory edge (~7.9 / 8.1 GB)** and the sm_120 driver:

| Failure | Cause | Fix |
|---|---|---|
| `CUDA error: device not ready` on the first forward | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on sm_120 | **don't use expandable_segments** on this GPU |
| `CUBLAS_STATUS_INTERNAL_ERROR` mid-run | fuller batches (batch 8) starve cuBLAS workspace at ~7.9 GB | for 4B, **only batch-2 / token-1024 is stable** (the proven pre-flight config) |
| Crashed at ~upd 200 with no way to continue mid-epoch | sm_120 driver flake; `run_sft.py` had no resume | **`--resume` + `--skip-updates`** (reload the checkpoint adapter, fast-forward the LR schedule, skip done micro-batch windows) + a self-healing **`supervisor.sh`** that relaunches-with-resume on crash |
| A resumed run ran **3.6× slower** (~72 vs ~20 s/upd) | the GPU context was degraded by the crash it resumed *into* | restart clean once the GPU is idle (not thermal — 54 °C, max clocks) |

Real rate ≈ **20 s/update → a full 4B epoch ≈ ~11 h** — 4-bit dequant is
memory-bandwidth-bound (100% util but only ~25 W; tensor cores idle). Stopped at
upd 400/1943 with the adapter preserved; evaluated with `--eval-only` on that checkpoint.

### What changed as a result (carried forward)
`flash-linear-attention` (Qwen3.5 ~10× speedup) · `--save-every` checkpointing ·
token-budget batching · fork-RNG-safe gradient checkpointing · fixed LR schedule ·
token-weighted grad-accum · stratified eval · **4-bit QLoRA** (`--load-4bit`, verified on
sm_120 → Qwen3.5-4B fits 8 GB, batch-2/token-1024 only) · **`--resume`/`--skip-updates`
+ self-healing `supervisor.sh`** · **`--eval-only`** (eval a checkpoint without training)
· **batched eval generation** (`--eval-batch`, left-padded → ~3× faster; batch-1 decoding
was memory-bound at ~30% util) · LGMT consistency metrics · Claude-API judge/eval via
**API key _or_ gateway** (`ANTHROPIC_BASE_URL` + `ANTHROPIC_AUTH_TOKEN`) + Step B (`.env`)
· FOLIO symbolic solver · single-file **`colab_standalone.py`**.

---

## Rebuild / rebalance the corpus
```bash
cd colab && python data_construction/build_corpus.py   # rewrites data/sft_train.jsonl + eval_items.json
```
Needs `z3-solver` for the FOLIO working-paths (`pip install z3-solver`; absent → template fallback).
Edit `CAPS` at the top of `data_construction/build_corpus.py` to rebalance. Deterministic (hashed
sampling + split): same caps + seed reproduce the same corpus.

## Step B (frontier traces, optional upgrade)
```bash
# paste ANTHROPIC_API_KEY into .env first
python data_construction/stepb_verbalize.py --concurrency 12   # Opus generate / Haiku reword / FOLIO solver
python run_sft.py --train-file data/sft_train_stepb.jsonl ...
```
Cost/latency: ~$150–250 (Opus) and ~1.5–2.5 h at concurrency 12 for the full corpus; use
`--limit` for a sample first. ProverQA (chain) + FOLIO-val (solver) run on cheap Haiku.

## Smoke test (no GPU)
```bash
python smoke.py     # Qwen3-0.6B on CPU: load → train 3 steps → eval
```
Green = the wiring connects on the *real* data. Does not prove learning.

## Regenerating from scratch
`data/_raw/` holds the source datasets (ReClor zip, LogiQA/LogicNLI/ProverQA, FOLIO train+val,
adversarial ARCT). `data_construction/build_corpus.py` reads them; re-download any missing one per
[`../../critical-reasoning-eval/SOURCES.md`](../../critical-reasoning-eval/SOURCES.md).

LogicNLI solver paths additionally need the **original structured release** at
`data/_raw/logicnli_sim/LogicNLI_sim/` (`dev_logic.json` + `dev_language.json`): download
`https://raw.githubusercontent.com/omnilabNLP/LogicNLI/main/dataset/LogicNLI_sim.zip` and unzip it
there. It joins to the logi_glue items by index (row `i` ⇒ block `i//20`, statement `i%20`). If the
zip is absent, LogicNLI degrades gracefully to templates. The z3-based FOLIO solver needs
`pip install z3-solver`.
