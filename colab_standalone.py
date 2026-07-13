# ==========================================================================
#  Critical-reasoning SLM — SINGLE-FILE Colab SFT (Qwen3.5)
#  Paste this whole file into ONE Colab cell (Runtime -> GPU) and run it.
#
#  DATA: it needs the pre-built corpus (from build_corpus.py):
#     - sft_train.jsonl   (~13 MB)
#     - eval_items.json   (~2 MB)
#  Put both in DATA_DIR below. Easiest options:
#     (a) mount Drive:  from google.colab import drive; drive.mount('/content/drive')
#         then set DATA_DIR to the folder that holds the two files; OR
#     (b) leave DATA_DIR="." and you'll be prompted to upload the two files.
# ==========================================================================
# ------------------------------ CONFIG ------------------------------
# Default = Qwen3.5-2B in bf16: fast (tensor cores, no nf4 dequant), good, fits any
# Colab GPU. FLAGSHIP (best accuracy: ~82% on the held-out slice, +23 over base) is
# Qwen3.5-4B via 4-bit QLoRA — uncomment the flagship block. (0.8B was the OLD default.)
# MODEL_ID        = "Qwen/Qwen3.5-2B"
# LOAD_4BIT       = False     # bf16. Set True for nf4 QLoRA (needed to fit 4B).
# BATCH_SIZE      = 8         # max rows/batch (token budget also caps it)
# MAX_TOKENS      = 3072      # token budget per batch (rows x longest row)
# GRAD_ACCUM      = 4
# --- FLAGSHIP 4B (best results) — uncomment on a 16 GB+ Colab GPU (T4/L4/A100): ---
MODEL_ID = "Qwen/Qwen3.5-4B"; LOAD_4BIT = False
BATCH_SIZE = 4; MAX_TOKENS = 1536; GRAD_ACCUM = 2
# (8 GB sm_120 laptop only: BATCH_SIZE=2, MAX_TOKENS=1024, and do NOT set
#  PYTORCH_CUDA_ALLOC_CONF=expandable_segments. Colab T4/A100 have no such limit.)
EPOCHS          = 1
LR              = 2e-4
LORA_R          = 16
LORA_ALPHA      = 32
MAX_LEN         = 1024
MAX_STEPS       = 0         # 0 = full EPOCHS
SAVE_EVERY      = 100       # checkpoint adapter every N updates (0 = only at end)
EVAL_PER_FAMILY = 60
MAX_NEW_TOKENS  = 512
EVAL_BATCH      = 8         # batched eval generation (~3x faster; lower if OOM)
NEUTRAL_FRAC    = 0.25      # over-sample neutral/non-entailment to ~25% of the eval
LGMT_EVAL       = 20        # LGMT consistency probe items/condition (0 = off)
LGMT300_FILE    = "lgmt300.json"   # full-20-MR LGMT set (tuned); "" = off
FRQ_EVAL        = 0         # >0 => open-ended satellite probe, N items/family (needs a judge)
JUDGE_MODEL     = ""        # e.g. "claude-haiku-4-5" / "claude-opus-4-8" for parse-fallback + FRQ judge
# --- judge/FRQ grading credentials (blank = no judging) ---
ANTHROPIC_BASE_URL   = ""   # gateway URL (blank = api.anthropic.com)
ANTHROPIC_AUTH_TOKEN = ""   # Bearer token for a gateway  (OR use ANTHROPIC_API_KEY)
ANTHROPIC_API_KEY    = ""   # x-api-key (if not using a gateway)
DATA_DIR        = "/content/drive/MyDrive/SLM/data"       # folder holding sft_train.jsonl + eval_items.json
OUT_DIR         = "/content/drive/MyDrive/SLM/runs/colab"
TRAIN_FILE      = ""        # "" => DATA_DIR/sft_train.jsonl (e.g. sft_train_stepb.jsonl)
RESUME          = ""        # dir of a prior checkpoint adapter to resume from
SKIP_UPDATES    = 0         # with RESUME: updates already done
EVAL_ONLY       = False     # True => skip training, just eval RESUME's adapter
CPU             = False
# --------------------------------------------------------------------
import os as _os
for _k, _v in (("ANTHROPIC_BASE_URL", ANTHROPIC_BASE_URL),
               ("ANTHROPIC_AUTH_TOKEN", ANTHROPIC_AUTH_TOKEN),
               ("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY)):
    if _v:
        _os.environ[_k] = _v   # judge/FRQ grading reads these (slm_core._client)

import subprocess as _sp, sys as _sys0
def _pip():
    pkgs = ["transformers>=4.51", "peft>=0.11", "accelerate>=0.30", "safetensors>=0.4"]
    if LOAD_4BIT:
        pkgs.append("bitsandbytes>=0.43")
    # Qwen3.5 linear-attention fast path (~10x); optional, skip failure quietly
    _sp.run([_sys0.executable, "-m", "pip", "install", "-q", *pkgs], check=False)
    _sp.run([_sys0.executable, "-m", "pip", "install", "-q", "flash-linear-attention"], check=False)
_pip()

from pathlib import Path as _Path
def _ensure_data():
    """Make sure sft_train.jsonl + eval_items.json exist under DATA_DIR (upload if not)."""
    need = ["sft_train_v7.jsonl", "eval_items_clean.json", "lgmt300.json"]
    missing = [f for f in need if not (_Path(DATA_DIR) / f).exists()]
    if not missing:
        return
    try:
        from google.colab import files          # noqa
        print(f"[data] upload these files: {missing}")
        up = files.upload()
        for name in up:
            dest = _Path(DATA_DIR) / _Path(name).name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if _Path(name) != dest:
                _Path(dest).write_bytes(up[name])
    except Exception as e:
        raise SystemExit(
            f"[data] missing {missing} under DATA_DIR={DATA_DIR!r} and no Colab upload "
            f"available ({e}). Put the two corpus files there (build_corpus.py produces them).")
    still = [f for f in need if not (_Path(DATA_DIR) / f).exists()]
    if still:
        raise SystemExit(f"[data] still missing after upload: {still}")


# ===================== inlined slm_core.py =====================

import hashlib
import json
import os
import re
from pathlib import Path

HERE = Path(".")
DATA = Path(DATA_DIR)

# Families this build touches. lsat_lr=ReClor (arg-analysis), arct=adversarial ARCT.
FAMILIES = ["lsat_lr", "logiqa", "arct", "folio", "logicnli", "proverqa"]
FAMILY_LABEL = {
    "lsat_lr": "LSAT LR (ReClor, arg-analysis)",
    "logiqa": "LogiQA 2.0 (deductive)",
    "arct": "ARCT (adversarial warrant)",
    "folio": "FOLIO (FOL NLI)",
    "logicnli": "LogicNLI (FOL entailment)",
    "proverqa": "ProverQA (FOL entailment)",
}
# Families whose FRQ answer is a fixed entailment label -> deterministically gradable.
ENTAILMENT_FAMILIES = {"folio", "logicnli", "proverqa"}
# Families we train/eval as multiple choice (recognition).
MCQ_FAMILIES = {"lsat_lr", "arct", "logiqa"}

# ---- entailment label vocabulary -----------------------------------------
CANON = {"true": "True", "false": "False",
         "uncertain": "Uncertain", "unknown": "Unknown"}
NEUTRAL = {"Uncertain", "Unknown"}
NEUTRAL_ALIASES = [
    "cannot be determined", "can't be determined", "cannot be known",
    "does not follow", "doesn't follow", "insufficient", "indeterminate",
    "neither", "not enough information", "cannot tell", "can't tell", "neutral",
]


# --------------------------------------------------------------------------
# Loaders for the pre-built corpus (data/sft_train.jsonl, data/eval_items.json)
# --------------------------------------------------------------------------
def load_sft_rows(path: str | Path | None = None) -> list[dict]:
    """Training rows: {id, family, mode, prompt, completion, gold, ...}."""
    p = Path(path or DATA / "sft_train_v7.jsonl")
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            r = json.loads(line)
            if r.get("prompt") and r.get("completion"):
                rows.append(r)
    return rows


def load_eval_items(path: str | Path | None = None) -> list[dict]:
    """Held-out eval items (full fields) across every family."""
    p = Path(path or DATA / "eval_items_clean.json")
    return json.loads(p.read_text(encoding="utf-8"))


def train_eval_split(items: list[dict], eval_frac: float = 0.2,
                     seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Deterministic split, stable regardless of item order.

    Hashes ``group_id`` (the shared-stimulus key) when present, else ``id``. Rows that
    share a stimulus but have distinct ids — ARCT per-annotator duplicates, LogiQA
    questions over one passage, LogicNLI statements over one facts+rules block — carry a
    common ``group_id`` so they all land on the SAME side. Splitting on ``id`` alone
    leaked ~4.8% of eval stimuli into training prompts (22.7% of ARCT), inflating
    measured accuracy; grouping fixes that."""
    train, ev = [], []
    for it in items:
        key = it.get("group_id") or it["id"]
        h = hashlib.sha1(f"{seed}:{key}".encode()).hexdigest()
        frac = int(h[:8], 16) / 0xFFFFFFFF
        (ev if frac < eval_frac else train).append(it)
    return train, ev


# --------------------------------------------------------------------------
# Multiple-choice accessors
# --------------------------------------------------------------------------
def mc_of(it: dict) -> tuple[list, int]:
    if it.get("mc_choices"):
        return it["mc_choices"], int(it["mc_credited_index"])
    return it.get("choices", []), int(it.get("credited_index", -1))


def has_mc(it: dict) -> bool:
    ch, idx = mc_of(it)
    return bool(ch) and 0 <= idx < len(ch)


def lettered(choices: list) -> str:
    return "\n".join(f"({chr(65 + i)}) {c}" for i, c in enumerate(choices))


# --------------------------------------------------------------------------
# Prompt builders (MIRROR ../common.py / ../../critical-reasoning-eval)
# --------------------------------------------------------------------------
def mcq_prompt(it: dict) -> str:
    ch, _ = mc_of(it)
    q = it.get("mc_question") or it["question"]
    return (
        "You are an expert in logic and argument analysis. Read the passage and "
        "the options, then choose the SINGLE best option.\n\n"
        f"{it['stimulus']}\n\n"
        f"Question: {q}\n\n"
        f"Options:\n{lettered(ch)}\n\n"
        "Reason through it in a few sentences, then clearly commit to your final "
        "answer, naming the single correct option by its letter."
    )


def frq_prompt(it: dict) -> str:
    return (
        "You are an expert in logic and argument analysis. Answer the following "
        "open-ended question about the argument. Reason carefully and commit to "
        "the single best answer - do not hedge by listing many possibilities. Be "
        "specific and concise.\n\n"
        f"{it['stimulus']}\n\n"
        f"Question: {it['question']}\n\n"
        "Explain your reasoning in a few sentences, then clearly state your final "
        "conclusion as exactly one of the labels named in the question."
    )


def build_prompt(it: dict, mode: str) -> str:
    return mcq_prompt(it) if mode == "mcq" else frq_prompt(it)


# --------------------------------------------------------------------------
# Deterministic answer parsing + grading
# --------------------------------------------------------------------------
def _answer_tail(text: str) -> str:
    m = list(re.finditer(r"answer\s*[:\-]\s*", text, re.I))
    return text[m[-1].end():] if m else text


def _letter_to_idx(c: str, n: int) -> int | None:
    idx = ord(c.upper()) - 65
    return idx if 0 <= idx < n else None


def parse_letter(text: str, n: int) -> int | None:
    """Extract the committed option letter from natural text. Tries, in order:
    an explicit "answer: X"; a parenthesized "(X)" (last one); a committing phrase
    ("option/choice/answer ... X", last one). Returns None if none is found (the
    caller may then use the judge fallback)."""
    # 1) explicit "answer: X" (whole word "answer" so "answerS" can't match; standalone
    #    letter, prefer the last)
    for mm in reversed(re.findall(r"\banswer\b\s*[:\-]?\s*\(?([A-Za-z])\)?(?![A-Za-z])", text, re.I)):
        r = _letter_to_idx(mm, n)
        if r is not None:
            return r
    # 2) parenthesized "(X)" — the commitment is usually the last one
    for mm in reversed(re.findall(r"\(([A-Za-z])\)", text)):
        r = _letter_to_idx(mm, n)
        if r is not None:
            return r
    # 3) committing phrase, e.g. "the correct option is C" / "choice B"
    for mm in reversed(re.findall(
            r"(?:option|choice|answer|correct)\b[^.\n]{0,30}?\b([A-Ea-e])\b", text, re.I)):
        r = _letter_to_idx(mm, n)
        if r is not None:
            return r
    return None


def canon_label(s: str) -> str | None:
    return CANON.get((s or "").strip().lower())


def parse_label(text: str, neutral_hint: str = "Unknown") -> str | None:
    """Extract the committed True/False/Uncertain|Unknown label from natural text.
    Uses the LAST label mention (the conclusion typically comes last), which is
    robust to reasoning that mentions other labels earlier."""
    low = text.lower()
    hits = list(re.finditer(r"\b(true|false|uncertain|unknown)\b", low))
    if hits:
        return CANON[hits[-1].group(1)]
    if any(a in low for a in NEUTRAL_ALIASES):
        return neutral_hint
    return None


def labels_equiv(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    if a == b:
        return True
    return a in NEUTRAL and b in NEUTRAL      # Uncertain == Unknown (same class)


def neutral_variant(it: dict) -> str:
    """Which neutral token this item uses (from its choices; default Unknown)."""
    ch, _ = mc_of(it)
    if "Uncertain" in ch:
        return "Uncertain"
    if "Unknown" in ch:
        return "Unknown"
    return canon_label(it.get("reference_answer", "")) if canon_label(
        it.get("reference_answer", "")) in NEUTRAL else "Unknown"


def credited_answer(it: dict, mode: str) -> str:
    if mode == "mcq":
        ch, idx = mc_of(it)
        return f"({chr(65 + idx)}) {ch[idx]}" if 0 <= idx < len(ch) else ""
    return it.get("reference_answer", "")


def grade(it: dict, completion: str, mode: str, judge_model: str | None = None) -> dict:
    """Deterministic grade with an optional non-deterministic fallback.

    If the response can't be parsed deterministically AND ``judge_model`` is given,
    a cheap model is asked to extract the one-word answer the response commits to
    (``judge_extract``); ``used_judge`` marks when this fired. correct=None
    (needs_judge) only for open-ended lsat_lr/logiqa/arct FRQ, not used here."""
    fam = it["family"]
    if mode == "mcq":
        ch, credited = mc_of(it)
        idx = parse_letter(completion, len(ch))
        used_judge = False
        if idx is None and judge_model:
            j = judge_extract(completion, "letter", choices=ch, model=judge_model)
            if j is not None:
                idx = _letter_to_idx(j, len(ch)); used_judge = idx is not None
        return {"parsed": (chr(65 + idx) if idx is not None else None),
                "correct": (idx == credited) if idx is not None else False,
                "parseable": idx is not None, "needs_judge": False, "used_judge": used_judge}
    if fam in ENTAILMENT_FAMILIES:
        gold = canon_label(it["reference_answer"]) or it["reference_answer"]
        pred = parse_label(completion, neutral_hint=neutral_variant(it))
        used_judge = False
        if pred is None and judge_model:
            j = judge_extract(completion, "label", model=judge_model)
            pred = canon_label(j) if j else None
            used_judge = pred is not None
        return {"parsed": pred, "correct": labels_equiv(pred, gold),
                "parseable": pred is not None, "needs_judge": False, "used_judge": used_judge}
    tail = _answer_tail(completion).strip().splitlines()
    parsed = tail[0].strip() if tail else ""
    return {"parsed": parsed, "correct": None, "parseable": bool(parsed),
            "needs_judge": True, "used_judge": False}


def is_neutral_gold(it: dict, mode: str) -> bool:
    if it["family"] not in ENTAILMENT_FAMILIES:
        return False
    if mode == "mcq":
        ch, idx = mc_of(it)
        return 0 <= idx < len(ch) and ch[idx] in NEUTRAL
    return (canon_label(it.get("reference_answer", "")) or "") in NEUTRAL


# --------------------------------------------------------------------------
# Non-deterministic fallback: a cheap model extracts the committed answer when
# deterministic parsing fails (used only as a last resort, results cached).
# --------------------------------------------------------------------------
import os  # noqa: E402

# Cheap default model for the fallback extractor / verbalizer (user chose Haiku).
DEFAULT_JUDGE_MODEL = "claude-haiku-4-5"
_JUDGE_CACHE: dict = {}
_ANTHROPIC_CLIENT = None


def _load_dotenv() -> None:
    """Load KEY=VALUE lines from colab/.env (or the repo root .env) into os.environ,
    so ANTHROPIC_API_KEY can be pasted into a file instead of exported."""
    for p in (HERE / ".env", HERE.parent / ".env"):
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            # skip empty values: an empty ANTHROPIC_API_KEY would win the SDK's
            # credential precedence and break auth (per the Anthropic SDK docs)
            if k and v and not os.environ.get(k):
                os.environ[k] = v
        break


def _client():
    """Lazily build one Anthropic client from .env/env. Supports either an API key
    (ANTHROPIC_API_KEY) or a gateway/proxy (ANTHROPIC_BASE_URL + ANTHROPIC_AUTH_TOKEN,
    Bearer auth) — the latter takes precedence when a token is present."""
    global _ANTHROPIC_CLIENT
    if _ANTHROPIC_CLIENT is None:
        _load_dotenv()
        import anthropic  # imported lazily so training/eval don't require it
        kw = {}
        if os.environ.get("ANTHROPIC_BASE_URL"):
            kw["base_url"] = os.environ["ANTHROPIC_BASE_URL"]
        if os.environ.get("ANTHROPIC_AUTH_TOKEN"):
            kw["auth_token"] = os.environ["ANTHROPIC_AUTH_TOKEN"]   # -> Authorization: Bearer
        elif os.environ.get("ANTHROPIC_API_KEY"):
            kw["api_key"] = os.environ["ANTHROPIC_API_KEY"]
        _ANTHROPIC_CLIENT = anthropic.Anthropic(**kw)
    return _ANTHROPIC_CLIENT


def call_agent(prompt: str, model: str, timeout: int = 60, max_tokens: int = 1024) -> str | None:
    """One Claude API call via the Anthropic SDK. Returns the text, or None on any
    failure (missing key, network, refusal) so callers degrade gracefully."""
    try:
        resp = _client().with_options(timeout=float(timeout)).messages.create(
            model=model, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}])
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        return text or None
    except Exception:
        return None


def judge_extract(response: str, kind: str, choices: list | None = None,
                  model: str = DEFAULT_JUDGE_MODEL, timeout: int = 60) -> str | None:
    """Ask a cheap model for the ONE-word answer a response commits to.

    kind='label' -> returns True/False/Uncertain/Unknown (canonicalized).
    kind='letter' -> returns a single option letter. None if it can't tell.
    Fires only when deterministic parsing failed; results are cached in-process.
    """
    if not response or not response.strip():
        return None
    key = (kind, model, response)
    if key in _JUDGE_CACHE:
        return _JUDGE_CACHE[key]
    if kind == "letter":
        opts = " ".join(f"({chr(65 + i)})" for i in range(len(choices or []))) or "(A)…"
        q = ("Below is a model's answer to a multiple-choice question. Reply with ONLY "
             f"the single option letter it commits to (one of {opts}). If it does not "
             "commit to one, reply NONE. No other text.\n\n--- response ---\n" + response)
    else:
        q = ("Below is a model's answer to a True/False/Uncertain-style logic question. "
             "Reply with ONLY one word — True, False, Uncertain, or Unknown — matching "
             "the conclusion it commits to. If it does not commit, reply NONE. No other "
             "text.\n\n--- response ---\n" + response)
    out = call_agent(q, model, timeout, max_tokens=16)   # one-word answer
    result = None
    if out:
        if kind == "letter":
            m = re.search(r"[A-Za-z]", out)
            result = m.group(0).upper() if m and out.strip().upper() != "NONE" else None
        else:
            tok = re.search(r"\b(true|false|uncertain|unknown)\b", out, re.I)
            result = CANON[tok.group(1).lower()] if tok else None
    _JUDGE_CACHE[key] = result
    return result


# ============ inlined run_sft.py (train + batched eval) ============

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys as _sys
S = _sys.modules[__name__]   # slm_core inlined above


def build_batches(encoded, max_tokens, max_bs, seed=0):
    """Token-budget, length-sorted batches: pack examples (shortest first) until
    padded size (rows x longest-row) would exceed max_tokens or max_bs rows. Long
    sequences therefore land in SMALL batches automatically -> bounded peak VRAM
    (fixes the batch-4 OOM on long ProverQA rows). Batch order shuffled."""
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i][0]))
    batches, cur, curmax = [], [], 0
    for i in order:
        L = len(encoded[i][0])
        nmax = max(curmax, L)
        if cur and (nmax * (len(cur) + 1) > max_tokens or len(cur) >= max_bs):
            batches.append(cur); cur, curmax = [], 0; nmax = L
        cur.append(i); curmax = nmax
    if cur:
        batches.append(cur)
    batches.sort(key=lambda b: (hash((seed, b[0])) & 0xFFFF))
    return batches


def main() -> int:
    from types import SimpleNamespace
    args = SimpleNamespace(
        model=MODEL_ID, epochs=EPOCHS, batch_size=BATCH_SIZE, max_tokens=MAX_TOKENS,
        grad_accum=GRAD_ACCUM, lr=LR, lora_r=LORA_R, lora_alpha=LORA_ALPHA,
        max_len=MAX_LEN, max_steps=MAX_STEPS, save_every=SAVE_EVERY,
        train_file=(TRAIN_FILE or ""), eval_per_family=EVAL_PER_FAMILY,
        max_new_tokens=MAX_NEW_TOKENS, eval_batch=EVAL_BATCH, lgmt_eval=LGMT_EVAL,
        neutral_frac=NEUTRAL_FRAC, frq_eval=FRQ_EVAL,
        judge_model=JUDGE_MODEL, out=OUT_DIR, load_4bit=LOAD_4BIT,
        lgmt300_file=(LGMT300_FILE or ""),
        resume=RESUME, skip_updates=SKIP_UPDATES, eval_only=EVAL_ONLY, cpu=CPU)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
    from peft import LoraConfig, get_peft_model

    use_cuda = torch.cuda.is_available() and not args.cpu
    device = "cuda" if use_cuda else "cpu"
    dtype = torch.bfloat16 if use_cuda else torch.float32
    print(f"[run_sft] device={device} model={args.model} "
          f"{'('+torch.cuda.get_device_name(0)+')' if use_cuda else ''}", flush=True)

    rows = S.load_sft_rows(args.train_file or None)
    eval_items = S.load_eval_items()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def fmt(prompt, completion):
        try:
            prefix = tok.apply_chat_template([{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            prefix = tok.apply_chat_template([{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True)
        full = prefix + completion + (tok.eos_token or "")
        pids = tok(prefix, add_special_tokens=False)["input_ids"]
        fids = tok(full, add_special_tokens=False)["input_ids"][: args.max_len]
        labels = list(fids)
        for i in range(min(len(pids), len(labels))):
            labels[i] = -100
        return fids, labels

    t_enc = time.time()
    encoded = [fmt(r["prompt"], r["completion"]) for r in rows]
    dropped = sum(1 for _, y in encoded if not any(t != -100 for t in y))
    encoded = [(x, y) for x, y in encoded if any(t != -100 for t in y)]
    print(f"[run_sft] encoded {len(encoded)} rows ({dropped} dropped: prompt>={args.max_len}) "
          f"in {time.time()-t_enc:.0f}s", flush=True)

    # SDPA attention: memory-efficient (no O(seq^2) matrix), unlike eager which OOMs
    # 8 GB VRAM on long ProverQA sequences. Gradient checkpointing (below) is the
    # other half of fitting an 8 GB laptop GPU. --load-4bit adds nf4 QLoRA so bigger
    # bases (e.g. Qwen3.5-4B: ~2 GB in 4-bit) fit 8 GB.
    quant_config = None
    if args.load_4bit and use_cuda:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=dtype, quantization_config=quant_config,
        attn_implementation="sdpa", device_map=(device if use_cuda else None))
    if not use_cuda:
        model = model.to(device)
    if quant_config is not None:
        from peft import prepare_model_for_kbit_training
        # GC handled below with our sm_120-safe kwargs, not by prepare_* (reentrant)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model.config.use_cache = False
    if use_cuda and args.load_4bit:            # GC only needed for the memory-constrained 4-bit path
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={
            "use_reentrant": False, "preserve_rng_state": False})
        model.enable_input_require_grads()
    if args.resume and (Path(args.resume) / "adapter_config.json").exists():
        from peft import PeftModel
        # load the prior adapter onto the (already kbit-prepared, GC-enabled) base and
        # keep it trainable -> continue the same run from a checkpoint after a crash.
        model = PeftModel.from_pretrained(model, args.resume, is_trainable=True)
        print(f"[run_sft] RESUMED adapter from {args.resume} "
              f"(skip {args.skip_updates} updates)", flush=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    pad = tok.pad_token_id

    def make_tensor(idxs):
        batch = [encoded[i] for i in idxs]
        maxlen = max(len(x) for x, _ in batch)
        ii, ll, aa = [], [], []
        for x, y in batch:
            n = maxlen - len(x)
            ii.append(x + [pad]*n); ll.append(y + [-100]*n); aa.append([1]*len(x) + [0]*n)
        return (torch.tensor(ii, device=device), torch.tensor(ll, device=device),
                torch.tensor(aa, device=device))

    batches = build_batches(encoded, args.max_tokens, args.batch_size)
    print(f"[run_sft] {len(batches)} token-budget batches "
          f"(<= {args.max_tokens} tok or {args.batch_size} rows each)", flush=True)
    updates_per_epoch = max(1, len(batches) // args.grad_accum)   # real optimizer steps
    total_updates = (args.max_steps or args.epochs * updates_per_epoch)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    # scheduler counts one step() per OPTIMIZER update (called every grad_accum micro-batches)
    sched = get_cosine_schedule_with_warmup(
        opt, max(1, int(0.03 * total_updates)), total_updates)
    # --resume: fast-forward the LR schedule to where the crashed run stopped so the
    # decay continues smoothly (Adam moments restart fresh — a benign warm restart).
    skip_updates = min(args.skip_updates, total_updates) if args.resume else 0
    for _ in range(skip_updates):
        sched.step()

    model.train()
    model.config.use_cache = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    t0, upd, micro, accum_tokens, losses = time.time(), skip_updates, 0, 0, []

    def optimizer_step(epoch):
        # normalize the token-weighted accumulated grads to a true token-mean,
        # then clip/step. Called every grad_accum micro-batches + once at the end
        # to flush the trailing partial window (so no batches are dropped).
        nonlocal accum_tokens, upd
        if accum_tokens == 0:
            return
        inv = 1.0 / accum_tokens
        for p in trainable:
            if p.grad is not None:
                p.grad.mul_(inv)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        accum_tokens = 0
        upd += 1
        if upd % 25 == 0 or upd == 1:
            print(f"  upd {upd}/{total_updates} ep{epoch} loss {losses[-1]:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        # periodic checkpoint: overwrite one adapter dir so a killed run leaves a
        # recent, usable adapter behind. Cheap (LoRA is tens of MB) — see --save-every.
        if args.save_every and upd % args.save_every == 0:
            ck = Path(args.out) / "checkpoint"; ck.mkdir(parents=True, exist_ok=True)
            ts = time.time()
            model.save_pretrained(str(ck)); tok.save_pretrained(str(ck))
            # record progress so a supervisor can --resume ... --skip-updates N exactly
            (ck / "progress.json").write_text(json.dumps(
                {"updates_done": upd, "total": total_updates, "loss": losses[-1]}))
            print(f"    [checkpoint] upd {upd} -> {ck} ({time.time()-ts:.1f}s)", flush=True)

    if args.eval_only:
        train_secs = 0.0
        print("[run_sft] --eval-only: skipping training; evaluating the loaded adapter "
              f"(resume={args.resume or 'NONE — WARNING: fresh random adapter'})", flush=True)
    else:
        stop = False
        skip_micro = skip_updates * args.grad_accum   # micro-batches already trained pre-crash
        seen = 0
        for epoch in range(args.epochs):
            for idxs in batches:
                if seen < skip_micro:                 # fast-skip done windows (same batch order)
                    seen += 1
                    continue
                ii, ll, aa = make_tensor(idxs)
                out = model(input_ids=ii, attention_mask=aa, labels=ll)
                n_tok = int((ll != -100).sum().item())            # supervised tokens
                (out.loss * n_tok).backward()                      # token-weighted accum
                accum_tokens += n_tok
                micro += 1
                losses.append(float(out.loss.detach()))
                if micro % args.grad_accum == 0:
                    optimizer_step(epoch)
                    if args.max_steps and upd >= args.max_steps:
                        stop = True; break
            if stop:
                break
        if not stop:
            optimizer_step(args.epochs - 1)                        # flush trailing window

        out_dir = Path(args.out) / "adapter"
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(out_dir)); tok.save_pretrained(str(out_dir))
        train_secs = time.time() - t0
        # losses can be empty when a resume skips every micro-batch (e.g. --skip-updates
        # equals total_updates and grad_accum divides len(batches)) -> guard the summary
        # so the run proceeds to eval instead of crashing with IndexError.
        loss_str = f"{losses[0]:.2f}->{losses[-1]:.2f}" if losses else "n/a (no micro-batches)"
        print(f"[run_sft] trained {upd} updates, loss {loss_str}, "
              f"{train_secs:.0f}s -> {out_dir}", flush=True)

    # ---- base-vs-tuned eval on the held-out split (deterministic) ----
    # Batched generation: single-sequence decoding is memory-bound (the weights are
    # re-read from VRAM for every token, GPU mostly idle); left-padding a batch of
    # prompts into ONE generate() call amortizes those reads -> much higher throughput.
    def _chat(it, mode):
        prompt = S.build_prompt(it, mode)
        try:
            return tok.apply_chat_template([{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template([{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def gen_many(items, modes):
        """Generate for a batch of (item, mode). Decoder-only models must LEFT-pad so
        every sequence's generated tokens begin at the same offset (plen)."""
        texts = [_chat(it, m) for it, m in zip(items, modes)]
        prev_side = tok.padding_side
        tok.padding_side = "left"
        enc = tok(texts, return_tensors="pt", padding=True).to(device)
        tok.padding_side = prev_side
        o = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                           pad_token_id=(tok.pad_token_id or tok.eos_token_id))
        plen = enc["input_ids"].shape[1]
        return [tok.decode(o[i][plen:], skip_special_tokens=True) for i in range(len(texts))]

    def gen_chunked(items, modes):
        """Batched generation over --eval-batch-sized chunks; preserves input order."""
        out, bs = [], max(1, args.eval_batch)
        for i in range(0, len(items), bs):
            out.extend(gen_many(items[i:i+bs], modes[i:i+bs]))
        return out

    # representative sample: even stride across each family's (ordered) eval slice
    # so we don't take only the first N (ProverQA is ordered easy->medium->hard,
    # so first-N would be all-easy and inflate its accuracy). For the entailment
    # families we ALSO label-stratify to hit ~--neutral-frac neutral overall, so the
    # hard 'does-not-follow' class isn't a tiny (statistically useless) denominator.
    def _stride(items, k):
        k = min(k, len(items))
        if k <= 0:
            return []
        s = len(items) / k
        return [items[int(i * s)] for i in range(k)]

    by_fam = defaultdict(list)
    for it in eval_items:
        by_fam[it["family"]].append(it)
    fams = list(by_fam)
    ent_fams = [f for f in fams if f in S.ENTAILMENT_FAMILIES]
    per = args.eval_per_family
    # per-entailment-family neutral quota s.t. overall neutral ~= neutral_frac
    # (satellite families have no neutral gold; overall = ent_share * within-ent neutral)
    neu_quota = 0
    if ent_fams and args.neutral_frac > 0 and per > 0:
        neu_quota = min(per, round(args.neutral_frac * len(fams) * per / len(ent_fams)))
    picked = []
    for fam, fam_items in by_fam.items():
        n = min(per, len(fam_items))
        if n <= 0:
            continue
        if fam in ent_fams and neu_quota > 0:
            neu = [it for it in fam_items if S.is_neutral_gold(it, "frq")]
            non = [it for it in fam_items if not S.is_neutral_gold(it, "frq")]
            kneu = min(neu_quota, len(neu), n)
            picked += _stride(neu, kneu) + _stride(non, n - kneu)
        else:
            picked += _stride(fam_items, n)

    judge = args.judge_model or None
    # gradeable (item, mode) list, computed once and shared by both conditions
    graded_items = []
    for it in picked:
        mode = "frq" if it["family"] in S.ENTAILMENT_FAMILIES else "mcq"
        if mode == "mcq" and not S.has_mc(it):
            continue
        graded_items.append((it, mode))

    def run_condition(label):
        b = defaultdict(lambda: {"n":0,"parse":0,"correct":0,"neu_n":0,"neu_c":0,"judged":0})
        _t = time.time()
        items = [it for it, _ in graded_items]
        modes = [m for _, m in graded_items]
        comps, bs = [], max(1, args.eval_batch)
        for i in range(0, len(items), bs):
            comps.extend(gen_many(items[i:i+bs], modes[i:i+bs]))
            done = min(i + bs, len(items))
            print(f"  [{label}] {done}/{len(items)} generations "
                  f"({time.time()-_t:.0f}s, {(time.time()-_t)/max(1,done):.1f}s/gen, bs={bs})",
                  flush=True)
        samples = []
        for (it, mode), comp in zip(graded_items, comps):
            g = S.grade(it, comp, mode, judge_model=judge)
            gold = S.credited_answer(it, mode)
            samples.append({"condition": label.strip(), "id": it.get("id"),
                            "family": it["family"], "mode": mode, "gold": gold,
                            "neutral_gold": S.is_neutral_gold(it, mode),
                            "parseable": bool(g["parseable"]), "correct": bool(g["correct"]),
                            "needs_judge": bool(g["needs_judge"]), "completion": comp})
            if g["needs_judge"]:
                continue
            k = b[it["family"]]
            k["n"] += 1; k["parse"] += int(g["parseable"]); k["correct"] += int(bool(g["correct"]))
            k["judged"] += int(g.get("used_judge", False))
            if S.is_neutral_gold(it, mode):
                k["neu_n"] += 1; k["neu_c"] += int(bool(g["correct"]))
        tot = defaultdict(int)
        for v in b.values():
            for kk, vv in v.items(): tot[kk] += vv
        pct = lambda a, d: round(100*a/d, 1) if d else None
        print(f"[{label}] n={tot['n']} parse={pct(tot['parse'],tot['n'])}% "
              f"acc={pct(tot['correct'],tot['n'])}% neutral_recall={pct(tot['neu_c'],tot['neu_n'])}%",
              flush=True)
        return b, tot, samples

    model.eval()
    model.config.use_cache = True                    # KV cache -> fast generation
    try:
        model.gradient_checkpointing_disable()       # training-only; slows generate
    except Exception:
        pass
    print(f"\n[run_sft] base-vs-tuned eval on {len(picked)} held-out items "
          f"({args.eval_per_family}/family)")
    with model.disable_adapter():
        base_b, base_t, base_s = run_condition("base ")
    tuned_b, tuned_t, tuned_s = run_condition("tuned")
    # raw-answer log: every item's completion + gold + verdict, for auditing (e.g.
    # is a 100% neutral recall real, or a tiny/unbalanced denominator?)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with (Path(args.out) / "eval_samples.jsonl").open("w", encoding="utf-8") as fh:
        for r in base_s + tuned_s:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[run_sft] raw eval answers -> {Path(args.out)/'eval_samples.jsonl'} "
          f"({len(base_s)+len(tuned_s)} rows)", flush=True)
    pct = lambda a, d: round(100*a/d, 1) if d else None
    print("\nby family (base_acc -> tuned_acc, n) [neutral]")
    for fam in sorted(set(base_b) | set(tuned_b)):
        bb, tt = base_b[fam], tuned_b[fam]
        neu = f"  neutral {pct(bb['neu_c'],bb['neu_n'])}%->{pct(tt['neu_c'],tt['neu_n'])}%" if tt["neu_n"] else ""
        print(f"  {fam:9} {pct(bb['correct'],bb['n'])}% -> {pct(tt['correct'],tt['n'])}% (n={tt['n']}){neu}")

    # ---- LGMT consistency (MVR / HDR): does the label survive a logic-preserving
    # reformulation? Append an irrelevant premise (the P3 over-inference probe) and
    # check the entailment label is unchanged. MVR = label-flip rate; HDR = flips among
    # answers that were correct on the source (the defects static accuracy hides). ----
    _IRR = "Additionally, an unrelated fact holds: Quennell owns a teal kite."
    def lgmt_consistency(label):
        ent = [it for it in eval_items if it["family"] in S.ENTAILMENT_FAMILIES]
        n = min(args.lgmt_eval, len(ent))
        if n <= 0:
            return None
        stepn = len(ent) / n
        sample = [ent[int(i * stepn)] for i in range(n)]
        fu_items = []
        for it in sample:
            fu = dict(it); fu["stimulus"] = it["stimulus"].rstrip() + " " + _IRR
            fu_items.append(fu)
        # batched: all source prompts, then all irrelevant-premise follow-ups
        src_out = gen_chunked(sample, ["frq"] * len(sample))
        fu_out = gen_chunked(fu_items, ["frq"] * len(sample))
        viol = hidden = usable = 0
        for it, s_txt, f_txt in zip(sample, src_out, fu_out):
            gold = S.canon_label(it["reference_answer"]) or it["reference_answer"]
            hint = S.neutral_variant(it)
            src = S.parse_label(s_txt, neutral_hint=hint)
            flab = S.parse_label(f_txt, neutral_hint=hint)
            if src is None or flab is None:
                continue
            usable += 1
            if not S.labels_equiv(src, flab):
                viol += 1
                if S.labels_equiv(src, gold):
                    hidden += 1
        mvr, hdr = pct(viol, usable), pct(hidden, usable)
        print(f"[{label}] LGMT consistency (n={usable}): MVR {mvr}%  HDR {hidden}/{usable}={hdr}%",
              flush=True)
        return {"n": usable, "mvr": mvr, "hdr": hdr}

    lgmt = {}
    if args.lgmt_eval:
        print(f"\n[run_sft] LGMT consistency probe ({args.lgmt_eval} entailment items/condition)")
        with model.disable_adapter():
            lgmt["base"] = lgmt_consistency("base ")
        lgmt["tuned"] = lgmt_consistency("tuned")

    # ---- full-20-MR LGMT probe (tuned): the fixed lgmt300.json set, same MVR/HDR as the
    # gateway lgmt300_eval.py, so the SLM row slots into data-synthesis.md's LGMT-300 table. ----
    def lgmt300_consistency(path):
        d3 = json.loads(Path(path).read_text(encoding="utf-8"))
        src3, cases3 = d3["sources"], d3["cases"]
        def _it(prem, concl, gold):
            stim = "Premises:\n" + "\n".join(f"- {x}" for x in prem) + f"\n\nConclusion: {concl}"
            return {"stimulus": stim, "family": "folio", "mode": "frq", "reference_answer": gold,
                    "question": ("Based only on the premises, is the conclusion True, False, or "
                                 "Unknown (it does not deductively follow either way)?")}
        sids = list(src3)
        s_out = gen_chunked([_it(src3[i]["premises"], src3[i]["conclusion"], src3[i]["gold"]) for i in sids],
                            ["frq"] * len(sids))
        c_out = gen_chunked([_it(m["premises"], m["conclusion"], m["gold"]) for m in cases3],
                            ["frq"] * len(cases3))
        slab = {sids[i]: S.parse_label(s_out[i], neutral_hint="Unknown") for i in range(len(sids))}
        per = {}; V = H = N = accs = accn = 0
        for m, txt in zip(cases3, c_out):
            yf = S.parse_label(txt, neutral_hint="Unknown"); ys = slab.get(m["item_id"]); g = m["gold"]
            if ys is None or yf is None:
                continue
            N += 1; viol = not S.labels_equiv(ys, yf); V += int(viol)
            if viol and S.labels_equiv(ys, S.canon_label(g) or g):
                H += 1
            cat = m.get("category", "?"); per.setdefault(cat, [0, 0])
            per[cat][0] += 1; per[cat][1] += int(viol)
        for i in sids:
            if slab[i] is not None:
                accn += 1; accs += int(S.labels_equiv(slab[i], S.canon_label(src3[i]["gold"]) or src3[i]["gold"]))
        return {"n": N, "mvr": pct(V, N), "hdr": pct(H, N), "acc_static": pct(accs, accn),
                "by_category": {c: {"n": v[0], "mvr": pct(v[1], v[0])} for c, v in sorted(per.items())}}

    lgmt300 = {}
    if args.lgmt300_file:
        p300 = Path(args.lgmt300_file)
        if not p300.exists():
            p300 = DATA / args.lgmt300_file
        if p300.exists():
            print(f"\n[run_sft] full-20-MR LGMT probe (tuned) <- {p300.name}", flush=True)
            lgmt300 = lgmt300_consistency(p300)
            print(f"[tuned] LGMT-300 (n={lgmt300['n']}): Acc_static {lgmt300['acc_static']}% | "
                  f"MVR {lgmt300['mvr']}% | HDR {lgmt300['hdr']}%", flush=True)
            for c, v in lgmt300["by_category"].items():
                print(f"    {c}: n={v['n']} MVR {v['mvr']}%", flush=True)
        else:
            print(f"[run_sft] LGMT300_FILE '{args.lgmt300_file}' not found — skipping", flush=True)

    # ---- optional satellite FRQ probe (EVAL-ONLY; training stays MCQ). Pose the
    # OPEN-ENDED question (state the flaw / warrant / assumption) instead of MCQ and
    # judge semantic correctness with the gateway. This measures the generation /
    # identifies-target weakness that MCQ recognition hides; expect low absolute
    # scores (train-MCQ/eval-FRQ mismatch) — it's a diagnostic, read base-vs-tuned. ----
    _SAT_OPEN_Q = {
        "flaw": "Identify the flaw in the argument's reasoning. State it specifically.",
        "assumption": "State the unstated assumption the argument depends on.",
        "weaken": "State what would most weaken this argument.",
        "warrant": "State the implicit warrant — the unstated assumption that makes the reason support the claim.",
        "inference": "State the single conclusion that can be validly drawn from the information.",
    }
    def _frq_prompt_sat(it):
        q = _SAT_OPEN_Q.get(it.get("task_type", "")) or (it.get("question") or "Answer the question.")
        return (f"{it['stimulus'].strip()}\n\n{q}\n\nReason briefly from the argument, then state "
                "your answer in a final sentence.")

    @torch.no_grad()
    def gen_prompts_chunked(prompts):   # generate from RAW prompt strings (not S.build_prompt)
        texts = []
        for p in prompts:
            try:
                texts.append(tok.apply_chat_template([{"role": "user", "content": p}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False))
            except TypeError:
                texts.append(tok.apply_chat_template([{"role": "user", "content": p}],
                    tokenize=False, add_generation_prompt=True))
        out, bs = [], max(1, args.eval_batch)
        for i in range(0, len(texts), bs):
            chunk = texts[i:i+bs]
            prev = tok.padding_side; tok.padding_side = "left"
            enc = tok(chunk, return_tensors="pt", padding=True).to(device); tok.padding_side = prev
            o = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                               pad_token_id=(tok.pad_token_id or tok.eos_token_id))
            plen = enc["input_ids"].shape[1]
            out += [tok.decode(o[j][plen:], skip_special_tokens=True) for j in range(len(chunk))]
        return out

    def _judge_frq(candidate, reference, model):
        if not candidate or not candidate.strip():
            return None
        q = ("Grade an open-ended answer to a critical-reasoning question against a reference "
             "answer. Reply with ONLY 'YES' if the candidate makes essentially the same point "
             "as the reference (a defensible paraphrase counts), otherwise 'NO'.\n\n"
             f"Reference:\n{reference}\n\nCandidate:\n{candidate}\n\nSame point? (YES/NO)")
        out = S.call_agent(q, model, timeout=60, max_tokens=8)
        return None if not out else out.strip().upper().startswith("Y")

    frq = {}
    if args.frq_eval:
        jm = args.judge_model or S.DEFAULT_JUDGE_MODEL
        byf = defaultdict(list)
        for it in eval_items:
            if it["family"] in S.MCQ_FAMILIES:
                byf[it["family"]].append(it)
        fpick = []
        for fam, its in byf.items():
            fpick += _stride(its, min(args.frq_eval, len(its)))
        prompts = [_frq_prompt_sat(it) for it in fpick]
        print(f"\n[run_sft] satellite FRQ probe: {len(fpick)} open-ended items, judge={jm}", flush=True)
        def frq_condition(label):
            outs = gen_prompts_chunked(prompts)
            b = defaultdict(lambda: {"n": 0, "c": 0})
            fsamp = []
            for it, o in zip(fpick, outs):
                ref = S.credited_answer(it, "mcq")
                v = _judge_frq(o, ref, jm)
                fsamp.append({"condition": label.strip(), "probe": "frq", "id": it.get("id"),
                              "family": it["family"], "task_type": it.get("task_type"),
                              "reference": ref, "judged_correct": v, "completion": o})
                if v is not None:
                    k = b[it["family"]]; k["n"] += 1; k["c"] += int(v)
            tot = defaultdict(int)
            for v in b.values():
                for kk, vv in v.items(): tot[kk] += vv
            print(f"[{label}] FRQ correctness n={tot['n']} acc={pct(tot['c'],tot['n'])}%", flush=True)
            return ({fam: {"acc": pct(b[fam]['c'], b[fam]['n']), "n": b[fam]['n']} for fam in b},
                    pct(tot['c'], tot['n']), tot['n'], fsamp)
        if fpick:
            with model.disable_adapter():
                fb_by, fb_acc, fb_n, fb_s = frq_condition("base ")
            ft_by, ft_acc, ft_n, ft_s = frq_condition("tuned")
            with (Path(args.out) / "eval_samples.jsonl").open("a", encoding="utf-8") as fh:
                for r in fb_s + ft_s:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            frq = {"judge": jm, "base": {"acc": fb_acc, "n": fb_n, "by_family": fb_by},
                   "tuned": {"acc": ft_acc, "n": ft_n, "by_family": ft_by}}
            if fb_n == 0 and ft_n == 0:
                print("[run_sft] FRQ probe: judge returned nothing (no gateway creds?)", flush=True)

    metrics = {"model": args.model, "device": device, "updates": upd,
               "train_seconds": round(train_secs, 1),
               "eval_only": args.eval_only, "resumed_from": args.resume or None,
               "loss_first": round(losses[0], 3) if losses else None,
               "loss_last": round(losses[-1], 3) if losses else None,
               "eval_per_family": args.eval_per_family, "neutral_frac": args.neutral_frac,
               "neutral_n": base_t['neu_n'],
               "base": {"parse": pct(base_t['parse'], base_t['n']),
                        "acc": pct(base_t['correct'], base_t['n']),
                        "neutral": pct(base_t['neu_c'], base_t['neu_n'])},
               "tuned": {"parse": pct(tuned_t['parse'], tuned_t['n']),
                         "acc": pct(tuned_t['correct'], tuned_t['n']),
                         "neutral": pct(tuned_t['neu_c'], tuned_t['neu_n'])},
               "lgmt": lgmt, "lgmt300": lgmt300, "frq": frq}
    Path(args.out).mkdir(parents=True, exist_ok=True)   # eval-only skips the adapter-save mkdir
    (Path(args.out) / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\n[run_sft] metrics -> {Path(args.out)/'metrics.json'}")
    return 0


# ------------------------------- run -------------------------------
_ensure_data()
main()
