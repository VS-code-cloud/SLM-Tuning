"""Self-contained core for the Colab SFT of Qwen3.5-0.8B.

Bundles everything the notebook needs with NO dependency on the parent repo or on
``cursor-agent``: the prompt contract (mirrors ``../common.py`` ->
``../../critical-reasoning-eval``), deterministic grading, a hashed train/eval
split, and loaders for the **pre-built** SFT corpus produced by
``build_corpus.py`` (the house-style verbalized traces are baked into
``data/sft_train.jsonl`` — explanations are generated at build time, not during
training).

Two regimes, never merged (see ../PLAN.md "Day 3+"):
  * entailment core  (folio, logicnli, proverqa) -> FRQ label True/False/neutral
  * argument satellite (lsat_lr, arct, logiqa)    -> MCQ option letter
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

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
    p = Path(path or DATA / "sft_train.jsonl")
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
    p = Path(path or DATA / "eval_items.json")
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


if __name__ == "__main__":
    from collections import Counter
    rows = load_sft_rows()
    ev = load_eval_items()
    print(f"train rows: {len(rows)}  ",
          dict(Counter((r['family'], r['mode']) for r in rows)))
    print(f"eval items: {len(ev)}  ", dict(Counter(i['family'] for i in ev)))
    print("neutral-gold eval items:",
          sum(1 for it in ev if is_neutral_gold(it, "frq" if it['family'] in ENTAILMENT_FAMILIES else "mcq")))
