#!/usr/bin/env python3
"""NL-conclusion → FOL for FOLIO *train* (it ships premises-FOL but NO conclusion-FOL,
unlike the validation split). Filling this one hole lets z3 solve FOLIO-train exactly
like FOLIO-val. Three-stage, z3-VERIFIED pipeline (all constrained to the row's own
predicate/constant lexicon, extracted from its premises-FOL):

  1. **Deterministic lexicon-constrained mapper** — offline, no model. Maps the simple
     atomic / single-negation / both-and / either-or conclusions to FOL by matching
     content tokens to the lexicon predicates and the named entity to a constant. High
     precision (z3 gates it), modest recall on FOLIO's more elaborate conclusions.
  2. **Opus blind** — ask Opus to translate the conclusion to FOL over the lexicon,
     WITHOUT being told the gold relation.
  3. **Opus answer-conditioned retries** — if blind doesn't verify, tell Opus the gold
     relation (entailed / contradicted / neither) and ask for the reading (negation
     scope, quantifier, arity) that yields it; retry up to 3×.

Every candidate is accepted ONLY if z3 (`fol_solver.classify`) decides on
(premises-FOL, candidate) AND the label equals the gold — so a wrong translation is
never kept (it falls back to the template downstream). Verified results are cached; a
genuinely-exhausted item is cached as a sentinel so it isn't re-attempted (delete the
cache to force a full re-translation). Gated on gateway creds + z3: with neither, only
the deterministic stage runs.

    python folio_concl_fol.py --limit 40      # sample: coverage by stage
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
_COLAB = HERE.parent
if str(_COLAB) not in sys.path:
    sys.path.insert(0, str(_COLAB))

import fol
try:
    import fol_solver
except Exception:
    fol_solver = None
try:
    import slm_core as S
except Exception:
    S = None

CACHE = _COLAB / "data" / "_raw" / "folio_train_concl_fol.json"
_LOCK = threading.Lock()
_FAIL = "__FAIL__"                       # sentinel: attempted, z3 never verified
OPUS = "claude-opus-4-8"
_NEU = {"Uncertain", "Unknown"}


def _lexicon(premises_fol):
    preds, consts = set(), set()
    for p in premises_fol or []:
        try:
            a = fol.parse(p)
            preds |= fol.predicates(a)
            consts |= fol.constants(a)
        except Exception:
            continue
    return sorted(preds), sorted(consts)


# ------------------------------------------------------------------ z3 verification
def _labels_equiv(a, b):
    if S is not None:
        return S.labels_equiv(a, b)
    return a is not None and (a == b or (a in _NEU and b in _NEU))


def _verify(premises_fol, concl_fol, gold) -> bool:
    """True iff z3 decides (premises ⊨ concl / ⊨¬concl / neither) AND matches the gold."""
    if fol_solver is None or not concl_fol:
        return False
    try:
        fol.parse(concl_fol)
        lab, _ = fol_solver.classify([p for p in premises_fol if str(p).strip()], concl_fol)
    except Exception:
        return False
    return lab is not None and _labels_equiv(lab, gold)


# ------------------------------------------------------------------ (1) deterministic mapper
_STOP = {"a", "an", "the", "is", "are", "was", "were", "be", "being", "who", "that", "of",
         "to", "and", "or", "not", "both", "either", "neither", "nor", "person", "thing",
         "it", "its", "he", "she", "they", "as", "in", "on", "for", "about", "there"}


def _tokens(s):
    return [w for w in re.findall(r"[a-z]+", s.lower()) if w not in _STOP and len(w) > 2]


def _pred_tokens(pred):                  # split CamelCase / underscores -> lowercase tokens
    parts = re.findall(r"[A-Z][a-z]+|[a-z]+", pred.replace("_", " "))
    return {p.lower() for p in parts}


def _best_pred(content, preds):
    """Best unary predicate whose token set overlaps the content tokens; needs a clear win."""
    ct = set(content)
    scored = sorted(((len(_pred_tokens(p) & ct), p) for p in preds), reverse=True)
    if not scored or scored[0][0] == 0:
        return None
    if len(scored) > 1 and scored[0][0] == scored[1][0]:      # ambiguous tie -> bail
        return None
    return scored[0][1]


def _deterministic_map(conclusion_nl, preds, consts):
    """Map ONLY the confidently-atomic conclusions; bail (None) on conditionals /
    quantifiers / multi-predicate combos, leaving those to Opus. z3 gates the result."""
    c = conclusion_nl.strip().rstrip(".")
    low = c.lower()
    if any(k in low for k in (" if ", "if ", "then", "every", "all ", "some ", "each ",
                              " implies", "unless")):
        return None                                          # not a simple atomic -> Opus
    # the named entity -> a lexicon constant (case-insensitive on the constant name)
    cl = {k.lower(): k for k in consts}
    ent = None
    for w in re.findall(r"[A-Za-z][A-Za-z]+", c):
        if w.lower() in cl:
            ent = cl[w.lower()]; break
    if ent is None:
        return None
    neg = bool(re.search(r"\b(not|no|never|n't)\b", low)) and " nor " not in low
    unary = [p for p in preds]                               # try all; _best_pred filters
    # "both A and B" / "either A or B" over two predicates about the same entity
    m = re.search(r"both (.+?) and (.+)$", low) or re.search(r"either (.+?) or (.+)$", low)
    if m:
        pa, pb = _best_pred(_tokens(m.group(1)), unary), _best_pred(_tokens(m.group(2)), unary)
        if pa and pb and pa != pb:
            op = "∧" if "both" in low else "∨"
            body = f"({pa}({ent}) {op} {pb}({ent}))"
            return f"¬{body}" if neg else body
        return None
    # plain atomic: one predicate about the entity
    p = _best_pred(_tokens(c), unary)
    if not p:
        return None
    return f"¬{p}({ent})" if neg else f"{p}({ent})"


# ------------------------------------------------------------------ (2/3) Opus translation
def _prompt_blind(conclusion_nl, preds, consts):
    return ("Translate the English conclusion into ONE first-order-logic formula.\n"
            f"Use ONLY these predicate symbols: {preds}\n"
            f"Use ONLY these constant symbols: {consts}\n"
            "Variables may be lowercase x, y, z. Operators: ∀ ∃ ¬ ∧ ∨ → ↔ (and ⊕). Match "
            "predicate names and arity EXACTLY (some are binary, e.g. Love(x, music)). "
            "Output ONLY the formula on one line, no prose, no code fence.\n\n"
            f"Conclusion: {conclusion_nl}")


def _prompt_answer(conclusion_nl, preds, consts, gold):
    rel = {"True": "logically ENTAILED by the premises",
           "False": "CONTRADICTED by the premises (its negation is entailed)",
           "Uncertain": "NEITHER entailed nor contradicted by the premises",
           "Unknown": "NEITHER entailed nor contradicted by the premises"}.get(gold, "")
    return ("Translate the English conclusion into ONE first-order-logic formula, using ONLY "
            "the given symbols. Known fact: a faithful translation is such that the conclusion "
            f"is {rel}. Choose the reading (negation scope, quantifier, arity) that makes this "
            "hold — but it must be a faithful translation, not an arbitrary formula.\n"
            f"Predicates: {preds}\nConstants: {consts}\n"
            "Operators: ∀ ∃ ¬ ∧ ∨ → ↔. Output ONLY the formula on one line.\n\n"
            f"Conclusion: {conclusion_nl}")


def _extract(resp):
    if not resp:
        return None
    return resp.strip().splitlines()[0].strip().strip("`").strip()


_llm_up = None


def _llm_reachable(model, timeout) -> bool:
    """One-time probe: is the model gateway actually reachable? Avoids hammering a dead
    gateway with hundreds of slow failing calls when creds/connectivity are absent."""
    global _llm_up
    if _llm_up is None:
        if S is None:
            _llm_up = False
        else:
            try:
                _llm_up = bool(S.call_agent("Reply with only: OK", model, min(timeout, 20), max_tokens=4))
            except Exception:
                _llm_up = False
        if not _llm_up:
            print("[folio_concl_fol] model gateway unreachable — deterministic stage only "
                  "(Opus stages will run where the gateway/creds are live).")
    return _llm_up


def _load_cache() -> dict:
    for path in (CACHE, CACHE.with_suffix(CACHE.suffix + ".bak")):
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return {k: v for k, v in data.items() if v}
            except Exception:
                print(f"[folio_concl_fol] WARNING: cache at {path} unreadable; trying backup")
                continue
    return {}


_CACHE = _load_cache()


def _store(key, val):
    with _LOCK:
        _CACHE[key] = val


def translate(key, conclusion_nl, premises_fol, gold=None, model=OPUS, timeout=90, tries=3):
    """Return a z3-VERIFIED conclusion-FOL (cached), or None. Stages: deterministic map →
    Opus blind → Opus answer-conditioned (≤`tries`). Verified against `gold` via z3."""
    with _LOCK:
        cached = _CACHE.get(key)
    if cached == _FAIL:
        return None
    if cached and (gold is None or _verify(premises_fol, cached, gold)):
        return cached                                        # cached & still verifies
    preds, consts = _lexicon(premises_fol)
    if not preds:
        return None

    def ok(cand):
        if not cand:
            return False
        if gold is None:                                     # can't z3-verify -> parse-only
            try:
                fol.parse(cand); return True
            except Exception:
                return False
        return _verify(premises_fol, cand, gold)

    # (1) deterministic, offline
    cand = _deterministic_map(conclusion_nl, preds, consts)
    if ok(cand):
        _store(key, cand); return cand

    # (2) Opus blind, (3) Opus answer-conditioned retries
    responded = False
    if S is not None and _llm_reachable(model, timeout):
        prompts = [_prompt_blind(conclusion_nl, preds, consts)]
        if gold is not None:
            prompts += [_prompt_answer(conclusion_nl, preds, consts, gold)] * tries
        for pr in prompts:
            r = S.call_agent(pr, model, timeout, max_tokens=200)
            if r:
                responded = True
                cand = _extract(r)
                if ok(cand):
                    _store(key, cand); return cand
    if responded and gold is not None:                       # genuinely exhausted -> sentinel
        _store(key, _FAIL)
    return None


def save_cache():
    with _LOCK:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(_CACHE, ensure_ascii=False, indent=0)
        tmp = CACHE.with_suffix(CACHE.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        if CACHE.exists():
            try:
                os.replace(CACHE, CACHE.with_suffix(CACHE.suffix + ".bak"))
            except Exception:
                pass
        os.replace(tmp, CACHE)


def prewarm(items, model=OPUS, concurrency=8):
    """Translate+cache for FOLIO-train items concurrently. Items need id, _statement,
    _premises_fol, and (for verification) reference_answer."""
    todo = [it for it in items if it.get("_premises_fol")]
    if not todo:
        return 0
    def _one(it):
        g = it.get("reference_answer")
        g = S.canon_label(g) if (S and g) else g
        return translate(it["id"], it.get("_statement", ""), it.get("_premises_fol"), gold=g, model=model)
    done = 0
    with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
        for _ in ex.map(_one, todo):
            done += 1
    save_cache()
    return done


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--model", default=OPUS)
    ap.add_argument("--no-llm", action="store_true", help="deterministic stage only")
    args = ap.parse_args()
    raw = _COLAB / "data" / "_raw" / "folio_train.jsonl"
    rows = [json.loads(l) for l in raw.read_text(encoding="utf-8").splitlines() if l.strip()]
    CANON = {"true": "True", "false": "False", "uncertain": "Uncertain", "unknown": "Uncertain"}
    items, n = [], 0
    for i, d in enumerate(rows):
        pf = d.get("premises-FOL") or []
        if not pf:
            continue
        items.append({"id": f"folio-t-{d.get('example_id', i)}",
                      "_statement": d.get("conclusion", ""), "_premises_fol": pf,
                      "reference_answer": CANON.get(str(d.get("label", "")).lower())})
        n += 1
        if n >= args.limit:
            break
    det = det_ok = llm_ok = 0
    for it in items:
        g = it["reference_answer"]
        preds, consts = _lexicon(it["_premises_fol"])
        dm = _deterministic_map(it["_statement"], preds, consts)
        if dm:
            det += 1
            if g and _verify(it["_premises_fol"], dm, g):
                det_ok += 1
        if not args.no_llm:
            r = translate(it["id"], it["_statement"], it["_premises_fol"], gold=g, model=args.model)
            if r:
                llm_ok += 1
    if not args.no_llm:
        save_cache()
    print(f"folio_concl_fol: {len(items)} items | deterministic fired {det} (z3-verified {det_ok}) | "
          f"pipeline verified {llm_ok if not args.no_llm else 'n/a'}")
