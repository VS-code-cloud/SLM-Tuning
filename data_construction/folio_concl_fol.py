#!/usr/bin/env python3
"""Cheap NL-conclusion → FOL for FOLIO *train* (it ships premises-FOL but NO
conclusion-FOL, unlike the validation split). We translate the one-sentence NL
conclusion into a single FOL formula CONSTRAINED to that row's own predicate/constant
lexicon (extracted from its premises-FOL), via a cheap gateway model (Haiku). Results
are cached to disk so it is a one-time cost; re-runs are free.

This is only a *candidate* — build_corpus runs the z3 solver on (premises-FOL, this
translated conclusion-FOL) and keeps the derivation ONLY if it decides and agrees with
the gold label (solver-verification). So a wrong translation is dropped (template
fallback), never poisoning the data. Gated on gateway creds: with none, returns None
(FOLIO-train stays on templates; FOLIO-val + LogicNLI are unaffected/offline).

    python folio_concl_fol.py --limit 30      # translate + cache a sample, report
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import threading
from pathlib import Path

# this module lives in colab/data_construction/; slm_core.py and data/ are one level up
HERE = Path(__file__).resolve().parent
_COLAB = HERE.parent
if str(_COLAB) not in sys.path:
    sys.path.insert(0, str(_COLAB))

import fol
try:
    import slm_core as S
except Exception:
    S = None

CACHE = _COLAB / "data" / "_raw" / "folio_train_concl_fol.json"
_LOCK = threading.Lock()


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


def _load_cache() -> dict:
    """Load the translation cache. On a corrupt/partial file (e.g. a kill mid-write on
    a pre-atomic-write build), fall back to the .bak sidecar rather than silently
    discarding every entry (which would force a full, costly re-translation)."""
    for path in (CACHE, CACHE.with_suffix(CACHE.suffix + ".bak")):
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return {k: v for k, v in data.items() if v}   # drop stale None entries
            except Exception:
                print(f"[folio_concl_fol] WARNING: cache at {path} is unreadable; "
                      "trying backup / starting empty")
                continue
    return {}


_CACHE = _load_cache()


def _prompt(conclusion_nl, preds, consts):
    return ("Translate the English conclusion into ONE first-order-logic formula.\n"
            f"Use ONLY these predicate symbols: {preds}\n"
            f"Use ONLY these constant symbols: {consts}\n"
            "Variables may be lowercase x, y, z. Operators allowed: ∀ ∃ ¬ ∧ ∨ → ↔ (and ⊕). "
            "Match predicate names and arity EXACTLY as given (some are binary, e.g. "
            "Love(x, music)). Output ONLY the formula on a single line, no prose, no code fence.\n\n"
            f"Conclusion: {conclusion_nl}")


def translate(key, conclusion_nl, premises_fol, model="claude-haiku-4-5", timeout=60):
    """Return a FOL string for the conclusion (cached), or None if unavailable/failed."""
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
    out = None
    if S is not None:
        preds, consts = _lexicon(premises_fol)
        if preds:
            r = S.call_agent(_prompt(conclusion_nl, preds, consts), model, timeout, max_tokens=160)
            if r:
                cand = r.strip().splitlines()[0].strip().strip("`").strip()
                try:                       # keep only if it parses as FOL
                    fol.parse(cand)
                    out = cand
                except Exception:
                    out = None
    # Only cache a SUCCESSFUL translation. Caching None would permanently poison the
    # cache: a transient/credless/parse failure would be returned verbatim forever and
    # prewarm's "id not in _CACHE" filter would never retry it. Leaving failures uncached
    # lets a later run (creds present, gateway healthy) translate the row.
    if out is not None:
        with _LOCK:
            _CACHE[key] = out
    return out


def save_cache():
    """Persist the cache atomically: write a temp file then os.replace() it into place,
    keeping the previous good copy as a .bak. write_text() truncates the target on open,
    so a kill mid-write used to leave a 0-byte/partial JSON that _load_cache silently
    discarded (forcing a full re-translation). os.replace is atomic on POSIX."""
    with _LOCK:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(_CACHE, ensure_ascii=False, indent=0)
        tmp = CACHE.with_suffix(CACHE.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        if CACHE.exists():                       # keep last good copy as backup
            try:
                os.replace(CACHE, CACHE.with_suffix(CACHE.suffix + ".bak"))
            except Exception:
                pass
        os.replace(tmp, CACHE)


def prewarm(items, model="claude-haiku-4-5", concurrency=8):
    """Translate+cache conclusion-FOL for a list of FOLIO-train items concurrently.
    Each item needs keys: id, _statement (NL conclusion), _premises_fol."""
    todo = [it for it in items if it.get("id") not in _CACHE and it.get("_premises_fol")]
    if not todo or S is None:
        return 0
    def _one(it):
        return translate(it["id"], it.get("_statement", ""), it.get("_premises_fol"))
    done = 0
    with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
        for _ in ex.map(_one, todo):
            done += 1
    save_cache()
    return done


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--model", default="claude-haiku-4-5")
    args = ap.parse_args()
    raw = _COLAB / "data" / "_raw" / "folio_train.jsonl"
    rows = [json.loads(l) for l in raw.read_text(encoding="utf-8").splitlines() if l.strip()]
    items, n = [], 0
    for i, d in enumerate(rows):
        pf = d.get("premises-FOL") or []
        if not pf:
            continue
        items.append({"id": f"folio-t-{d.get('example_id', i)}",
                      "_statement": d.get("conclusion", ""), "_premises_fol": pf})
        n += 1
        if n >= args.limit:
            break
    got = prewarm(items, args.model)
    ok = sum(1 for it in items if _CACHE.get(it["id"]))
    print(f"folio_concl_fol: attempted {len(items)} | translated+parsed {ok} | new calls {got}")
    for it in items[:3]:
        print(f"  {it['_statement'][:60]!r} -> {_CACHE.get(it['id'])}")
