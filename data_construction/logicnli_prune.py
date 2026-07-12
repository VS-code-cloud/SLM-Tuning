#!/usr/bin/env python3
"""Per-conclusion classically-consistent premise pruning for LogicNLI.

LogicNLI blocks are inconsistent by construction (every block has 5 paradox statements
⇒ the 12 facts + 12 rules are jointly UNSAT under classical FOL). That makes the raw
blocks unusable for a classical entailment oracle (z3) and for LGMT-style metamorphic
testing, which both require a satisfiable premise set Γ. This module rebuilds each
NON-paradox statement into a classically-consistent ⟨Γ, q⟩:

  1. Per block, compute a MAXIMAL CONSISTENT SUBSET (MSS) of the 24 premises once
     (drops only the minimal conflicting rules; keeps everything else — "keep the
     irrelevant premises too").
  2. Per statement, verify with z3 that ⟨MSS, q⟩ gives the gold verdict (True/False
     entailed, or Unknown left open). If the MSS dropped a rule this particular
     conclusion needed, fall back to the RELEVANT premises the forward-chaining
     derivation used (``fol_forward.support_indices``) and verify those.
  3. Anything that still can't be made consistent-and-gold-matching is dropped (never
     emitted inconsistent).

z3 expressions are parsed once per block (shared preds/consts) and reused, so the whole
dev set prunes in seconds.

    python logicnli_prune.py     # self-test: coverage + kept-premise stats on dev
"""
from __future__ import annotations

import fol
import fol_solver
import fol_forward as FF
import logicnli_logic as LL

try:
    import z3
except Exception:                                   # pragma: no cover
    z3 = None

_NEU = {"Uncertain", "Unknown"}
_BLK: dict = {}                                     # (split, block) -> parsed block cache


def _equiv(lab, gold) -> bool:
    return lab is not None and (lab == gold or (lab in _NEU and gold in _NEU))


def _is_fact(fol_str: str) -> bool:
    return not any(ch in fol_str for ch in "∀∃∧∨→↔")   # ground literal, no connectives


def _z3_classify(prem_exprs, concl_expr, timeout=4000):
    """Classical z3 verdict for ⟨prem, concl⟩ over pre-parsed exprs; None if Γ is UNSAT."""
    s = z3.Solver(); s.set("timeout", timeout)
    for e in prem_exprs:
        s.add(e)
    if s.check() != z3.sat:
        return None                                 # inconsistent premises
    s.push(); s.add(z3.Not(concl_expr)); rt = s.check(); s.pop()
    s.push(); s.add(concl_expr); rf = s.check(); s.pop()
    if rt == z3.unsat and rf == z3.sat:
        return "True"
    if rf == z3.unsat and rt == z3.sat:
        return "False"
    if rt == z3.sat and rf == z3.sat:
        return "Uncertain"
    return None


def _block(block_idx: int, split: str):
    key = (split, block_idx)
    if key in _BLK:
        return _BLK[key]
    prem, _, nl = LL.fol_for(block_idx, 0, split)       # premises identical across a block
    if not prem:
        _BLK[key] = None; return None
    preds, consts = {}, {}
    try:
        exprs = [fol_solver._to_z3(fol.parse(p), {}, preds, consts) for p in prem]
    except Exception:
        _BLK[key] = None; return None
    # maximal consistent subset: add premises (facts first) keeping the set satisfiable
    order = sorted(range(len(prem)), key=lambda i: (0 if _is_fact(prem[i]) else 1, i))
    kept = []
    for i in order:
        s = z3.Solver(); s.set("timeout", 4000)
        for j in kept + [i]:
            s.add(exprs[j])
        if s.check() == z3.sat:
            kept.append(i)
    info = {"prem": prem, "nl": nl, "exprs": exprs, "preds": preds, "consts": consts,
            "mss": set(kept)}
    _BLK[key] = info
    return info


def prune_for(block_idx: int, stmt_idx: int, gold: str, split: str = "dev"):
    """Return (kept_idxs, premises_fol_kept, conclusion_fol, nl_kept) — a classically
    consistent, z3-verified, gold-preserving premise subset — or None if it can't be
    made consistent-and-gold-matching."""
    if z3 is None:
        return None
    info = _block(block_idx, split)
    if not info:
        return None
    prem, nl, exprs = info["prem"], info["nl"], info["exprs"]
    _, concl_fol, _ = LL.fol_for(block_idx, stmt_idx, split)
    if not concl_fol:
        return None
    try:
        concl_expr = fol_solver._to_z3(fol.parse(concl_fol), {}, info["preds"], info["consts"])
    except Exception:
        return None

    # candidate 1: the block's maximal consistent subset (keeps the most premises)
    mss = sorted(info["mss"])
    if _equiv(_z3_classify([exprs[i] for i in mss], concl_expr), gold):
        keep = mss
    else:
        # candidate 2: the premises the forward proof actually used (relevant only)
        if gold in ("True", "False"):
            lab, support = FF.support_indices(prem, concl_fol)
            if not _equiv(lab, gold) or not support:
                return None
            keep = sorted(support)
        else:                                        # neutral: the ground facts
            keep = sorted(i for i in range(len(prem)) if _is_fact(prem[i]))
        if not _equiv(_z3_classify([exprs[i] for i in keep], concl_expr), gold):
            return None
        # grow the relevant seed with any premise that preserves consistency + gold
        keep = set(keep)
        for i in range(len(prem)):
            if i in keep:
                continue
            trial = sorted(keep | {i})
            if _equiv(_z3_classify([exprs[j] for j in trial], concl_expr), gold):
                keep.add(i)
        keep = sorted(keep)
    return keep, [prem[i] for i in keep], concl_fol, [nl[i] for i in keep]


# ------------------------------------------------------------------ self-test
if __name__ == "__main__":
    import time
    from collections import Counter
    if not LL.available("dev"):
        print("[logicnli_prune] LogicNLI_sim not available"); raise SystemExit(1)
    logic, lang = LL._load("dev")
    MAP = {"entailment": "True", "contradiction": "False", "neutral": "Unknown"}
    tot = Counter(); ok = Counter(); kept_all = []
    t0 = time.time()
    for b in range(len(logic)):
        for s in range(len(lang[str(b)]["labels"])):
            gold = MAP.get(lang[str(b)]["labels"][s])
            if gold is None:
                continue
            tot[gold] += 1
            r = prune_for(b, s, gold)
            if r is None:
                continue
            ok[gold] += 1
            kept_all.append(len(r[0]))
    n = sum(tot.values()); a = sum(ok.values())
    print(f"LogicNLI dev pruning: {a}/{n} statements -> consistent gold-preserving Γ "
          f"({100*a/max(1,n):.0f}%) in {time.time()-t0:.0f}s")
    for k in ("True", "False", "Unknown"):
        print(f"    {k:8} {ok[k]:4}/{tot[k]:4}")
    if kept_all:
        print(f"    kept premises/stmt: avg {sum(kept_all)/len(kept_all):.1f} of 24")
