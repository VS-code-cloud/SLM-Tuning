#!/usr/bin/env python3
"""Symbolic FOL entailment solver for FOLIO (validation split ships conclusion-FOL).

Parses FOLIO's first-order-logic premises + conclusion with the eval's ``fol.py``
parser, translates the AST to Z3 over a single uninterpreted sort, and classifies
the entailment three ways:

    True       iff  premises ⊨ conclusion        (premises ∧ ¬conclusion  UNSAT)
    False      iff  premises ⊨ ¬conclusion       (premises ∧  conclusion  UNSAT)
    Uncertain  otherwise (both satisfiable — a countermodel exists each way)

Returns ``None`` when Z3 can't decide (quantifier "unknown") or parsing fails, so
callers fall back. This gives FOLIO a *derivation* it doesn't ship — so a cheap
model (Haiku) can reword the verified verdict into prose instead of Opus reasoning
from scratch.

Why not ProverQA? ProverQA already ships a natural-language reasoning chain
(Prover9-generated) — it needs a reword, not a solver. This module is FOLIO-only.
"""
from __future__ import annotations

import itertools

import fol   # the eval's FOL parser (copied into colab/)

try:
    import z3
except Exception:  # pragma: no cover
    z3 = None

_U = None
_CHECK_TIMEOUT_MS = 8000


def _sort():
    global _U
    if _U is None:
        _U = z3.DeclareSort("U")
    return _U


def _to_z3(node, env, preds, consts):
    """Translate a fol.py AST node to a Z3 expression. env: bound var -> z3 const."""
    t = type(node).__name__
    if t == "Top":
        return z3.BoolVal(True)
    if t == "Bot":
        return z3.BoolVal(False)
    if t == "Pred":
        args = []
        for a in node.args:
            if a in env:
                args.append(env[a])                      # bound variable
            else:
                if a not in consts:
                    consts[a] = z3.Const(a, _sort())     # free constant
                args.append(consts[a])
        key = (node.name, len(node.args))
        if key not in preds:
            dom = [_sort()] * len(node.args)
            preds[key] = z3.Function(node.name, *dom, z3.BoolSort())
        return preds[key](*args) if node.args else preds[key]()
    if t == "Not":
        return z3.Not(_to_z3(node.f, env, preds, consts))
    if t == "And":
        return z3.And(_to_z3(node.a, env, preds, consts), _to_z3(node.b, env, preds, consts))
    if t == "Or":
        return z3.Or(_to_z3(node.a, env, preds, consts), _to_z3(node.b, env, preds, consts))
    if t == "Implies":
        return z3.Implies(_to_z3(node.a, env, preds, consts), _to_z3(node.b, env, preds, consts))
    if t == "Iff":
        return _to_z3(node.a, env, preds, consts) == _to_z3(node.b, env, preds, consts)
    if t == "Xor":
        return z3.Xor(_to_z3(node.a, env, preds, consts), _to_z3(node.b, env, preds, consts))
    if t == "Quant":
        v = z3.Const(node.var, _sort())
        env2 = dict(env); env2[node.var] = v
        body = _to_z3(node.body, env2, preds, consts)
        return z3.ForAll([v], body) if node.kind == "∀" else z3.Exists([v], body)
    raise ValueError(f"unknown node {t}")


def _check(assumptions):
    s = z3.Solver()
    s.set("timeout", _CHECK_TIMEOUT_MS)
    for a in assumptions:
        s.add(a)
    return s.check()   # sat / unsat / unknown


def classify(premises_fol: list[str], conclusion_fol: str) -> tuple[str | None, str]:
    """Return (label, detail). label in {True,False,Uncertain} or None if undecided."""
    if z3 is None:
        return None, "z3 unavailable"
    preds, consts = {}, {}
    try:
        prem = [_to_z3(fol.parse(p), {}, preds, consts) for p in premises_fol if p.strip()]
        concl = _to_z3(fol.parse(conclusion_fol), {}, preds, consts)
    except Exception as e:
        return None, f"parse: {type(e).__name__}"
    if not prem:
        return None, "no premises"
    # premises must be self-consistent, else entailment is degenerate
    if _check(prem) != z3.sat:
        return None, "premises unsat/unknown"
    r_true = _check(prem + [z3.Not(concl)])     # unsat => entailed (True)
    r_false = _check(prem + [concl])            # unsat => contradicted (False)
    if r_true == z3.unknown or r_false == z3.unknown:
        return None, "z3 unknown (quantifiers)"
    if r_true == z3.unsat and r_false == z3.sat:
        return "True", "premises entail the conclusion"
    if r_false == z3.unsat and r_true == z3.sat:
        return "False", "premises entail the negation of the conclusion"
    if r_true == z3.sat and r_false == z3.sat:
        return "Uncertain", "a model satisfies the premises with the conclusion either way"
    return None, "both unsat (contradictory)"


# --------------------------------------------------------------------------
# Working-path variant: same 3-way verdict, PLUS a real derivation to verbalize —
# the specific premises that force True/False (z3 unsat-core), and for Uncertain a
# concrete COUNTERMODEL WITNESS (a ground atom the premises leave open, whose value
# flips the statement). Used by build_corpus to replace FOLIO/LogicNLI templates
# with genuine derivations (like ProverQA's shipped chain). Returns
#   (label, path_prose | None, detail)
# path_prose carries NO trailing label word, so build_corpus's committing final
# sentence remains the only label token (grade invariant preserved).
# --------------------------------------------------------------------------
def _check_keep(assumptions):
    """Like _check but returns (result, solver) so model()/unsat_core() are reachable."""
    s = z3.Solver()
    s.set("timeout", _CHECK_TIMEOUT_MS)
    for a in assumptions:
        s.add(a)
    return s.check(), s


def _unsat_core_idxs(prem, extra):
    """Indices of the premises in the minimal unsat core of (prem ∧ extra)."""
    s = z3.Solver()
    s.set("timeout", _CHECK_TIMEOUT_MS)
    track = {}
    for i, p in enumerate(prem):
        b = z3.Bool(f"__p{i}")
        track[b.decl().name()] = i
        s.assert_and_track(p, b)
    s.add(extra)                                   # the (negated) conclusion, untracked
    if s.check() != z3.unsat:
        return list(range(len(prem)))              # fallback: cite all
    idxs = sorted({track[c.decl().name()] for c in s.unsat_core()
                   if c.decl().name() in track})
    return idxs or list(range(len(prem)))


def _render_entail(idxs, prem_nl, statement, refute):
    # prem_nl is index-aligned with the solver's premise list (may hold None entries when
    # the NL/FOL arrays didn't line up) -> cite only the real, correctly-mapped premises.
    used = [prem_nl[i].rstrip(". ").strip() for i in idxs
            if i < len(prem_nl) and prem_nl[i]]
    lead = (f"From the premises that {'; '.join(used[:6])}" if used
            else "Combining the relevant premises")
    if refute:
        return (f"{lead}, assuming \"{statement}\" leads to a contradiction with them, "
                f"so it cannot hold.")
    return (f"{lead}, the conclusion \"{statement}\" is forced — denying it contradicts "
            f"the premises.")


def _differing_atom(m1, m2, preds, consts):
    """A ground atom Pred(consts...) whose truth differs between the two models."""
    cs = list(consts.values())
    for (name, arity), fn in preds.items():
        combos = [()] if arity == 0 else itertools.product(cs, repeat=arity)
        for combo in combos:
            atom = fn(*combo) if combo else fn()
            try:
                v1 = z3.is_true(m1.eval(atom, model_completion=True))
                v2 = z3.is_true(m2.eval(atom, model_completion=True))
            except Exception:
                continue
            if v1 != v2:
                inner = ", ".join(str(c) for c in combo)
                return f"{name}({inner})" if combo else name
    return None


def classify_path(premises_fol, conclusion_fol, nl_premises=None, statement="the statement"):
    """(label, path_prose|None, detail). label in {True,False,Uncertain} or None."""
    if z3 is None:
        return None, None, "z3 unavailable"
    preds, consts = {}, {}
    # Pair each FOL premise with its NL by ORIGINAL index and filter empties TOGETHER, so
    # unsat-core indices map to the right NL premise. Only trust the NL mapping when the two
    # arrays are the same length (14/1208 FOLIO rows have mismatched arrays -> cite generically).
    aligned = nl_premises if (nl_premises and len(nl_premises) == len(premises_fol)) else None
    prem, prem_nl = [], []
    try:
        for i, p in enumerate(premises_fol):
            if not str(p).strip():
                continue
            prem.append(_to_z3(fol.parse(p), {}, preds, consts))
            prem_nl.append(aligned[i] if aligned else None)
        concl = _to_z3(fol.parse(conclusion_fol), {}, preds, consts)
    except Exception as e:
        return None, None, f"parse: {type(e).__name__}"
    if not prem:
        return None, None, "no premises"
    if _check(prem) != z3.sat:                     # premises must be consistent
        return None, None, "premises unsat/unknown"
    r_true = _check(prem + [z3.Not(concl)])
    r_false = _check(prem + [concl])
    if r_true == z3.unknown or r_false == z3.unknown:
        return None, None, "z3 unknown (quantifiers)"
    if r_true == z3.unsat and r_false == z3.sat:
        core = _unsat_core_idxs(prem, z3.Not(concl))
        return "True", _render_entail(core, prem_nl, statement, refute=False), "entailed"
    if r_false == z3.unsat and r_true == z3.sat:
        core = _unsat_core_idxs(prem, concl)
        return "False", _render_entail(core, prem_nl, statement, refute=True), "contradicted"
    if r_true == z3.sat and r_false == z3.sat:
        _, s1 = _check_keep(prem + [concl])          # model where statement holds
        _, s2 = _check_keep(prem + [z3.Not(concl)])  # model where it fails
        diff = None
        try:
            diff = _differing_atom(s1.model(), s2.model(), preds, consts)
        except Exception:
            diff = None
        if diff:
            # accurate claim: {diff} is genuinely unconstrained (both models satisfy the
            # premises and disagree on it). Do NOT claim it *alone* decides the statement —
            # the two models differ on many atoms; that overclaims a causal link.
            path = (f"Testing both directions, the premises are consistent with \"{statement}\" "
                    f"and equally with its negation — one model of the premises makes it hold and "
                    f"another makes it fail (they leave {diff} open) — so nothing given settles it.")
        else:
            path = (f"Testing both directions, one reading of the premises makes \"{statement}\" "
                    f"hold and another makes it fail, so the premises do not settle it.")
        return "Uncertain", path, "under-determined (countermodel witness)"
    return None, None, "both unsat (contradictory)"


if __name__ == "__main__":   # quick self-test on FOLIO-val
    import json
    from pathlib import Path
    from collections import Counter
    raw = Path(__file__).resolve().parent.parent / "data" / "_raw" / "folio_validation.jsonl"
    items = [json.loads(l) for l in raw.read_text(encoding="utf-8").splitlines() if l.strip()]
    stats = Counter()
    agree = 0
    solved = 0
    CANON = {"true": "True", "false": "False", "uncertain": "Uncertain", "unknown": "Uncertain"}
    for it in items:
        gold = CANON.get(str(it.get("label", "")).strip().lower())
        lab, _ = classify(it.get("premises-FOL", []), it.get("conclusion-FOL", "") or "")
        stats["solved" if lab else "unsolved"] += 1
        if lab:
            solved += 1
            if lab == gold:
                agree += 1
    n = len(items)
    print(f"FOLIO-val: {n} items | solved {solved} ({100*solved/n:.0f}%) | "
          f"solver==gold on solved: {agree}/{solved} ({100*agree/max(1,solved):.0f}%)")
