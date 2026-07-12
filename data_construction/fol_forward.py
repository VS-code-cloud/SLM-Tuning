#!/usr/bin/env python3
"""Bounded forward-chaining reasoner for LogicNLI (with proof extraction).

LogicNLI's gold labels come from BOUNDED-DEPTH FORWARD-CHAINING, not global
satisfiability. Its blocks are deliberately near-paradoxical (a 4th
``self_contradiction`` label), so a z3 global-SAT check (``fol_solver.classify_path``)
declares the premises unsat and refuses to decide — even when the hypothesis is fixed
by a short *local* chain (e.g. the hypothesis directly negates a stated fact). This
module instead forward-chains from the facts through the rules, so it decides the
hypothesis LOCALLY and emits the actual derivation as a working path.

Operates on the GROUNDED (quantifier-free) propositional premises produced by
``logicnli_logic.fol_for`` (∀/∃ already expanded over the named individuals). A
grounded ``∀x (P(x)→Q(x))`` is a top-level conjunction of per-person implications, so
we flatten top-level ∧ into individual clauses before chaining.

    classify_forward(premises_fol, conclusion_fol, nl_premises, statement)
        -> (label|None, path_prose|None, detail)
    label: "True" (entailment) | "False" (contradiction) | "Unknown" (neutral) | None (drop)

Soundness: a forward-chained derivation of the hypothesis is a valid modus-ponens proof
from the premises regardless of the theory's global consistency; build_corpus keeps a
path ONLY when its label equals the gold, so kept paths are both sound and correct.
The path carries NO trailing label word (True/False/Uncertain/Unknown) — build_corpus
appends the single committing sentence, preserving the last-label-hit grade invariant.

    python fol_forward.py    # self-test: agreement with gold on LogicNLI dev
"""
from __future__ import annotations

import re

import fol

_MAXD = 8


# ------------------------------------------------------------------ AST helpers
def _lit_of(node):
    """(atom_str, truth) if node is a literal Pred / ¬Pred, else None."""
    t = type(node).__name__
    if t == "Pred":
        return (fol.ser(node), True)
    if t == "Not" and type(node.f).__name__ == "Pred":
        return (fol.ser(node.f), False)
    return None


def _flatten(node, cls):
    if type(node).__name__ == cls:
        return _flatten(node.a, cls) + _flatten(node.b, cls)
    return [node]


def _eval3(node, known):
    """3-valued truth of a boolean combo of literals under `known` (True/False/None)."""
    t = type(node).__name__
    if t == "Pred":
        return known.get(fol.ser(node), (None,))[0]
    if t == "Top":
        return True
    if t == "Bot":
        return False
    if t == "Not":
        v = _eval3(node.f, known)
        return None if v is None else (not v)
    if t == "And":
        a, b = _eval3(node.a, known), _eval3(node.b, known)
        if a is False or b is False:
            return False
        if a is True and b is True:
            return True
        return None
    if t == "Or":
        a, b = _eval3(node.a, known), _eval3(node.b, known)
        if a is True or b is True:
            return True
        if a is False and b is False:
            return False
        return None
    return None


def _cons_lits(node):
    """Literals of a conjunctive/atomic consequent → [(atom, truth)]; else []."""
    if type(node).__name__ == "And":
        return _cons_lits(node.a) + _cons_lits(node.b)
    l = _lit_of(node)
    return [l] if l else []


def _support(node, known):
    """The atoms actually responsible for the antecedent being true (minimal-ish):
    all conjuncts of an ∧, the satisfied disjunct of an ∨, the atom of a literal."""
    t = type(node).__name__
    if t == "Pred":
        a = fol.ser(node)
        return [a] if a in known else []
    if t == "Not":
        return _support(node.f, known)
    if t == "And":
        return _support(node.a, known) + _support(node.b, known)
    if t == "Or":
        for side in (node.a, node.b):
            if _eval3(side, known) is True:
                return _support(side, known)
        return _support(node.a, known) + _support(node.b, known)
    return []


# ------------------------------------------------------------------ rendering
def _prose(atom: str, truth: bool) -> str:
    m = re.match(r"([A-Za-z0-9_]+)\((.*)\)", atom)
    if not m:
        return atom
    subj = m.group(2).strip()
    subj = subj[:1].upper() + subj[1:]
    adj = m.group(1)
    adj = adj[:1].lower() + adj[1:]
    return f"{subj} is {'' if truth else 'not '}{adj}"


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def _render(hatom, hpos, label, statement, known):
    steps, seen = [], set()

    def build(atom, depth=0):
        if atom in seen or depth > 4 or atom not in known:
            return
        seen.add(atom)
        val, (src, supp) = known[atom]
        if src == "given":
            return
        for s in supp:
            build(s, depth + 1)
        steps.append((src, supp, atom, val))

    build(hatom)
    lines = []
    for src, supp, atom, v in steps[:5]:
        sup = "; ".join(_prose(s, known[s][0]) for s in supp if s in known) or "the premises"
        rule = str(src).rstrip(". ").strip() or "the rules"   # quote verbatim (keep case)
        lines.append(f"since {sup}, the rule \"{rule}\" gives that {_prose(atom, v)}")
    hval = known[hatom][0]
    if label == "True":
        tail = f"so \"{statement}\" is forced."
    else:
        tail = (f"the premises establish that {_prose(hatom, hval)}, which contradicts "
                f"\"{statement}\", so it cannot hold.")
    if lines:
        body = "; ".join(_cap(x) for x in lines)
        return f"{body} — {tail}" if label == "True" else f"{body}; {tail}"
    # decided directly from a stated fact (no rule chain)
    if label == "True":
        return f"The premises directly give that {_prose(hatom, hval)}, so \"{statement}\" is forced."
    return (f"The premises directly give that {_prose(hatom, hval)}, which contradicts "
            f"\"{statement}\", so it cannot hold.")


# ------------------------------------------------------------------ reasoner
def classify_forward(premises_fol, conclusion_fol, nl_premises=None,
                     statement="the statement", max_depth=_MAXD):
    try:
        prem = [fol.parse(p) for p in premises_fol if str(p).strip()]
        concl = fol.parse(conclusion_fol)
    except Exception as e:
        return None, None, f"parse: {type(e).__name__}"
    hl = _lit_of(concl)
    if hl is None:
        return None, None, "conclusion not a literal"
    hatom, hpos = hl
    statement = str(statement).rstrip(". ").strip() or "the statement"   # clean quoted prose

    known = {}                 # atom -> (bool value, (source_nl|"given", [support atoms]))
    conflict = [False]

    def force(atom, val, prov):
        if atom in known:
            if known[atom][0] != val:
                conflict[0] = True
            return False       # first-write-wins
        known[atom] = (val, prov)
        return True

    nlp = list(nl_premises) if nl_premises else []
    clauses = []               # (kind, node, nl)
    for i, node in enumerate(prem):
        nl = nlp[i] if i < len(nlp) else None
        for c in _flatten(node, "And"):
            l = _lit_of(c)
            if l:
                force(l[0], l[1], ("given", []))
            else:
                clauses.append((type(c).__name__, c, nl))

    for _ in range(max_depth):
        changed = False
        for kind, c, nl in clauses:
            if kind == "Implies":
                pairs = [(c.a, c.b)]
            elif kind == "Iff":
                pairs = [(c.a, c.b), (c.b, c.a)]
            else:
                continue
            for ante, cons in pairs:
                if _eval3(ante, known) is True:
                    supp = _support(ante, known)
                    for atom, truth in _cons_lits(cons):
                        changed = force(atom, truth, (nl, supp)) or changed
                # NB: no contrapositive (¬cons ⇒ ¬ante). LogicNLI's gold is generated by a
                # FORWARD modus-ponens closure; adding contrapositive over-derives on the
                # deliberately-inconsistent blocks and turns true-neutrals into spurious
                # True/False (measured: contrapositive drops neutral agreement 100%→46% and
                # overall 96%→75%). Biconditionals still fire in BOTH directions (that is the
                # meaning of ↔, and the generator treats equ rules symmetrically).
        if not changed:
            break

    v = known.get(hatom)
    if v is None:
        # Neutral: forward reasoning reached neither the hypothesis nor its negation. This is
        # a LOCAL judgement — global inconsistency elsewhere in the block does NOT make THIS
        # statement paradoxical (paradox = the statement itself derivable both ways, and those
        # rows carry the self_contradiction label, already excluded upstream). The label==gold
        # safety net drops any case where under-derivation gave a false neutral.
        path = (f"Reasoning forward from the given facts through the rules derives neither "
                f"\"{statement}\" nor its negation, so it does not settle it.")
        return "Unknown", path, "neutral (not derivable)"
    val = v[0]
    label = "True" if (val == hpos) else "False"
    return label, _render(hatom, hpos, label, statement, known), "derived"


def support_indices(premises_fol, conclusion_fol, max_depth=_MAXD):
    """Forward-chain and return (label, set_of_premise_indices) — the indices (into
    ``premises_fol``) of the facts + rules that actually participate in deriving the
    conclusion atom. Empty set when the conclusion is not derived (neutral). Used to seed
    the classically-consistent per-conclusion premise pruning (logicnli_prune)."""
    try:
        prem = [fol.parse(p) for p in premises_fol if str(p).strip()]
        concl = fol.parse(conclusion_fol)
    except Exception:
        return None, set()
    hl = _lit_of(concl)
    if hl is None:
        return None, set()
    hatom, hpos = hl
    known = {}                       # atom -> (val, premise_idx, [support_atoms])

    def force(atom, val, pidx, supp):
        if atom in known:
            return False
        known[atom] = (val, pidx, supp)
        return True

    clauses = []                     # (kind, node, premise_idx)
    for i, node in enumerate(prem):
        for c in _flatten(node, "And"):
            l = _lit_of(c)
            if l:
                force(l[0], l[1], i, [])
            else:
                clauses.append((type(c).__name__, c, i))
    for _ in range(max_depth):
        changed = False
        for kind, c, i in clauses:
            if kind == "Implies":
                pairs = [(c.a, c.b)]
            elif kind == "Iff":
                pairs = [(c.a, c.b), (c.b, c.a)]
            else:
                continue
            for ante, cons in pairs:
                if _eval3(ante, known) is True:
                    supp = _support(ante, known)
                    for atom, truth in _cons_lits(cons):
                        changed = force(atom, truth, i, supp) or changed
        if not changed:
            break
    if hatom not in known:
        return "Unknown", set()
    label = "True" if known[hatom][0] == hpos else "False"
    idxs, seen, stack = set(), set(), [hatom]
    while stack:                     # backtrace provenance to the premises used
        a = stack.pop()
        if a in seen or a not in known:
            continue
        seen.add(a)
        _, pidx, supp = known[a]
        if pidx is not None:
            idxs.add(pidx)
        stack.extend(supp)
    return label, idxs


# ------------------------------------------------------------------ self-test
if __name__ == "__main__":
    from collections import Counter
    import logicnli_logic as LL
    if not LL.available("dev"):
        print("[fol_forward] LogicNLI_sim not available; run download first")
        raise SystemExit(1)
    logic, lang = LL._load("dev")
    MAP = {"entailment": "True", "contradiction": "False", "neutral": "Unknown"}
    tot = Counter()
    agree = Counter()
    for b in range(len(logic)):
        for s in range(len(lang[str(b)]["labels"])):
            gold = MAP.get(lang[str(b)]["labels"][s])
            if gold is None:
                continue
            prem, concl, nl = LL.fol_for(b, s)
            if not prem:
                tot["_fol_fail"] += 1
                continue
            stmt_nl = lang[str(b)]["statements"][s].rstrip(". ")
            lab, path, detail = classify_forward(prem, concl, nl, stmt_nl)
            tot[gold] += 1
            if lab == gold:
                agree[gold] += 1
    n = sum(tot[k] for k in ("True", "False", "Unknown"))
    a = sum(agree.values())
    print(f"LogicNLI dev forward-chain: {n} statements | solver==gold {a} ({100*a/max(1,n):.0f}%)")
    for k in ("True", "False", "Unknown"):
        print(f"    {k:8} {agree[k]:4}/{tot[k]:4}  ({100*agree[k]/max(1,tot[k]):.0f}%)")
