#!/usr/bin/env python3
"""LogicNLI **structured** logical form → first-order logic (``fol.py`` syntax).

The original LogicNLI release (omnilabNLP/LogicNLI, ``LogicNLI_sim.zip``) ships, for
every block, a machine-precise logical form in ``*_logic.json`` alongside the natural
language in ``*_language.json``:

    facts:      {"i": [entity, attribute, "+"/"-", "fact i"]}
    rules:      {"i": {"p": {...}, "q": {...}, "type": "imp"|"equ", ...}}
                  side = {"fact": [[subj, attr, pol], ...], "conj": "none"|"and"|"or"}
                  subj = "all" (∀ person) | "exist" (∃ person) | a NAMED entity (grounded)
    statements: {"i": [entity, attr, pol, ...]}   # atomic hypothesis
    labels:     {"i": "entailment"|"contradiction"|"neutral"|"self_contradiction"}

This is DETERMINISTIC and unambiguous — it replaces the fragile NL→FOL regex parser
(``logicnli_fol.py``) and its whole bug class. We emit ``∀∃¬∧∨→↔`` strings that
``fol.py``/``fol_forward`` consume unchanged, then ground the quantifiers over the
finite set of named individuals (reusing ``logicnli_fol._ground``).

The corpus item source stays the logi_glue ``logicnli_dev.jsonl`` (for eval alignment);
this module attaches the structured FOL by the verified index-join:
    logi_glue row i  ==  block i//20, statement i%20   (matched 2000/2000).

    python logicnli_logic.py    # self-test: FOL coverage over dev blocks
"""
from __future__ import annotations

import json
from pathlib import Path

import logicnli_fol as _L   # reuse _pred / _const / _ground

HERE = Path(__file__).resolve().parent
_SIM = HERE.parent / "data" / "_raw" / "logicnli_sim" / "LogicNLI_sim"
VAR_P, VAR_Q = "x", "y"

_CACHE: dict = {}


def available(split: str = "dev") -> bool:
    return (_SIM / f"{split}_logic.json").exists() and (_SIM / f"{split}_language.json").exists()


def _load(split: str):
    if split not in _CACHE:
        logic = json.loads((_SIM / f"{split}_logic.json").read_text(encoding="utf-8"))
        lang = json.loads((_SIM / f"{split}_language.json").read_text(encoding="utf-8"))
        _CACHE[split] = (logic, lang)
    return _CACHE[split]


# ------------------------------------------------------------------ structured -> FOL
def _lit(subj: str, attr: str, pol: str, var: str) -> str | None:
    p = _L._pred(attr)
    if not p:
        return None
    arg = var if subj in ("all", "exist") else _L._const(subj)
    a = f"{p}({arg})"
    return f"¬{a}" if pol == "-" else a


def _side(side: dict, var: str):
    """Return (fol_str, subj_kind) for one rule side, or (None, None)."""
    facts = side.get("fact") or []
    if not facts:
        return None, None
    lits = [_lit(f[0], f[1], f[2], var) for f in facts]
    if any(l is None for l in lits):
        return None, None
    subj = facts[0][0]
    kind = subj if subj in ("all", "exist") else "named"
    if len(lits) == 1:
        return lits[0], kind
    op = "∧" if side.get("conj") == "and" else "∨"     # 'none' won't reach here (1 lit)
    return "(" + f" {op} ".join(lits) + ")", kind


def _wrap(fol_str: str, kind: str, var: str) -> str:
    if kind == "all":
        return f"(∀{var} {fol_str})"
    if kind == "exist":
        return f"(∃{var} {fol_str})"
    return fol_str                                       # named -> already grounded


def _rule_to_fol(rule: dict) -> str | None:
    op = "→" if rule.get("type") == "imp" else "↔"
    psub = (rule.get("p", {}).get("fact") or [[None]])[0][0]
    qsub = (rule.get("q", {}).get("fact") or [[None]])[0][0]
    # both sides quantify the SAME person (∀x (P(x) op Q(x))) — LogicNLI's "someone ... he ..."
    if psub == "all" and qsub == "all":
        p, _ = _side(rule["p"], VAR_P)
        q, _ = _side(rule["q"], VAR_P)
        return f"∀{VAR_P} ({p} {op} {q})" if p and q else None
    # otherwise hoist each quantified side independently over its own variable
    p, pk = _side(rule["p"], VAR_P)
    q, qk = _side(rule["q"], VAR_Q)
    if not p or not q:
        return None
    return f"({_wrap(p, pk, VAR_P)} {op} {_wrap(q, qk, VAR_Q)})"


def _fact_to_fol(fact) -> str | None:
    return _lit(fact[0], fact[1], fact[2], VAR_P)        # named subj -> grounded literal


def _stmt_to_fol(stmt) -> str | None:
    return _lit(stmt[0], stmt[1], stmt[2], VAR_P)


# ------------------------------------------------------------------ public
def fol_for(block_idx: int, stmt_idx: int, split: str = "dev"):
    """Return (premises_fol, conclusion_fol, nl_premises) for one (block, statement),
    or (None, None, None) on any failure. `premises_fol` is quantifier-free (grounded);
    `nl_premises` is index-aligned NL (facts then rules) for derivation rendering."""
    if not available(split):
        return None, None, None
    logic, lang = _load(split)
    rec = logic.get(str(block_idx))
    lrec = lang.get(str(block_idx))
    if rec is None or lrec is None:
        return None, None, None
    facts = rec.get("facts", {})
    rules = rec.get("rules", {})
    fact_fol, rule_fol, nl = [], [], []
    for i in sorted(facts, key=lambda k: int(k)):
        f = _fact_to_fol(facts[i])
        if f is None:
            return None, None, None
        fact_fol.append(f)
    for i in sorted(rules, key=lambda k: int(k)):
        r = _rule_to_fol(rules[i])
        if r is None:
            return None, None, None
        rule_fol.append(r)
    nl = [str(s).strip() for s in lrec.get("facts", [])] + \
         [str(s).strip() for s in lrec.get("rules", [])]
    stmts = rec.get("statements", {})
    stmt = stmts.get(str(stmt_idx))
    if stmt is None:
        return None, None, None
    concl = _stmt_to_fol(stmt)
    if concl is None:
        return None, None, None
    prem = fact_fol + rule_fol
    grounded = _L._ground(prem, concl)                   # ∀/∃ -> finite conj/disj
    if not grounded:
        return None, None, None
    g_prem, g_concl = grounded
    if len(g_prem) != len(nl):                           # keep NL alignment sound
        nl = nl[:len(g_prem)] + [None] * max(0, len(g_prem) - len(nl))
    return g_prem, g_concl, nl


# ------------------------------------------------------------------ self-test
if __name__ == "__main__":
    from collections import Counter
    if not available("dev"):
        print(f"[logicnli_logic] MISSING {_SIM} — download LogicNLI_sim.zip first")
        raise SystemExit(1)
    logic, lang = _load("dev")
    MAP = {"entailment": "True", "contradiction": "False", "neutral": "Unknown"}
    tot = ok = 0
    stats = Counter()
    for b in range(len(logic)):
        lrec = lang[str(b)]
        for s in range(len(lrec["labels"])):
            gold = MAP.get(lrec["labels"][s])
            if gold is None:                             # skip self_contradiction
                continue
            tot += 1
            prem, concl, nl = fol_for(b, s)
            if prem:
                ok += 1
                stats["fol_ok"] += 1
            else:
                stats["fol_fail"] += 1
    print(f"LogicNLI dev structured->FOL: {tot} non-self-contra statements | "
          f"FOL built {ok} ({100*ok/max(1,tot):.0f}%)  {dict(stats)}")
