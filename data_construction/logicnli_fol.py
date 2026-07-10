#!/usr/bin/env python3
"""LogicNLI natural-language → first-order logic (``fol.py`` syntax).

KEPT SEPARATE from FOLIO's shipped FOL (per plan): LogicNLI ships no logical form,
but it is a **synthetic, closed-grammar** dataset — propositional logic over UNARY
adjective predicates with a single monadic variable (no binary relations, no nested
quantifiers, inclusive-or, no xor). We parse each puzzle's ~12 facts + ~12 rules and
the atomic hypothesis into FOL strings that ``fol_solver`` / ``fol.py`` consume
unchanged, so the same z3 working-path machinery gives LogicNLI real derivations
(and countermodel witnesses for the neutral class) exactly like FOLIO.

This is BEST-EFFORT: anything not cleanly matched returns None and the caller falls
back to the template. Correctness is guaranteed downstream by solver-verification
(build_corpus only keeps a solver path when the label it derives equals the gold),
so an imperfect parse costs coverage, never correctness.

    python logicnli_fol.py    # self-test: parse coverage + solver-vs-gold on dev
"""
from __future__ import annotations

import re

import fol   # reuse the FOL AST/parser for finite-domain grounding

VAR = "x"

# words that look like a capitalized proper name at sentence start but are NOT people
RESERVED = {"someone", "somebody", "anyone", "anybody", "everyone", "everybody", "no",
            "nobody", "people", "person", "he", "she", "they", "it", "all", "there",
            "who", "if", "then", "as", "when", "whenever", "both", "either", "neither",
            "every", "everything", "each", "some", "something", "any", "anything",
            "nothing", "none", "most", "when", "while"}


# ------------------------------------------------------------------ finite-domain grounding
# LogicNLI's semantics is over the FINITE set of named individuals, so ∀/∃ must be
# expanded over those names (∀ → conjunction, ∃ → disjunction). Grounding also makes
# the problem quantifier-free, which z3 decides instantly (no "unknown").
def _ground_ast(x, names):
    if isinstance(x, fol.Quant):
        insts = [_ground_ast(fol._subst_var(x.body, x.var, n), names) for n in names]
        if not insts:
            return fol.Top() if x.kind == "∀" else fol.Bot()
        acc = insts[0]
        for it in insts[1:]:
            acc = fol.And(acc, it) if x.kind == "∀" else fol.Or(acc, it)
        return acc
    if isinstance(x, fol.Not):
        return fol.Not(_ground_ast(x.f, names))
    for cls in (fol.And, fol.Or, fol.Implies, fol.Iff, fol.Xor):
        if isinstance(x, cls):
            return cls(_ground_ast(x.a, names), _ground_ast(x.b, names))
    return x   # Pred / Top / Bot


def _ground(fol_strs, concl_str):
    """Ground all quantifiers in the premises + conclusion over the union of named
    constants. Returns (grounded_premise_strs, grounded_conclusion_str) or None."""
    try:
        asts = [fol.parse(s) for s in fol_strs]
        c_ast = fol.parse(concl_str)
    except Exception:
        return None
    names = set()
    for a in asts + [c_ast]:
        names |= fol.constants(a)
    names = sorted(names)
    if not names:
        return None
    g_prem = [fol.ser(_ground_ast(a, names)) for a in asts]
    g_concl = fol.ser(_ground_ast(c_ast, names))
    return g_prem, g_concl


# ------------------------------------------------------------------ atoms
def _pred(adj: str) -> str | None:
    a = re.sub(r"[^a-z0-9_]", "", adj.strip().lower().replace("-", "_").replace(" ", "_"))
    return (a[:1].upper() + a[1:]) if a else None


def _const(name: str) -> str:
    c = re.sub(r"[^a-z0-9]", "", name.strip().lower())
    return c if len(c) > 1 else c + "_"        # keep >1 char so fol.constants() sees a const


def _atom(adj: str, arg: str, neg: bool) -> str | None:
    p = _pred(adj)
    if not p:
        return None
    a = f"{p}({arg})"
    return f"¬{a}" if neg else a


# ------------------------------------------------------------------ boolean sides
def _one_var(lit: str) -> str | None:
    """A single adjective literal about the monadic variable x (with optional 'not')."""
    lit = lit.strip().strip(",.")
    # case-insensitive: callers (the ↔/equivalence branches) pass original-cased strings
    # like "Someone is ...", "He is ..." — a case-sensitive strip would leave the prefix
    # in and collapse the whole clause into one bogus predicate.
    lit = re.sub(r"^(he|she|they|it)\s+is\s+", "", lit, flags=re.I)
    lit = re.sub(r"^(being|is|always)\s+", "", lit, flags=re.I).strip()
    neg = False
    m = re.match(r"not\s+(.+)$", lit, re.I)
    if m:
        neg, lit = True, m.group(1).strip()
    return _atom(lit, VAR, neg)


def _side_var(clause: str) -> str | None:
    """Parse a clause about 'someone/he/who' into FOL over x (≤3 literals)."""
    c = clause.strip().strip(",.")
    c = re.sub(r"^(someone|somebody|people|person|a person|he|she|they|who|anyone|everyone)\s+",
               "", c, flags=re.I)
    c = re.sub(r"^(is|being|who is|always)\s+", "", c, flags=re.I).strip()
    c = re.sub(r"\balways\b", "", c, flags=re.I).strip()
    m = re.match(r"neither\s+(.+?)\s+nor\s+(.+)$", c, re.I)
    if m:
        a, b = _atom(m.group(1), VAR, True), _atom(m.group(2), VAR, True)
        return f"({a} ∧ {b})" if a and b else None
    if " or " in c and " and " in c:      # mixed connectives -> ambiguous precedence: bail
        return None
    if " or " in c and " and " not in c:
        # strip a leading "either"/"eithor" (LogicNLI ships the misspelling "eithor";
        # left in place it swallows the following "not" into a free predicate).
        parts = re.split(r"\s+or\s+", re.sub(r"^(?:either|eithor)\s+", "", c, flags=re.I))
        lits = [_one_var(p) for p in parts]
        return "(" + " ∨ ".join(lits) + ")" if all(lits) else None
    if " and " in c and " or " not in c:
        parts = re.split(r"\s+and\s+", re.sub(r"^both\s+", "", c, flags=re.I))
        lits = [_one_var(p) for p in parts]
        return "(" + " ∧ ".join(lits) + ")" if all(lits) else None
    return _one_var(c)


_GLIT = re.compile(r"([A-Z][A-Za-z'\-]*)\s+(?:is|being)\s+(not\s+)?([a-z][A-Za-z\-]*)")


def _side_grounded(clause: str) -> str | None:
    """Parse 'NAME is/being [not] ADJ (and/or ...)' over specific named individuals."""
    c = clause.strip().strip(",.")
    lits = [(name, neg, adj) for name, neg, adj in _GLIT.findall(c)
            if name.lower() not in RESERVED]     # exclude pronoun/quantifier "names"
    if not lits:
        return None
    atoms = [_atom(adj, _const(name), bool(neg)) for name, neg, adj in lits]
    if any(a is None for a in atoms):
        return None
    if len(atoms) == 1:
        return atoms[0]
    if " or " in c and " and " in c:      # mixed connectives -> ambiguous precedence: bail
        return None
    op = "∨" if " or " in c else "∧"
    out = atoms[0]
    for a in atoms[1:]:
        out = f"({out} {op} {a})"
    return out


# ------------------------------------------------------------------ rules
def _forall(ante_fol: str, cons_fol: str) -> str:
    return f"∀{VAR} ({ante_fol} → {cons_fol})"


def _rule_to_fol(rule: str) -> str | None:  # noqa: C901  (a flat template dispatcher)
    r = rule.strip().strip(".").strip()
    low = r.lower()

    # --- biconditionals (3 spellings) ---
    if low.endswith(", and vice versa") or low.endswith(" and vice versa"):
        core = re.sub(r",?\s+and vice versa$", "", r, flags=re.I)
        m = re.match(r"if someone is (.+?),?\s+then (?:he|she|they|it) is (.+)$", core, re.I)
        if m:
            a, b = _side_var(m.group(1)), _side_var(m.group(2))
            return f"∀{VAR} ({a} ↔ {b})" if a and b else None
        m = re.match(r"if (.+?),?\s+then (.+)$", core, re.I)   # grounded biconditional
        if m:
            a, b = _side_grounded(m.group(1)), _side_grounded(m.group(2))
            return f"({a} ↔ {b})" if a and b else None
        return None      # "vice versa" present but unrecognized -> bail (never one-way implies)
    if " if and only if " in low:
        lhs, rhs = re.split(r"\s+if and only if\s+", r, maxsplit=1, flags=re.I)
        # ∀ form: "Someone is [not] X if and only if he is Y" — the RHS must be a PRONOUN
        # back-reference to the same person. Require an actual pronoun (he/she/they/it),
        # NOT a bare "is": "Someone is happy if and only if Bob is sad" names a different
        # individual on the RHS and must NOT be forced onto the ∀ path (that produced a
        # junk predicate like Bob_is_sad(x)); it falls through to the grounded reading.
        mv = re.match(r"(?:someone|anyone|people|a person|everyone)\s+(?:is\s+)?(.+)$", lhs, re.I)
        if mv and re.search(r"\b(he|she|they|it)\b", rhs, re.I) and not _side_grounded(rhs):
            a, b = _side_var(lhs), _side_var(rhs)
            return f"∀{VAR} ({a} ↔ {b})" if a and b else None
        a, b = _side_grounded(lhs), _side_grounded(rhs)
        return f"({a} ↔ {b})" if a and b else None
    m = re.match(r"(.+?)\s+is equivalent to\s+(.+)$", r, re.I)
    if m:
        lhs, rhs = m.group(1), m.group(2)
        # ∀ form only when the RHS is about the same (unnamed) person; a named RHS
        # (e.g. "... is equivalent to Bob being sad") is grounded, not universal.
        if re.match(r"^(someone|anyone|people|a person|everyone)\b", lhs, re.I) \
                and not _side_grounded(rhs):
            a, b = _side_var(lhs), _side_var(rhs)
            return f"∀{VAR} ({a} ↔ {b})" if a and b else None
        a, b = _side_grounded(lhs), _side_grounded(rhs)
        return f"({a} ↔ {b})" if a and b else None

    # --- reversed implication: "It can be concluded that C once knowing that A" ---
    m = re.match(r"it can be concluded that (.+?) once knowing that (.+)$", r, re.I)
    if m:
        c, a = _side_grounded(m.group(1)), _side_grounded(m.group(2))
        return f"({a} → {c})" if a and c else None

    # --- "<A> implies that <C>" ---
    m = re.match(r"(.+?)\s+implies that\s+(.+)$", r, re.I)
    if m:
        a, c = _side_grounded(m.group(1)), _side_grounded(m.group(2))
        return f"({a} → {c})" if a and c else None

    # --- universal (people): "If someone [who] is A, then he is C" (∀; consequent about
    #     the same person). "As long as someone is A, he is C". "Someone who is A is C".
    m = re.match(r"(?:if|as long as) someone (?:who )?is (.+?),?\s+then (?:he|she|they|it) is (.+)$", r, re.I) \
        or re.match(r"as long as someone is (.+?),\s+(?:he|she|they|it) is (.+)$", r, re.I)
    if m:
        a, c = _side_var(m.group(1)), _side_var(m.group(2))
        return _forall(a, c) if a and c else None
    m = re.match(r"someone who is (.+?) is (?:always )?(.+)$", r, re.I)
    if m:
        a, c = _side_var(m.group(1)), _side_var(m.group(2))
        return _forall(a, c) if a and c else None
    m = re.match(r"all (.+?) people are (.+)$", r, re.I)
    if m:
        a, c = _side_var(m.group(1)), _side_var(m.group(2))
        return _forall(a, c) if a and c else None

    # --- existential / global antecedent, grounded consequent ---
    #     "If someone/there is someone who is A, then <grounded C>" -> (∃x A) → C
    m = re.match(r"if (?:there is )?(?:someone|at least one people?|a person|anybody|somebody) who is (.+?),?\s+then (.+)$", r, re.I)
    if m:
        a, c = _side_var(m.group(1)), _side_grounded(m.group(2))
        return f"((∃{VAR} {a}) → {c})" if a and c else None
    #     "If all people are A, then <grounded C>" / "If everyone is A, then C" -> (∀x A) → C
    m = re.match(r"if (?:all people are|everyone is) (.+?),?\s+then (.+)$", r, re.I)
    if m:
        a, c = _side_var(m.group(1)), _side_grounded(m.group(2))
        return f"((∀{VAR} {a}) → {c})" if a and c else None
    #     "If there is nobody who is not A, then C"  ≡  ∀x A(x) → C  (double-negation
    #     rephrasing of "everybody is A"). The empty-slot artifact "…nobody who is not,
    #     then…" has no adjective, so this won't match -> falls through to template.
    m = re.match(r"if there is nobody who is not (.+?),?\s+then (.+)$", r, re.I)
    if m:
        a, c = _side_var(m.group(1)), _side_grounded(m.group(2))
        return f"((∀{VAR} {a}) → {c})" if a and c else None

    # --- grounded implication: "If <named A>, then <named C>" ---
    m = re.match(r"if (.+?),?\s+then (.+)$", r, re.I)
    if m and _side_grounded(m.group(1)):
        a, c = _side_grounded(m.group(1)), _side_grounded(m.group(2))
        return f"({a} → {c})" if a and c else None

    return None


# ------------------------------------------------------------------ facts / statement
_FACT = re.compile(r"^([A-Z][A-Za-z'\-]*)\s+is\s+(not\s+)?([a-z][A-Za-z\-]*)$")


def _fact_to_fol(s: str) -> str | None:
    m = _FACT.match(s.strip().rstrip("."))
    if not m or m.group(1).lower() in RESERVED:      # "Someone is X" is a rule, not a fact
        return None
    return _atom(m.group(3), _const(m.group(1)), bool(m.group(2)))


def parse_statement(hyp: str) -> str | None:
    """Hypothesis is always atomic: NAME is [not] ADJ."""
    return _fact_to_fol(hyp)


# ------------------------------------------------------------------ context splitter
_BLOCK_CACHE: dict = {}


def _split_context(context: str) -> tuple[str, str] | None:
    """(facts_rules_block, hypothesis) from a LogicNLI `context` string."""
    c = re.sub(r"^\s*context:\s*", "", context, flags=re.I)
    idx = c.lower().rfind("statement:")
    if idx < 0:
        return None
    block, hyp = c[:idx].strip(), c[idx + len("statement:"):].strip()
    block = re.sub(r"\.([A-Z])", r". \1", block)       # unglue joined sentences
    return block, hyp


def _sentences(block: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[a-z\)])\.\s+", block) if s.strip()]


def context_to_fol(context: str):
    """Return (premises_fol: list[str], conclusion_fol: str) or (None, None) on failure.
    Parses each sentence as a fact then a rule; a sentence that parses as NEITHER makes
    the whole puzzle fail (so we never solve on a partial premise set)."""
    sp = _split_context(context)
    if not sp:
        return None, None
    block, hyp = sp
    concl = parse_statement(hyp)
    if not concl:
        return None, None
    key = block
    if key in _BLOCK_CACHE:
        prem = _BLOCK_CACHE[key]
    else:
        prem = []
        ok = True
        for s in _sentences(block):
            f = _fact_to_fol(s) or _rule_to_fol(s)
            if f is None:
                ok = False
                break
            prem.append(f)
        prem = prem if ok else None
        _BLOCK_CACHE[key] = prem
    if not prem:
        return None, None
    grounded = _ground(prem, concl)          # finite-domain: expand ∀/∃ over named individuals
    if not grounded:
        return None, None
    return grounded[0], grounded[1]


# ------------------------------------------------------------------ self-test
if __name__ == "__main__":
    import json
    from collections import Counter
    from pathlib import Path
    try:
        import fol_solver
    except Exception:
        fol_solver = None
    raw = Path(__file__).resolve().parent.parent / "data" / "_raw" / "logicnli_dev.jsonl"
    rows = [json.loads(l) for l in raw.read_text(encoding="utf-8").splitlines() if l.strip()]
    MAP = {"entailment": "True", "contradiction": "False", "neutral": "Unknown"}
    parsed = solved = agree = total = 0
    stats = Counter()
    for d in rows:
        gold = MAP.get(d.get("answer_text", ""))
        if gold is None:                       # skip self_contradiction (dropped in corpus)
            continue
        total += 1
        prem, concl = context_to_fol(d.get("context", ""))
        if not prem:
            stats["parse_fail"] += 1
            continue
        parsed += 1
        if fol_solver:
            lab, _ = fol_solver.classify(prem, concl)
            if lab:
                solved += 1
                agree += int(lab == gold)
                stats["agree" if lab == gold else "disagree"] += 1
            else:
                stats["unsolved"] += 1
    print(f"LogicNLI dev (3-way): {total} rows | parsed {parsed} ({100*parsed/total:.0f}%) | "
          f"solved {solved} | solver==gold {agree}/{solved} "
          f"({100*agree/max(1,solved):.0f}%)  {dict(stats)}")
    import os
    if os.environ.get("LNLI_DEBUG"):
        fails = Counter()
        seen_blocks = set()
        for d in rows:
            if MAP.get(d.get("answer_text", "")) is None:
                continue
            sp = _split_context(d.get("context", ""))
            if not sp:
                continue
            block = sp[0]
            if block in seen_blocks:
                continue
            seen_blocks.add(block)
            for s in _sentences(block):
                if _fact_to_fol(s) is None and _rule_to_fol(s) is None:
                    fails[" ".join(s.split()[:4]).lower()] += 1
        print(f"\n[debug] distinct blocks={len(seen_blocks)}; top unparsed sentence shapes:")
        for shape, c in fails.most_common(25):
            print(f"  {c:3}  {shape}")
