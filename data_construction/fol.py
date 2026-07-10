#!/usr/bin/env python3
"""A small, tolerant first-order-logic engine for FOLIO-style FOL.

Just enough to support the LGMT metamorphic rewrites: tokenize -> recursive-descent
parse -> AST -> equivalence-preserving rewrites -> serialize (fully parenthesized).

FOLIO FOL uses: ∀ ∃ ¬ ∧ ∨ → ↔ ⊕(xor), predicates `Name(args)`, lowercase
constants, single-letter variables. The data has occasional quirks (unbalanced
parens, ⟷ for ↔, stray '.'), so the parser raises ParseError on anything it can't
cleanly handle and callers treat that formula as "MR-inapplicable".
"""
from __future__ import annotations

from dataclasses import dataclass


# ------------------------------------------------------------------ AST
@dataclass(frozen=True)
class Pred:
    name: str
    args: tuple  # tuple[str, ...]

@dataclass(frozen=True)
class Top: pass          # ⊤ / True

@dataclass(frozen=True)
class Bot: pass          # ⊥ / False

@dataclass(frozen=True)
class Not:
    f: object

@dataclass(frozen=True)
class And:
    a: object
    b: object

@dataclass(frozen=True)
class Or:
    a: object
    b: object

@dataclass(frozen=True)
class Implies:
    a: object
    b: object

@dataclass(frozen=True)
class Iff:
    a: object
    b: object

@dataclass(frozen=True)
class Xor:
    a: object
    b: object

@dataclass(frozen=True)
class Quant:
    kind: str            # '∀' or '∃'
    var: str
    body: object


class ParseError(Exception):
    pass


# ------------------------------------------------------------------ tokenizer
_OPS = {"∀", "∃", "¬", "∧", "∨", "→", "↔", "⟷", "⊕", "(", ")", ","}
_NORM = {"⟷": "↔", "->": "→", "<->": "↔"}


def tokenize(s: str) -> list:
    s = s.replace("⟷", "↔").replace("’", "'").strip()
    toks, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c.isspace() or c == ".":
            i += 1; continue
        if c in _OPS:
            toks.append(c); i += 1; continue
        if c.isalnum() or c == "_" or c == "'":
            j = i
            while j < n and (s[j].isalnum() or s[j] in "_'"):
                j += 1
            toks.append(s[i:j]); i = j; continue
        raise ParseError(f"bad char {c!r} in {s!r}")
    return toks


# ------------------------------------------------------------------ parser
class _P:
    def __init__(self, toks):
        self.t = toks
        self.i = 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def eat(self, tok=None):
        c = self.peek()
        if c is None:
            raise ParseError("unexpected end")
        if tok is not None and c != tok:
            raise ParseError(f"expected {tok!r}, got {c!r}")
        self.i += 1
        return c

    def parse(self):
        f = self.p_iff()
        if self.peek() is not None:
            raise ParseError(f"trailing {self.t[self.i:]!r}")
        return f

    def p_iff(self):
        a = self.p_impl()
        while self.peek() == "↔":
            self.eat(); a = Iff(a, self.p_impl())
        return a

    def p_impl(self):
        a = self.p_xor()
        if self.peek() == "→":
            self.eat(); return Implies(a, self.p_impl())  # right-assoc
        return a

    def p_xor(self):
        a = self.p_or()
        while self.peek() == "⊕":
            self.eat(); a = Xor(a, self.p_or())
        return a

    def p_or(self):
        a = self.p_and()
        while self.peek() == "∨":
            self.eat(); a = Or(a, self.p_and())
        return a

    def p_and(self):
        a = self.p_unary()
        while self.peek() == "∧":
            self.eat(); a = And(a, self.p_unary())
        return a

    def p_unary(self):
        c = self.peek()
        if c == "¬":
            self.eat(); return Not(self.p_unary())
        if c in ("∀", "∃"):
            self.eat(); var = self.eat()
            if var in _OPS:
                raise ParseError("bad quantifier var")
            return Quant(c, var, self.p_unary())
        if c == "(":
            self.eat("("); f = self.p_iff(); self.eat(")"); return f
        return self.p_atom()

    def p_atom(self):
        name = self.eat()
        if name in _OPS:
            raise ParseError(f"expected atom, got {name!r}")
        if name in ("True", "⊤"):
            return Top()
        if name in ("False", "⊥"):
            return Bot()
        if self.peek() == "(":
            self.eat("(")
            args = []
            while self.peek() != ")":
                args.append(self.eat())
                if self.peek() == ",":
                    self.eat(",")
                elif self.peek() != ")":
                    raise ParseError("bad arg list")
            self.eat(")")
            return Pred(name, tuple(args))
        return Pred(name, ())


def parse(s: str):
    return _P(tokenize(s)).parse()


# ------------------------------------------------------------------ serializer
def ser(f) -> str:
    if isinstance(f, Top):
        return "⊤"
    if isinstance(f, Bot):
        return "⊥"
    if isinstance(f, Pred):
        return f.name + ("(" + ", ".join(f.args) + ")" if f.args else "")
    if isinstance(f, Not):
        return "¬" + ser(f.f)
    if isinstance(f, Quant):
        return f"{f.kind}{f.var} ({ser(f.body)})"
    ops = {And: "∧", Or: "∨", Implies: "→", Iff: "↔", Xor: "⊕"}
    for cls, op in ops.items():
        if isinstance(f, cls):
            return f"({ser(f.a)} {op} {ser(f.b)})"
    raise ValueError(f"cannot serialize {f!r}")


# ------------------------------------------------------------------ traversal helpers
def walk(f):
    yield f
    for child in _children(f):
        yield from walk(child)


def _children(f):
    if isinstance(f, Not):
        return [f.f]
    if isinstance(f, Quant):
        return [f.body]
    if isinstance(f, (And, Or, Implies, Iff, Xor)):
        return [f.a, f.b]
    return []


def predicates(f) -> set:
    return {n.name for n in walk(f) if isinstance(n, Pred) and n.args}


def constants(f) -> set:
    """Constant symbols = lowercase multi-char arg names that are not bound vars."""
    bound = {n.var for n in walk(f) if isinstance(n, Quant)}
    out = set()
    for n in walk(f):
        if isinstance(n, Pred):
            for a in n.args:
                if a not in bound and (len(a) > 1 or not a.isalpha()):
                    out.add(a)
    return out


def rename_symbol(f, old: str, new: str, kind: str):
    """kind='pred' renames predicate names; kind='const' renames arg constants."""
    def go(x):
        if isinstance(x, Pred):
            name = new if (kind == "pred" and x.name == old) else x.name
            args = tuple(new if (kind == "const" and a == old) else a for a in x.args)
            return Pred(name, args)
        if isinstance(x, Not):
            return Not(go(x.f))
        if isinstance(x, Quant):
            return Quant(x.kind, x.var, go(x.body))
        for cls in (And, Or, Implies, Iff, Xor):
            if isinstance(x, cls):
                return cls(go(x.a), go(x.b))
        return x
    return go(f)


# ------------------------------------------------------------------ rewrite rules
# Each rule: f -> (new_f, applied?). Applied once at the first matching redex
# (top-down), preserving logical equivalence.
def _rewrite_once(f, rule):
    """Apply `rule` to the first node (top-down) where it fires."""
    new = rule(f)
    if new is not None:
        return new, True
    # recurse into children, rebuild
    if isinstance(f, Not):
        c, done = _rewrite_once(f.f, rule)
        return (Not(c), True) if done else (f, False)
    if isinstance(f, Quant):
        c, done = _rewrite_once(f.body, rule)
        return (Quant(f.kind, f.var, c), True) if done else (f, False)
    for cls in (And, Or, Implies, Iff, Xor):
        if isinstance(f, cls):
            a, done = _rewrite_once(f.a, rule)
            if done:
                return cls(a, f.b), True
            b, done = _rewrite_once(f.b, rule)
            return (cls(f.a, b), True) if done else (f, False)
    return f, False


def e1_1(f):  # implication / biconditional elimination
    def r(x):
        if isinstance(x, Implies):
            return Or(Not(x.a), x.b)
        if isinstance(x, Iff):
            return And(Or(Not(x.a), x.b), Or(Not(x.b), x.a))
        return None
    return _rewrite_once(f, r)


def e1_2(f):  # negation normalization (push ¬ inward / drop double neg)
    def r(x):
        if isinstance(x, Not):
            g = x.f
            if isinstance(g, Not):
                return g.f
            if isinstance(g, And):
                return Or(Not(g.a), Not(g.b))
            if isinstance(g, Or):
                return And(Not(g.a), Not(g.b))
            if isinstance(g, Quant):
                flip = "∃" if g.kind == "∀" else "∀"
                return Quant(flip, g.var, Not(g.body))
        return None
    return _rewrite_once(f, r)


def e1_3(f):  # quantifier lifting: (Qx φ) ∘ ψ ⤳ Qx (φ ∘ ψ)  when x ∉ FV(ψ)
    def r(x):
        for cls in (And, Or):
            if isinstance(x, cls):
                if isinstance(x.a, Quant) and x.a.var not in _freevars(x.b):
                    return Quant(x.a.kind, x.a.var, cls(x.a.body, x.b))
                if isinstance(x.b, Quant) and x.b.var not in _freevars(x.a):
                    return Quant(x.b.kind, x.b.var, cls(x.a, x.b.body))
        return None
    return _rewrite_once(f, r)


def e1_4(f):  # structural normalization: sort operands of ∧/∨ by serialized order
    def r(x):
        for cls in (And, Or):
            if isinstance(x, cls):
                if ser(x.a) > ser(x.b):
                    return cls(x.b, x.a)
        return None
    return _rewrite_once(f, r)


_TAUT = Pred("SelfEvidentlyTrue", ())  # a fresh nullary proposition for T ∨ ¬T


def e2_1(f):  # idempotence:  φ ⤳ φ ∧ φ  (equivalence-preserving introduction)
    return And(f, f), True


def e2_2(f):  # tautology conjunction:  φ ⤳ φ ∧ (τ ∨ ¬τ)  (τ a fresh proposition)
    return And(f, Or(_TAUT, Not(_TAUT))), True


def e2_3(f):  # identity:  φ ⤳ φ ∧ ⊤
    return And(f, Top()), True


def e2_4(f):  # distributivity: φ∧(ψ∨θ) ⤳ (φ∧ψ)∨(φ∧θ)
    def r(x):
        if isinstance(x, And) and isinstance(x.b, Or):
            return Or(And(x.a, x.b.a), And(x.a, x.b.b))
        if isinstance(x, Or) and isinstance(x.b, And):
            return And(Or(x.a, x.b.a), Or(x.a, x.b.b))
        return None
    return _rewrite_once(f, r)


def e1_5(f):  # α-renaming: rename the first bound variable to a fresh one
    used = {a for n in walk(f) if isinstance(n, Pred) for a in n.args}
    used |= {n.var for n in walk(f) if isinstance(n, Quant)}
    fresh = next(v for v in ("u", "v", "w", "z", "y", "t") if v not in used)
    def r(x):
        if isinstance(x, Quant):
            return Quant(x.kind, fresh, _subst_var(x.body, x.var, fresh))
        return None
    return _rewrite_once(f, r)


def e1_6(f):  # canonical quantifier ordering: swap adjacent same-kind quantifiers
    def r(x):
        if isinstance(x, Quant) and isinstance(x.body, Quant) and x.kind == x.body.kind \
           and x.var != x.body.var:
            return Quant(x.body.kind, x.body.var, Quant(x.kind, x.var, x.body.body))
        return None
    return _rewrite_once(f, r)


def _subst_var(f, old: str, new: str):
    def go(x):
        if isinstance(x, Pred):
            return Pred(x.name, tuple(new if a == old else a for a in x.args))
        if isinstance(x, Not):
            return Not(go(x.f))
        if isinstance(x, Quant):
            if x.var == old:      # shadowed: stop
                return x
            return Quant(x.kind, x.var, go(x.body))
        for cls in (And, Or, Implies, Iff, Xor):
            if isinstance(x, cls):
                return cls(go(x.a), go(x.b))
        return x
    return go(f)


def _freevars(f) -> set:
    bound, free = set(), set()
    def go(x, b):
        if isinstance(x, Pred):
            for a in x.args:
                if a not in b:
                    free.add(a)
        elif isinstance(x, Not):
            go(x.f, b)
        elif isinstance(x, Quant):
            go(x.body, b | {x.var})
        elif isinstance(x, (And, Or, Implies, Iff, Xor)):
            go(x.a, b); go(x.b, b)
    go(f, set())
    return free


E_RULES = {
    "E1.1_impl_elim": e1_1, "E1.2_neg_norm": e1_2, "E1.3_quant_lift": e1_3,
    "E1.4_struct_norm": e1_4, "E1.5_alpha_rename": e1_5, "E1.6_quant_order": e1_6,
    "E2.1_idempotence": e2_1, "E2.2_taut_contra": e2_2,
    "E2.3_identity": e2_3, "E2.4_distrib": e2_4,
}
