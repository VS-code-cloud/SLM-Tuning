#!/usr/bin/env python3
"""Full-taxonomy LGMT metamorphic augmentation for TRAIN entailment rows, matching the
EVAL's transforms exactly (../critical-reasoning-eval/lgmt_full.py + fol.py).

The prior version used ad-hoc NL string edits that did NOT match the eval's transforms, so
the model stayed brittle on MR-C (conclusion rewrites) and MR-E (formula rewrites). This
version reuses the eval's own definitions:

  * MR-P (P1_reorder, P2_duplicate, P3_irrelevant, P4_fusion) and MR-C (C1/C2/C3) and
    MR-S (S1_const_rename) are DETERMINISTIC NL templates copied verbatim from the eval —
    no FOL, no gateway. Works for folio / logicnli / proverqa.
  * MR-E (the 10 symbolic equivalence rewrites) operate on real FOL from `data/lgmt_sources.json`
    (folio + logicnli) via `fol.py` (parse -> E_RULES rewrite -> ser), then the mutated formula
    is translated back to one NL sentence by the gateway (cached in data/lgmt_fol_nl_cache.json),
    exactly like the eval's `translate()`. One MR-E variant per source (a per-source-rotated rule
    order gives variety across the 10 rules).

Completions: reuse the source row's derivation VERBATIM (label-preserving transforms => the
derivation still commits to the same gold). No fixed "this reformulation preserves the logic"
lead-in (that risked teaching a shortcut). S1 also renames the entity inside the completion so
it stays coherent. => 0 grade-mismatch by construction (the committing sentence is untouched).

`gen_variants(row, src=None, call_agent=None)` -> list of new SFT rows. `src` is the FOL record
for `row` (from `load_fol_sources()`); pass `call_agent=slm_core.call_agent` to enable MR-E.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fol  # noqa: E402  (data_construction/fol.py — byte-identical to the eval's)

DATA = HERE.parent / "data"
_TRANS_CACHE = DATA / "lgmt_fol_nl_cache.json"

# MR ids -> category, matching lgmt_full.MR_SPECS (P5/S2 intentionally omitted here).
MR_CAT = {
    "P1_reorder": "MR-P", "P2_duplicate": "MR-P", "P3_irrelevant": "MR-P", "P4_fusion": "MR-P",
    "C1_conj_true": "MR-C", "C2_disj_false": "MR-C", "C3_double_neg": "MR-C",
    "S1_const_rename": "MR-S",
    **{name: "MR-E" for name in fol.E_RULES},
}

# verbatim from the eval (lgmt_full.py:282-283)
IRRELEVANT = "The integer seven is a prime number."
FRESH_ENTITY = "Zylvane"

# Rule tiers by how much the NL actually changes. Tier 0 = meaningful structural rewrites
# (implication<->disjunction, De Morgan, distributivity, quantifier lifting) — preferred, and
# rotated per-source for variety. Tier 1 = weak (reorder/rename, NL nearly identical). Tier 2 =
# the always-firing φ∧φ / φ∧⊤ / φ∧(τ∨¬τ) redundancy adds. We take the first rule (across tiers)
# that actually applies AND yields different NL, so trivial rules only fire as a fallback.
_E_TIERS = [
    ["E1.1_impl_elim", "E1.2_neg_norm", "E2.4_distrib", "E1.3_quant_lift"],
    ["E1.6_quant_order", "E1.5_alpha_rename", "E1.4_struct_norm"],
    ["E2.1_idempotence", "E2.2_taut_contra", "E2.3_identity"],
]

TRANS_PROMPT = (  # verbatim from lgmt_full.py:253-262
    "You are a strict first-order-logic-to-English translator. Convert the FOL formula "
    "into ONE natural English sentence stating exactly the same logical claim. Do NOT "
    "simplify, evaluate, resolve, or add information; preserve the exact structure.\n"
    "Symbols: ∀=for all, ∃=there exists, ¬=not, ∧=and, ∨=or, →=if...then, ↔=if and only "
    "if, ⊕=exactly one of (exclusive or), ⊤=a statement that is always true, ⊥=a "
    "statement that is always false. A predicate like Engaged(x) means 'x is engaged' or "
    "'x has property Engaged'; lowercase names like bonnie are individuals. Output ONLY "
    "the English sentence.\n\nFOL: {fol}"
)

# ----------------------------------------------------------------- FOL source lookup
_SRC = None
def load_fol_sources() -> dict:
    """base_id -> {family, premises, premises_fol, conclusion, conclusion_fol, gold} (folio+logicnli)."""
    global _SRC
    if _SRC is None:
        _SRC = {}
        p = DATA / "lgmt_sources.json"
        if p.exists():
            for r in json.loads(p.read_text(encoding="utf-8")):
                _SRC[r["id"]] = r
    return _SRC


def base_id(iid: str) -> str:
    """Strip global-dedup '#n' and any '-lgmt-*' suffix to recover the source id."""
    return iid.split("#", 1)[0].split("-lgmt-", 1)[0]


# ----------------------------------------------------------------- FOL->NL translation (cached)
_TRANS = None
def _cache() -> dict:
    global _TRANS
    if _TRANS is None:
        _TRANS = json.loads(_TRANS_CACHE.read_text(encoding="utf-8")) if _TRANS_CACHE.exists() else {}
    return _TRANS


def save_cache() -> None:
    if _TRANS is not None:
        _TRANS_CACHE.write_text(json.dumps(_TRANS, ensure_ascii=False, indent=0), encoding="utf-8")


def translate(formula_str: str, call_agent=None, model: str = "claude-haiku-4-5", timeout: int = 60):
    """FOL string -> one NL sentence (cached). Returns None on failure/drift (eval semantics)."""
    c = _cache()
    key = hashlib.md5((model + "||" + formula_str).encode()).hexdigest()
    if key in c:
        return c[key]
    if call_agent is None:
        return None
    out = call_agent(TRANS_PROMPT.format(fol=formula_str), model, timeout)
    if not out:
        return None
    out = out.strip().strip('"').split("\n")[-1].strip()
    if not out or any(sym in out for sym in "∀∃∧∨→↔⊕¬⊤⊥"):  # drift guard
        return None
    c[key] = out
    return out


# ----------------------------------------------------------------- NL prompt parsing
def _split_prompt(row):
    """-> (head_incl_marker, stimulus, question_tail) or None if not an FRQ entailment prompt."""
    p = row["prompt"]; mk = "Be specific and concise.\n\n"
    if mk not in p or "\n\nQuestion:" not in p:
        return None
    head, rest = p.split(mk, 1)
    stim, qtail = rest.split("\n\nQuestion:", 1)
    return head + mk, stim, qtail


def parse_src(row):
    sp = _split_prompt(row)
    if not sp:
        return None
    head, stim, qtail = sp
    fam = row["family"]
    if fam == "folio" and "\n\nConclusion:" in stim:
        pblock, concl = stim.split("\n\nConclusion:", 1)
        prems = [l[2:].strip() for l in pblock.splitlines() if l.startswith("- ")]
        concl = concl.strip(); concl_in = "stim"
    else:
        prems = [s.strip() for s in re.split(r"(?<=[.!?])\s+", stim.strip()) if s.strip()]
        m = re.search(r'statement "([^"]+)"', qtail)
        concl = m.group(1) if m else None; concl_in = "q"
    if len(prems) < 2 or not concl:
        return None
    return {"fam": fam, "stim": stim, "prems": prems, "concl": concl, "concl_in": concl_in}


def _rebuild_stim(fam, prems, concl):
    if fam == "folio":
        return "Premises:\n" + "\n".join(f"- {p}" for p in prems) + f"\n\nConclusion: {concl}"
    return " ".join(prems)


def _entity(premises, conclusion):  # eval's _entity (lgmt_full.py:287-293)
    for c in re.findall(r"\b([A-Z][a-z]{2,})\b", conclusion):
        if c in {"The", "If", "All", "Some", "Every", "No", "There", "Based", "It", "A"}:
            continue
        if re.search(r"\b" + re.escape(c) + r"\b", " ".join(premises)):
            return c
    return None


# ----------------------------------------------------------------- variant generation
def gen_variants(row, src=None, call_agent=None, enable_mre=True):
    it = parse_src(row)
    if not it:
        return []
    fam, P, C, concl_in = it["fam"], it["prems"], it["concl"], it["concl_in"]
    comp, gold = row["completion"], row["gold"]
    rows = []

    def _row(mr, prompt, completion):
        return {"id": f"{row['id']}-lgmt-{mr}", "family": fam, "mode": "frq",
                "task_type": row.get("task_type"), "difficulty": row.get("difficulty"),
                "prompt": prompt, "completion": completion, "gold": gold,
                "trace_source": (row.get("trace_source") or "") + "|lgmt",
                "is_lgmt": True, "lgmt_mr": mr}

    def emit_prems(mr, new_prems, completion=None):
        prompt = row["prompt"].replace(it["stim"], _rebuild_stim(fam, new_prems, C), 1)
        rows.append(_row(mr, prompt, completion if completion is not None else comp))

    def emit_concl(mr, new_concl, completion=None):
        if fam == "folio":
            prompt = row["prompt"].replace(it["stim"], _rebuild_stim(fam, P, new_concl), 1)
        else:
            prompt = row["prompt"].replace(f'statement "{C}"', f'statement "{new_concl}"', 1)
        rows.append(_row(mr, prompt, completion if completion is not None else comp))

    # One variant per (source, category); WHICH sub-MR varies by source hash so the category
    # spreads across all its members (else every MR-P row would be P1, every MR-C row C1).
    h = int(hashlib.sha1(row["id"].encode()).hexdigest()[:8], 16)

    # ---- MR-P (deterministic; eval templates) ----
    p_cands = [("P2_duplicate", P + [P[0]]), ("P3_irrelevant", P + [IRRELEVANT])]
    if len(P) >= 2:
        p_cands += [("P1_reorder", list(reversed(P))),
                    ("P4_fusion", [f"{P[0].rstrip('.')}. Moreover, {P[1]}"] + P[2:])]
    mr, np_ = p_cands[h % len(p_cands)]
    emit_prems(mr, np_)

    # ---- MR-C (deterministic; eval templates) ----
    c_cands = [("C1_conj_true", f"{C.rstrip('.')}, and two plus two equals four."),
               ("C2_disj_false", f"{C.rstrip('.')}, or five times five equals one."),
               ("C3_double_neg", f"It is not the case that the following statement is false: {C}")]
    mr, nc = c_cands[(h // 11) % len(c_cands)]
    emit_concl(mr, nc)

    # ---- MR-S1 constant rename (deterministic NL; rename in the completion too) ----
    e = _entity(P, C)
    if e:
        sub = lambda s: re.sub(r"\b" + re.escape(e) + r"\b", FRESH_ENTITY, s)
        prompt = row["prompt"].replace(it["stim"], _rebuild_stim(fam, [sub(p) for p in P], sub(C)), 1)
        if concl_in == "q":
            prompt = prompt.replace(f'statement "{C}"', f'statement "{sub(C)}"', 1)
        rows.append(_row("S1_const_rename", prompt, sub(comp)))

    # ---- MR-E (real FOL rewrite -> gateway translate). One per source; rotate rule order. ----
    if enable_mre and src and src.get("premises_fol"):
        PF, CF = src["premises_fol"], src.get("conclusion_fol")
        srcP, srcC = src.get("premises", []), src.get("conclusion")
        slots = [("prem", srcP[i], PF[i]) for i in range(min(len(PF), len(srcP)))]
        if CF and srcC:
            slots.append(("concl", srcC, CF))
        t0 = _E_TIERS[0][:]
        rot = int(hashlib.sha1(row["id"].encode()).hexdigest()[:6], 16) % len(t0)
        order = (t0[rot:] + t0[:rot]) + _E_TIERS[1] + _E_TIERS[2]
        done = False
        for name in order:
            if done:
                break
            rule = fol.E_RULES[name]
            for where, nl_orig, fstr in slots:
                if nl_orig not in row["prompt"]:
                    continue
                try:
                    ast = fol.parse(fstr)
                    new_ast, applied = rule(ast)
                except Exception:
                    continue
                if not applied or new_ast == ast:
                    continue
                nl = translate(fol.ser(new_ast), call_agent)
                if not nl or nl == nl_orig:
                    continue
                prompt = row["prompt"].replace(nl_orig, nl, 1)
                rows.append(_row(name, prompt, comp))
                done = True
                break

    return rows


if __name__ == "__main__":  # quick offline self-test (deterministic MRs only, no gateway)
    src = load_fol_sources()
    print("fol sources:", len(src), "| e.g.", next(iter(src)) if src else None)
    demo = {"id": "folio-t-1", "family": "folio", "mode": "frq", "gold": "True",
            "task_type": "entailment", "difficulty": None, "trace_source": "solver_path",
            "completion": "Since all cats are mammals and Tom is a cat, Tom is a mammal. So the statement is True.",
            "prompt": ("You are an expert in logic and argument analysis. Answer the following "
                       "open-ended question about the argument. Reason carefully and commit to the "
                       "single best answer - do not hedge by listing many possibilities. Be specific "
                       "and concise.\n\nPremises:\n- All cats are mammals.\n- Tom is a cat.\n\n"
                       "Conclusion: Tom is a mammal.\n\nQuestion: Based ONLY on the premises, is the "
                       "conclusion True, False, or Uncertain?\n\nExplain your reasoning in a few "
                       "sentences, then clearly state your final conclusion as exactly one of the "
                       "labels named in the question.")}
    vs = gen_variants(demo)
    print("deterministic variants:", [(v["lgmt_mr"], MR_CAT[v["lgmt_mr"]]) for v in vs])
    for v in vs:
        stim = v["prompt"].split("Be specific and concise.\n\n")[1].split("\n\nQuestion")[0]
        print(f"  {v['lgmt_mr']}: {stim[:140]!r}")
