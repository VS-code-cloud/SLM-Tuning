#!/usr/bin/env python3
"""Build the ~10k critical-reasoning SFT corpus (Day-3 plan).

Reads the raw datasets under ``data/_raw/`` and produces, with the **explanations
generated now** (not during training):

  * ``data/sft_train.jsonl`` — training rows {id, family, mode, prompt, completion,
    gold, ...}. Every ``completion`` is a **house-style** trace built by the SINGLE
    verbalizer here (``verbalize``): ProverQA uses its shipped prover-derived chain
    restyled; FOLIO/LogicNLI use a grounded entailment/neutral template; the MCQ
    families cite the credited option. One format for all sources.
  * ``data/eval_items.json`` — the held-out slice of EVERY family (full fields) so
    the notebook can generate + deterministically grade.
  * ``data/corpus_meta.json`` — provenance + exact counts.

Two regimes, never merged: entailment core (folio/logicnli/proverqa) -> FRQ label;
argument satellite (lsat_lr/arct/logiqa) -> MCQ. LGMT structural follow-ups augment
the entailment core in TRAIN only (built from train sources -> no leakage).

Usage:
    python build_corpus.py                 # build with default caps
    python build_corpus.py --eval-frac 0.2 --seed 0
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

# this module lives in colab/data_construction/; slm_core.py and data/ are one level up
HERE = Path(__file__).resolve().parent
_COLAB = HERE.parent
if str(_COLAB) not in sys.path:
    sys.path.insert(0, str(_COLAB))

import slm_core as S
try:
    import fol_solver            # z3 working-path solver (True/False core, Uncertain witness)
except Exception:
    fol_solver = None
try:
    import logicnli_fol          # separate LogicNLI NL->FOL parser (best-effort fallback)
except Exception:
    logicnli_fol = None
try:
    import logicnli_logic         # structured LogicNLI _logic.json -> FOL (primary, exact)
except Exception:
    logicnli_logic = None
try:
    import fol_forward            # bounded forward-chaining reasoner (seeds the prune)
except Exception:
    fol_forward = None
try:
    import logicnli_prune         # per-conclusion classically-consistent premise pruning
except Exception:
    logicnli_prune = None
try:
    import folio_concl_fol       # FOLIO-train conclusion NL->FOL (cached; gateway-gated)
except Exception:
    folio_concl_fol = None

RAW = _COLAB / "data" / "_raw"
OUT = _COLAB / "data"

# ---- budget caps (targets; the script takes min(target, available)) --------
# Plan 1 (rebalance): equalize families, trim ProverQA's over-reliance, add FOLIO
# validation, and split LGMT evenly across the 3 entailment families with varied MRs.
CAPS = {
    "proverqa": {"medium": 500, "hard": 500},  # ~1000, drop 'easy' (96.7% ceiling, teaches nothing); all depth
    "folio": 1100,             # train + validation
    "logicnli": 1500,          # all usable 3-way (neutral-rich)
    "reclor": 1500,            # of ~1902 arg-analysis
    "arct": 1300,
    "logiqa": 1400,
}
LGMT_TARGET = 1200             # ~400 per entailment family, MRs varied (P3 capped)
RECLOR_PW = b"for_non-commercial_research_purpose_only"


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace("−", "-")).strip()


def _norm_key(s: str) -> str:
    """Normalized passage key for cross-dataset dedup (whitespace + case folded)."""
    return norm_ws(s).lower()


_RECLOR_KEYS = None


def _reclor_keys() -> set:
    """Normalized ReClor passages (train+val+test). LogiQA and ReClor share LSAT source
    items, so any LogiQA passage duplicated here is dropped in parse_logiqa (ReClor owns
    it) — a shared passage then never straddles the train/eval split. Cached."""
    global _RECLOR_KEYS
    if _RECLOR_KEYS is None:
        keys = set()
        try:
            zf = zipfile.ZipFile(RAW / "reclor_data.zip")
            for sp in ("train.json", "val.json", "test.json"):
                for ex in json.loads(zf.read(sp, pwd=RECLOR_PW)):
                    c = ex.get("context")
                    if c:
                        keys.add(_norm_key(c))
        except Exception:
            pass
        _RECLOR_KEYS = keys
    return _RECLOR_KEYS


def _rank(iid: str, seed: int = 0) -> float:
    """Deterministic [0,1) shuffle key (no RNG so builds are reproducible)."""
    h = hashlib.sha1(f"shuf:{seed}:{iid}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def cap(items: list[dict], n: int, seed: int = 0) -> list[dict]:
    return sorted(items, key=lambda it: _rank(it["id"], seed))[:n]


# ==========================================================================
# Source parsers -> normalized item dicts
# ==========================================================================
def reclor_task_type(question: str) -> str | None:
    q = question.lower()
    if "flaw" in q or "vulnerable to criticism" in q or "questionable" in q \
            or "error in reasoning" in q or "reasoning is most" in q:
        return "flaw"
    if "assum" in q or "depends on" in q or "presupposes" in q or "relies on" in q:
        return "assumption"
    if "weaken" in q or "undermin" in q or "casts the most doubt" in q \
            or "calls into question" in q or "damage" in q:
        return "weaken"
    return None  # keep ONLY flaw / assumption / weaken (arg-analysis of the gap)


def parse_reclor() -> list[dict]:
    zf = zipfile.ZipFile(RAW / "reclor_data.zip")
    out = []
    for split in ("train.json", "val.json"):
        for ex in json.loads(zf.read(split, pwd=RECLOR_PW)):
            tt = reclor_task_type(ex.get("question", ""))
            if tt is None:
                continue
            ctx = norm_ws(ex.get("context", ""))
            ans = ex.get("answers", [])
            label = ex.get("label")
            if not ctx or label is None or not isinstance(label, int) or label >= len(ans):
                continue
            out.append({
                "id": f"reclor-{tt}-{ex.get('id_string', len(out))}",
                # ReClor reuses one passage across several questions (distinct id_strings);
                # group by the passage so a passage never straddles the train/eval split.
                "group_id": f"reclor-ctx-{hashlib.sha1(ctx.encode()).hexdigest()[:12]}",
                "family": "lsat_lr", "task_type": tt, "difficulty": "hard",
                "mode": "mcq", "stimulus": ctx, "question": norm_ws(ex["question"]),
                "mc_question": norm_ws(ex["question"]),
                "mc_choices": [norm_ws(a) for a in ans], "mc_credited_index": int(label),
                "reference_answer": norm_ws(ans[label]),
                "source": "ReClor (ICLR 2020) train+val",
            })
    return out


def parse_logiqa() -> list[dict]:
    out = []
    for line in (RAW / "logiqa2_test.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        opts, ans = d.get("options") or [], d.get("answer")
        text, q = norm_ws(d.get("text", "")), norm_ws(d.get("question", ""))
        if not text or not q or not isinstance(ans, int) or len(opts) != 4 or ans >= 4:
            continue
        # cross-family dedup: drop LogiQA passages duplicated in ReClor (both draw from
        # LSAT). ReClor keeps them; the appended repl-* replacements (not in ReClor) survive.
        if _norm_key(text) in _reclor_keys():
            continue
        out.append({
            "id": f"logiqa-{d.get('id', len(out))}",
            # multiple questions share one passage across distinct ids -> group by the
            # passage so a passage never appears in both train and eval.
            "group_id": f"logiqa-ctx-{hashlib.sha1(text.encode()).hexdigest()[:12]}",
            "family": "logiqa",
            "task_type": "inference", "difficulty": "hard", "mode": "mcq",
            "stimulus": text, "question": q, "mc_question": q,
            "mc_choices": [norm_ws(str(o)) for o in opts], "mc_credited_index": int(ans),
            "reference_answer": norm_ws(str(opts[ans])),
            "source": "LogiQA 2.0 (English) test",
        })
    return out


def parse_arct() -> list[dict]:
    out = []
    for split in ("test-adv-negated", "dev-adv-negated", "train-adv-negated"):
        p = RAW / f"arct_{split}.csv"
        if not p.exists():
            continue
        for row in csv.DictReader(io.StringIO(p.read_text(encoding="utf-8")), delimiter="\t"):
            try:
                correct = int(row["correctLabelW0orW1"])
            except (KeyError, ValueError, TypeError):
                continue
            w0, w1 = norm_ws(row.get("warrant0", "")), norm_ws(row.get("warrant1", ""))
            reason, claim = norm_ws(row.get("reason", "")), norm_ws(row.get("claim", ""))
            title = norm_ws(row.get("debateTitle", ""))
            if not (w0 and w1 and reason and claim) or correct not in (0, 1):
                continue
            neg = str(row.get("adversarial", "")).strip().lower() == "true"
            rid = norm_ws(row.get("#id", len(out)))
            # ARCT ships one row per crowd-annotator of the SAME argument (#id =
            # <claimid>_<workerid>, claimid = \d+_\d+). All annotators (and the negated
            # variant) share the Topic/Reason/Claim stimulus, so group by claim id to keep
            # them on one split side (else the same argument leaks train<->eval).
            cm = re.match(r"(\d+_\d+)", str(rid))
            claim_id = cm.group(1) if cm else str(rid)     # group key only — NOT the claim text
            out.append({
                "id": f"arct-{split.split('-')[0]}-{rid}-{'n' if neg else 'o'}",
                "group_id": f"arct-{claim_id}",
                "family": "arct", "task_type": "warrant", "difficulty": "hard",
                "mode": "mcq", "stimulus": f"Topic: {title}\nReason: {reason}\nClaim: {claim}",
                "question": "Which warrant makes the reason support the claim?",
                "mc_question": ("Which of the two candidate warrants correctly completes the "
                                "argument (is the unstated assumption that makes the reason "
                                "support the claim)?"),
                "mc_choices": [w0, w1], "mc_credited_index": correct,
                "reference_answer": w0 if correct == 0 else w1,
                "source": "Adversarial ARCT (Niven & Kao 2019)",
            })
    return out


def parse_folio() -> list[dict]:
    out = []
    for src in ("folio_train.jsonl", "folio_validation.jsonl"):
        tag = "v" if "valid" in src else "t"
        p = RAW / src
        if not p.exists():
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            label = S.canon_label(str(d.get("label", "")))
            prem = d.get("premises", [])
            concl = norm_ws(d.get("conclusion", ""))
            if label not in ("True", "False", "Uncertain", "Unknown") or not prem or not concl:
                continue
            neutral = "Uncertain"                       # FOLIO's neutral token
            label = neutral if label in S.NEUTRAL else label
            if isinstance(prem, str):
                prem = [prem]
            stim = ("Premises:\n" + "\n".join(f"- {norm_ws(p)}" for p in prem)
                    + f"\n\nConclusion: {concl}")
            pf = d.get("premises-FOL") or d.get("premises_fol") or []
            if isinstance(pf, str):
                pf = [pf]
            out.append({
                "id": f"folio-{tag}-{d.get('example_id', i)}",
                # group by the full premises+conclusion so any duplicate item stays on one side
                "group_id": f"folio-ctx-{hashlib.sha1(stim.encode()).hexdigest()[:12]}",
                "family": "folio",
                "task_type": "entailment", "difficulty": "hard", "mode": "frq",
                "stimulus": stim, "_premises": [norm_ws(x) for x in prem],
                "_statement": concl,
                "_premises_fol": [x for x in pf if str(x).strip()],   # val + train ship this
                "_conclusion_fol": d.get("conclusion-FOL") or d.get("conclusion_fol") or None,
                "question": ("Based ONLY on the premises, is the conclusion True, False, "
                             "or Uncertain (it does not deductively follow either way)?"),
                "choices": ["True", "False", "Uncertain"],
                "credited_index": ["True", "False", "Uncertain"].index(label),
                "reference_answer": label,
                "source": f"FOLIO (Han et al.) {'validation' if tag=='v' else 'train'}",
            })
    return out


def parse_logicnli() -> list[dict]:
    MAP = {"entailment": "True", "contradiction": "False", "neutral": "Unknown"}
    out = []
    for i, line in enumerate((RAW / "logicnli_dev.jsonl").read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        at = str(d.get("answer_text", ""))
        if at not in MAP:                       # drop self_contradiction
            continue
        ctx = norm_ws(d.get("context", ""))
        ctx = re.sub(r"^context:\s*", "", ctx, flags=re.I)
        parts = re.split(r"\bstatement:\s*", ctx, flags=re.I)
        premises = norm_ws(parts[0])
        statement = norm_ws(parts[1]) if len(parts) > 1 else ""
        if not premises or not statement:
            continue
        label = MAP[at]
        # FOL from the ORIGINAL structured logic form (exact), joined by the verified index
        # (logi_glue row i == block i//20, statement i%20). Then PER-CONCLUSION PRUNING to a
        # classically-consistent, z3-verified, gold-preserving premise subset (logicnli_prune):
        # LogicNLI blocks are inconsistent by construction, so the whole block is unusable for a
        # classical solver / LGMT. We keep the largest premise subset that stays satisfiable and
        # still yields the gold verdict, and REBUILD the stimulus from just those premises. A
        # statement that can't be made consistent-and-gold-matching is DROPPED (never inconsistent).
        pf = cf = nl_prem = None
        b, sidx = i // 20, i % 20
        structured = logicnli_logic is not None and logicnli_logic.available("dev")
        if structured and logicnli_prune is not None:
            pr = logicnli_prune.prune_for(b, sidx, label)
            if pr is None:
                continue                            # cannot make classically consistent -> drop
            _kept, pf, cf, nl_prem = pr
            premises = " ".join(s for s in nl_prem if s)   # stimulus = kept premises only
        else:                                       # degrade: structured source / pruner absent
            if structured:
                try:
                    pf, cf, nl_prem = logicnli_logic.fol_for(b, sidx)
                except Exception:
                    pf = cf = nl_prem = None
            if pf is None and logicnli_fol is not None:
                try:
                    pf, cf = logicnli_fol.context_to_fol(d.get("context", ""))
                except Exception:
                    pf = cf = None
        out.append({
            # one facts+rules block (id_) carries ~15 distinct statements; keep the id
            # per-context here (the global uniqueness pass in main() disambiguates the
            # rows), and group the split by context so all statements of a block stay on
            # the same side (this preserves the previous, leakage-free logicnli split).
            "id": f"logicnli-{d.get('id_', i)}", "group_id": f"logicnli-{d.get('id_', i)}",
            "family": "logicnli",
            "task_type": "entailment", "difficulty": "hard", "mode": "frq",
            "stimulus": premises, "_premises": [premises], "_statement": statement,
            "_premises_fol": pf, "_conclusion_fol": cf, "_nl_premises_fol": nl_prem,
            "question": (f"Based ONLY on the context above, is the statement "
                         f"\"{statement}\" True, False, or Unknown?"),
            "choices": ["True", "False", "Unknown"],
            "credited_index": ["True", "False", "Unknown"].index(label),
            "reference_answer": label, "source": "LogicNLI (Tian et al. 2021) dev",
        })
    return out


def parse_proverqa() -> list[dict]:
    out = []
    for diff in ("easy", "medium", "hard"):
        p = RAW / f"proverqa_{diff}.json"
        if not p.exists():
            continue
        for i, d in enumerate(json.loads(p.read_text(encoding="utf-8"))):
            opts = d.get("options", [])
            ans = str(d.get("answer", "")).strip()
            if not opts or ans not in ("A", "B", "C"):
                continue
            label = norm_ws(opts[ord(ans) - 65].split(")", 1)[-1])   # "A) True" -> "True"
            label = S.canon_label(label) or label
            ctx = norm_ws(d.get("context", ""))
            statement = norm_ws(d.get("question", "")).replace(
                "Based on the above information, is the following statement true, false, or uncertain? ", "")
            if label not in ("True", "False", "Uncertain") or not ctx:
                continue
            # ProverQA ships FOL for every item: nl2fol (NL premise -> FOL) + conclusion_fol.
            # Carry it so z3 can cover the NEUTRAL (Uncertain) class, which has no Prover9 chain
            # (the chain only exists for True/False). True/False keep the shipped chain.
            n2f = d.get("nl2fol") or {}
            pf = [v for v in n2f.values() if str(v).strip()]
            nlp = [k for k, v in n2f.items() if str(v).strip()]
            out.append({
                "id": f"proverqa-{diff}-{d.get('id', i)}", "family": "proverqa",
                "group_id": f"proverqa-ctx-{hashlib.sha1(ctx.encode()).hexdigest()[:12]}",
                "task_type": "entailment", "difficulty": diff, "mode": "frq",
                "stimulus": ctx, "_premises": [s.strip() for s in ctx.split(".") if s.strip()],
                "_statement": statement, "_reasoning": d.get("reasoning", ""),
                "_premises_fol": pf, "_conclusion_fol": d.get("conclusion_fol") or None,
                "_nl_premises_fol": nlp,
                "question": ("Based only on the premises, is the statement "
                             f"\"{statement}\" True, False, or Uncertain?"),
                "choices": ["True", "False", "Uncertain"],
                "credited_index": ["True", "False", "Uncertain"].index(label),
                "reference_answer": label, "source": f"ProverQA ({diff}) eval",
            })
    return out


# ==========================================================================
# The ONE house-style verbalizer (explanations generated here, at build time)
# ==========================================================================
def _variant(iid: str, options: list[str]) -> str:
    return options[int(hashlib.sha1(iid.encode()).hexdigest()[:4], 16) % len(options)]


def _clip(s: str, n: int) -> str:
    """Truncate at a word boundary with an ellipsis (no mid-word cuts)."""
    s = norm_ws(s)
    if len(s) <= n:
        return s
    return s[:n].rsplit(" ", 1)[0].rstrip(",.;:") + "…"


def _proverqa_chain(raw: str) -> str:
    """Restyle ProverQA's shipped fact/rule/conclusion chain into prose."""
    sents = []
    for block in [b for b in raw.split("\n\n") if b.strip()]:
        facts, rule, concl = [], None, None
        for ln in block.splitlines():
            low = ln.strip().lower()
            if low.startswith("fact"):
                facts.append(ln.split(":", 1)[-1].strip().rstrip("."))
            elif low.startswith("rule"):
                rule = ln.split(":", 1)[-1].strip().rstrip(".")
            elif low.startswith("conclusion"):
                concl = ln.split(":", 1)[-1].strip().rstrip(".")
        if concl and rule:
            prem = " and ".join(f for f in facts if f)
            sents.append(f"Since {prem}, and given that {rule.lower()}, it follows that {concl}."
                         if prem else f"Given that {rule.lower()}, {concl}.")
    return " ".join(sents[:6])


def attach_solver_path(it: dict) -> None:
    """For a FOLIO/LogicNLI item carrying FOL, run the appropriate working-path solver and,
    ONLY if it decides AND agrees with the gold label, stash a real derivation on
    `_solver_path`. Solver-verified, so a wrong parse/label is silently dropped -> template
    fallback (never poisons data). This is how FOLIO/LogicNLI get ProverQA-style real traces.

    Both FOLIO and (post-pruning) LogicNLI carry a classically CONSISTENT FOL theory, so both
    use z3 global entailment (`fol_solver.classify_path`): True/False cite the entailing
    premises, Uncertain/Unknown gets a countermodel witness. LogicNLI is consistent here only
    because `parse_logicnli` pruned each block to a satisfiable, gold-preserving premise subset
    (raw LogicNLI blocks are inconsistent and z3 would refuse them)."""
    if fol_solver is None:
        return
    pf, cf = it.get("_premises_fol"), it.get("_conclusion_fol")
    if not pf or not cf:
        return
    gold = S.canon_label(it["reference_answer"]) or it["reference_answer"]
    stmt = it.get("_statement", "the statement")
    nlp = it.get("_nl_premises_fol") or it.get("_premises")   # aligned NL for citation
    try:
        lab, path, _ = fol_solver.classify_path(pf, cf, nlp, stmt)
    except Exception:
        return
    if lab and path and S.labels_equiv(lab, gold):
        it["_solver_path"] = path


def verbalize(it: dict, lgmt_mr: str | None = None) -> str:
    """Return the house-style completion 'Reasoning: ...\\nAnswer: <X>'."""
    fam, mode = it["family"], it["mode"]

    if mode == "mcq":
        ch, idx = S.mc_of(it)
        letter = chr(65 + idx)
        credited = norm_ws(ch[idx])
        if fam == "lsat_lr":
            move = {"assumption": "the unstated assumption the conclusion depends on",
                    "flaw": "the flaw in how the conclusion is supported",
                    "weaken": "what most weakens the argument"}.get(it.get("task_type"),
                                                                    "the best-supported option")
            # options here can be whole parallel arguments -> only quote if short
            tail = f" (\"{_clip(credited, 130)}\")" if len(credited) <= 140 else ""
            reason = (f"Option {letter} is the choice that captures {move}{tail}. The other "
                      f"options do not bear on the gap between the premises and the conclusion.")
        elif fam == "arct":
            reason = (f"Warrant {letter} is the unstated assumption that makes the reason "
                      f"actually support the claim: \"{_clip(credited, 160)}\". The alternative "
                      f"warrant would not license the inference.")
        else:  # logiqa
            reason = (f"Option {letter} is the one that can be deduced from the passage: "
                      f"\"{_clip(credited, 160)}\". The others are not entailed by the premises.")
        # natural prose + a committing final sentence (parseable, but not a rigid tag)
        return f"{reason} So the correct option is ({letter})."

    # FRQ entailment core. The reasoning avoids the literal label word so the ONLY
    # label token is the committing final sentence (keeps parse_label's last-hit clean).
    label = S.canon_label(it["reference_answer"]) or it["reference_answer"]
    stmt = it.get("_statement", "the conclusion")
    reason = ""
    if fam == "proverqa" and it.get("_reasoning") and label != "Uncertain":
        reason = _proverqa_chain(it["_reasoning"])          # real prover chain, no label word
    if not reason and it.get("_solver_path"):
        reason = it["_solver_path"]      # real z3 derivation / countermodel witness (FOLIO/LogicNLI)
    if not reason:
        if label in S.NEUTRAL:
            reason = _variant(it["id"], [
                (f"Working only from the premises, the statement \"{stmt}\" can be neither "
                 f"derived nor refuted — nothing given settles it either way."),
                (f"Testing both directions, the premises neither establish \"{stmt}\" nor its "
                 f"negation; no rule chain reaches it, so it is under-determined."),
            ])
        elif label == "True":
            reason = _variant(it["id"], [
                (f"Each step needed to establish \"{stmt}\" is supported by the premises, so "
                 f"it follows deductively."),
                (f"Chaining the premises forward establishes \"{stmt}\"; it is entailed."),
            ])
        else:  # False
            reason = _variant(it["id"], [
                (f"The premises establish something inconsistent with \"{stmt}\", so it "
                 f"cannot hold."),
                (f"Following the premises leads to the negation of \"{stmt}\", so it does "
                 f"not hold."),
            ])
    if lgmt_mr:
        reason = (f"This reformulation ({lgmt_mr}) preserves the logical content, so the "
                  f"conclusion is unchanged. ") + reason
    # natural prose + a committing final sentence
    return f"{reason} Therefore, based only on the premises, the statement is {label}."


# ==========================================================================
# LGMT structural follow-ups (drift-free; TRAIN entailment sources only)
# ==========================================================================
_IRRELEVANT = "Additionally, an unrelated fact holds: Quennell owns a teal kite."


def _lgmt_variants(it: dict) -> list[dict]:
    """Logic-preserving structural morphs of an entailment item, MRs varied per
    family (P1 reorder, P2 duplicate, P3 irrelevant) so LGMT isn't P3-dominated."""
    outs = []
    base_id = it["id"]

    if it["family"] == "folio" and "Premises:" in it["stimulus"]:
        head, _, tail = it["stimulus"].partition("\n\nConclusion:")
        lines = [ln for ln in head.splitlines() if ln.startswith("- ")]
        if len(lines) >= 2:
            outs.append(("P1-reorder",
                         "Premises:\n" + "\n".join(reversed(lines)) + "\n\nConclusion:" + tail))
            outs.append(("P2-duplicate",
                         "Premises:\n" + "\n".join(lines + [lines[0]]) + "\n\nConclusion:" + tail))
            outs.append(("P3-irrelevant",
                         "Premises:\n" + "\n".join(lines + [f"- {_IRRELEVANT}"]) + "\n\nConclusion:" + tail))
    else:
        # ProverQA / LogicNLI paragraph stimulus: reorder / duplicate / extend sentences
        sents = [s.strip() for s in it["stimulus"].split(". ") if s.strip()]
        if len(sents) >= 2:
            rev = ". ".join(reversed(sents))
            outs.append(("P1-reorder", rev if rev.endswith(".") else rev + "."))
            outs.append(("P2-duplicate", it["stimulus"].rstrip() + " " + sents[0].rstrip(".") + "."))
        outs.append(("P3-irrelevant", it["stimulus"].rstrip() + " " + _IRRELEVANT))

    morphs = []
    for mr, stim in outs:
        m = dict(it)
        m["id"] = f"{base_id}-lgmt-{mr}"
        m["stimulus"] = stim
        m["is_lgmt"] = True
        m["lgmt_mr"] = mr
        morphs.append(m)
    return morphs


# ==========================================================================
# Build
# ==========================================================================
def to_row(it: dict, lgmt_mr: str | None = None) -> dict:
    mode = it["mode"]
    return {"id": it["id"], "family": it["family"], "mode": mode,
            "task_type": it.get("task_type"), "difficulty": it.get("difficulty"),
            "prompt": S.build_prompt(it, mode), "completion": verbalize(it, lgmt_mr),
            "gold": S.credited_answer(it, mode), "trace_source":
            ("solver_path" if it.get("_solver_path")
             else "prover_chain" if (it["family"] == "proverqa" and it.get("_reasoning")
                                     and it["reference_answer"] != "Uncertain")
             else "grounded_template"),
            "is_lgmt": bool(lgmt_mr), "lgmt_mr": lgmt_mr}


def eval_view(it: dict) -> dict:
    """Full item for the held-out eval (drop build-only underscore fields)."""
    return {k: v for k, v in it.items() if not k.startswith("_")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lgmt", type=int, default=LGMT_TARGET)
    args = ap.parse_args()

    print("Parsing raw sources ...")
    pools = {
        "lsat_lr": cap(parse_reclor(), CAPS["reclor"], args.seed),
        "logiqa": cap(parse_logiqa(), CAPS["logiqa"], args.seed),
        "arct": cap(parse_arct(), CAPS["arct"], args.seed),
        "folio": cap(parse_folio(), CAPS["folio"], args.seed),
        "logicnli": cap(parse_logicnli(), CAPS["logicnli"], args.seed),
    }
    # ProverQA capped per difficulty (med/hard emphasis)
    pv = parse_proverqa()
    by_diff = defaultdict(list)
    for it in pv:
        by_diff[it["difficulty"]].append(it)
    pools["proverqa"] = []
    for diff, n in CAPS["proverqa"].items():
        pools["proverqa"] += cap(by_diff[diff], n, args.seed)

    items = [it for fam in pools.values() for it in fam]
    print(f"  base items: {len(items)}  ", dict(Counter(it["family"] for it in items)))

    # Guarantee globally-unique row ids (LogicNLI reuses one id per facts+rules block;
    # LogiQA has a few duplicate source ids). The split groups by group_id, so making ids
    # unique here does NOT move any item across the split; it only stops distinct problems
    # (and their LGMT morphs, whose id derives from this one) from colliding on one id.
    _seen: dict = {}
    for it in items:
        base = it["id"]
        n = _seen.get(base, 0)
        _seen[base] = n + 1
        if n:
            it["id"] = f"{base}#{n}"

    # attach real solver working-paths (FOLIO val ships FOL; LogicNLI via its parser)
    # BEFORE the split so LGMT follow-ups (built from train) inherit the derivation too.
    if fol_solver is not None:
        for it in items:
            # FOLIO-train ships no conclusion-FOL -> fill it from the cached NL->FOL
            # translation (cache-first; makes a gateway call only for uncached items).
            if (it["family"] == "folio" and it.get("_premises_fol")
                    and not it.get("_conclusion_fol") and folio_concl_fol is not None):
                it["_conclusion_fol"] = folio_concl_fol.translate(
                    it["id"], it.get("_statement", ""), it["_premises_fol"],
                    gold=S.canon_label(it["reference_answer"]) or it["reference_answer"])
            if it["family"] in ("folio", "logicnli"):
                attach_solver_path(it)
            # ProverQA: z3 ONLY for the neutral (Uncertain) class — it has no Prover9 chain.
            # True/False keep the shipped chain (prover_chain), which is a richer derivation.
            elif (it["family"] == "proverqa" and it.get("_premises_fol")
                  and it.get("_conclusion_fol")
                  and (S.canon_label(it["reference_answer"]) in S.NEUTRAL)):
                attach_solver_path(it)
        if folio_concl_fol is not None:
            try:
                folio_concl_fol.save_cache()
            except Exception:
                pass
        sp = Counter((it["family"], "solver" if it.get("_solver_path") else "template")
                     for it in items if it["family"] in ("folio", "logicnli", "proverqa"))
        print(f"  solver-path coverage: {dict(sp)}")

    # hashed per-id split -> held-out slice of EVERY family
    train, ev = S.train_eval_split(items, eval_frac=args.eval_frac, seed=args.seed)

    # LGMT follow-ups from TRAIN entailment items only (no leakage). Balanced:
    # ~lgmt/3 per entailment family, with P3-irrelevant capped at half so the mix
    # isn't P3-dominated (Plan 1 fix for the FOLIO over-representation).
    per_family = max(1, args.lgmt // len(S.ENTAILMENT_FAMILIES))
    p3_cap = per_family // 2
    lgmt_rows = []
    for fam in sorted(S.ENTAILMENT_FAMILIES):
        fam_items = sorted([it for it in train if it["family"] == fam],
                           key=lambda x: _rank(x["id"], args.seed + 1))
        made, p3 = 0, 0
        for it in fam_items:
            if made >= per_family:
                break
            for m in _lgmt_variants(it):
                if made >= per_family:
                    break
                if m["lgmt_mr"].startswith("P3") and p3 >= p3_cap:
                    continue
                lgmt_rows.append(to_row(m, lgmt_mr=m["lgmt_mr"]))
                made += 1
                if m["lgmt_mr"].startswith("P3"):
                    p3 += 1

    train_rows = [to_row(it) for it in train] + lgmt_rows

    # Step-B overlay: replace a row's deterministic completion with a VERIFIED frontier trace
    # (Opus blind / answer-conditioned + blind re-read) from data/folio_stepb_review.jsonl, so
    # those traces survive rebuilds. Only 'blind'/'backfill' rows are applied, and only onto the
    # matching id — the committing final sentence still yields the gold, so 0-grade-mismatch holds.
    overlay = {}
    for overlay_path in sorted(OUT.glob("*_stepb_review.jsonl")):
        for line in overlay_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            o = json.loads(line)
            if o.get("trace_source") in ("blind", "backfill") and o.get("completion"):
                overlay[o["id"]] = o
    if overlay:
        applied = 0
        for r in train_rows:
            o = overlay.get(r["id"])
            if o and r["gold"] == o.get("gold"):
                r["completion"] = o["completion"]
                r["trace_source"] = o["trace_source"]
                applied += 1
        print(f"  step-B overlay: applied {applied} frontier traces "
              f"from {len(list(OUT.glob('*_stepb_review.jsonl')))} review file(s)")

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "sft_train.jsonl").open("w", encoding="utf-8") as fh:
        for r in train_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (OUT / "eval_items.json").write_text(
        json.dumps([eval_view(it) for it in ev], ensure_ascii=False, indent=2), encoding="utf-8")

    # LGMT source export: the classically-CONSISTENT FOL entailment cases (LogicNLI is
    # consistent only after the per-conclusion pruning above; FOLIO ships consistent FOL).
    # Format matches critical-reasoning-eval/lgmt_full.py's source items so LGMT can run
    # metamorphic testing on them (it verifies satisfiability again with z3 before use).
    lgmt_src = []
    for it in items:
        if it["family"] not in ("logicnli", "folio"):
            continue
        if not it.get("_solver_path"):
            continue                                  # only z3-VERIFIED (SAT + verdict==gold) cases
        pf = it.get("_premises_fol") or []
        cf = it.get("_conclusion_fol")
        nlp = it.get("_nl_premises_fol") if it["family"] == "logicnli" else it.get("_premises")
        if not pf or not cf or not nlp or len(nlp) != len(pf):
            continue                                  # need aligned NL/FOL premises
        g = it["reference_answer"]
        g = "Unknown" if g in ("Uncertain", "Unknown") else g   # LGMT's 3-valued scheme
        lgmt_src.append({
            "id": it["id"], "family": it["family"],
            "premises": [str(x) for x in nlp], "premises_fol": [str(x) for x in pf],
            "conclusion": it.get("_statement", ""), "conclusion_fol": str(cf),
            "gold": g,
        })
    (OUT / "lgmt_sources.json").write_text(
        json.dumps(lgmt_src, ensure_ascii=False, indent=2), encoding="utf-8")

    # meta / provenance
    fmt = Counter()
    for r in train_rows:
        key = ("FRQ-entailment" if r["family"] in S.ENTAILMENT_FAMILIES else f"MCQ-{r['family']}")
        fmt[key] += 1
    meta = {
        "n_train_rows": len(train_rows), "n_eval_items": len(ev),
        "n_lgmt": len(lgmt_rows), "eval_frac": args.eval_frac, "seed": args.seed,
        "train_by_family": dict(Counter(r["family"] for r in train_rows)),
        "train_by_mode": dict(Counter(r["mode"] for r in train_rows)),
        "train_by_format_type": dict(fmt),
        "proverqa_by_difficulty": dict(Counter(
            r["difficulty"] for r in train_rows if r["family"] == "proverqa")),
        "eval_by_family": dict(Counter(it["family"] for it in ev)),
        "eval_neutral_gold": sum(1 for it in ev if S.is_neutral_gold(
            it, "frq" if it["family"] in S.ENTAILMENT_FAMILIES else "mcq")),
        "trace_source": dict(Counter(r["trace_source"] for r in train_rows)),
    }
    (OUT / "corpus_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nTRAIN rows: {len(train_rows)} (incl. {len(lgmt_rows)} LGMT)")
    print("  by family:", meta["train_by_family"])
    print("  by format:", meta["train_by_format_type"])
    print("  proverqa difficulty:", meta["proverqa_by_difficulty"])
    print("  trace source:", meta["trace_source"])
    print(f"EVAL items: {len(ev)}  {meta['eval_by_family']}  (neutral-gold {meta['eval_neutral_gold']})")
    print(f"\nWrote {OUT}/sft_train.jsonl, eval_items.json, corpus_meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
