#!/usr/bin/env python3
"""Step B — frontier verbalizer with gold-verification (PLAN.md step B).

Replaces the deterministic grounded-template completions in ``data/sft_train.jsonl``
with **real, faithfully-verified natural reasoning** from a frontier teacher, in the
same house style (natural prose + a committing final sentence). Applies uniformly to
the entailment core AND the MCQ families (the MCQ analog of a solver, since those have
no FOL):

  1. **Blind solve** — ask the teacher the item prompt with NO gold. If its committed
     answer matches gold, keep that trace (it's genuinely derived → faithful).
  2. **Answer-conditioned backfill** — for the blind misses, give the teacher the gold
     answer and ask for a rigorous natural justification. Then a **blind re-read check**
     (a second call shown ONLY the justification's reasoning) must re-commit to the
     gold answer — this catches rationalization. Keep only if it passes.
  3. Rows that never pass keep their deterministic completion (never poison the data).

Notes:
- Needs a Claude API key in ``colab/.env`` (``ANTHROPIC_API_KEY=...``) — calls go
  through the Anthropic SDK (``slm_core.call_agent``). If the key is missing, every
  teacher call returns None and rows fall back to their deterministic completion, so a
  run degrades cleanly to a no-op (logged) rather than crashing.
- **Resumable**: per-row results are cached to ``<out>.cache.jsonl``; re-running skips
  finished ids.
- The symbolic solver for FOLIO-val / ProverQA (proof/countermodel handed to a
  translate-only verbalizer) is the documented higher-fidelity variant; this uses the
  teacher as reasoner+verifier, which needs no fragile FOL->Z3 parser.

Usage:
    python stepb_verbalize.py --teacher claude-opus-4-8-thinking-high --limit 20   # sample
    python stepb_verbalize.py --teacher gpt-5.5-high --out data/sft_train_stepb.jsonl
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import sys
import threading
import time
from collections import Counter
from pathlib import Path

# this module lives in colab/data_construction/; slm_core.py and data/ are one level up
HERE = Path(__file__).resolve().parent
_COLAB = HERE.parent
DATA = _COLAB / "data"
if str(_COLAB) not in sys.path:
    sys.path.insert(0, str(_COLAB))

import slm_core as S
try:
    import fol_solver          # FOLIO symbolic solver (needs z3); optional
except Exception:
    fol_solver = None


def load_folio_fol() -> dict:
    """Map corpus id 'folio-v-{i}' -> the FOLIO-val record (carries premises-FOL /
    conclusion-FOL), matching build_corpus.parse_folio's line-index ids."""
    p = DATA / "_raw" / "folio_validation.jsonl"
    if not p.exists():
        return {}
    out = {}
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            out[f"folio-v-{i}"] = json.loads(line)
        except json.JSONDecodeError:
            continue
    return out


def folio_reason_prompt(item: dict, verdict: str, detail: str) -> str:
    prem = "\n".join(f"- {p}" for p in item.get("premises", []))
    return (f"Premises:\n{prem}\n\nConclusion: {item.get('conclusion','')}\n\n"
            f"A formal logic solver determined that, based ONLY on the premises, the "
            f"conclusion is {verdict} ({detail}). Write a short, natural chain of "
            f"reasoning showing why, then end by clearly stating the conclusion is "
            f"{verdict}. Do not mention the solver.")


def n_options(prompt: str) -> int:
    """Count the (A)/(B)/… options listed in an MCQ prompt (for letter validation)."""
    opts = re.findall(r"^\(([A-Z])\)\s", prompt, re.M)
    return len(opts) if opts else 5


def gold_letter(gold: str) -> str | None:
    m = re.match(r"\(([A-Z])\)", gold or "")
    return m.group(1) if m else None


def verify(response: str, row: dict) -> bool:
    """Does the teacher's response commit to the gold answer? (deterministic)"""
    if not response:
        return False
    if row["mode"] == "mcq":
        n = n_options(row["prompt"])
        idx = S.parse_letter(response, n)
        return idx is not None and chr(65 + idx) == gold_letter(row["gold"])
    gold = S.canon_label(row["gold"]) or row["gold"]
    return S.labels_equiv(S.parse_label(response), gold)


# FRQ style constraint: keep the label word (True/False/Uncertain/Unknown) OUT of the
# reasoning body so the ONLY label token is the committing final sentence (the whole corpus
# relies on this last-label-hit invariant; free-form teacher traces otherwise scatter labels
# mid-reasoning and teach hedging/multi-label output — measured to hurt commitment + LGMT MVR).
NO_LABEL_FRQ = ("\n\nSTYLE REQUIREMENT: In your reasoning, do NOT use the words \"true\", "
                "\"false\", \"uncertain\", or \"unknown\" — refer to it as \"the statement\" and "
                "describe what does or does not follow from the premises. State your verdict in "
                "ONE final sentence ONLY, formatted exactly: \"Therefore, the statement is X.\" "
                "where X is True, False, or Uncertain (use Unknown only if the options say so).")

# MCQ analog: the option LETTER is the label. Keep it out of the reasoning body (refer to
# options by content) so the only letter token is the committing final sentence — same
# last-hit-parse invariant that FRQ needs (parse_letter also takes the LAST letter / the last
# "option X" phrase, so a mid-reasoning "Option A is wrong" would otherwise pollute it).
NO_LETTER_MCQ = ("\n\nSTYLE REQUIREMENT: In your reasoning, refer to the options by their CONTENT, "
                 "NOT by their letter — do not write any option letter (A, B, C, D, E) or the word "
                 "\"option\" until the very end. State your choice in ONE final sentence ONLY, "
                 "formatted exactly: \"Therefore, the correct option is (X).\"")


def blind_prompt(row: dict) -> str:
    """Blind solve prompt (no gold), with the style constraint that keeps the label/letter
    out of the reasoning body (FRQ: no True/False/…; MCQ: no option letters until the end)."""
    return row["prompt"] + (NO_LABEL_FRQ if row["mode"] == "frq" else NO_LETTER_MCQ)


def backfill_prompt(row: dict) -> str:
    """Answer-conditioned: hand the teacher the gold answer, ask for a natural,
    rigorous justification in the house style (reason first, commit last)."""
    if row["mode"] == "mcq":
        goal = f"the correct option is {gold_letter(row['gold'])}"
    else:
        goal = f"the correct answer is {S.canon_label(row['gold']) or row['gold']}"
    return (row["prompt"] + f"\n\n(For reference, {goal}.) Write a rigorous, natural "
            "explanation that derives this from the given information — reason step by "
            "step, then end by clearly committing to the answer. Do not mention that the "
            "answer was given to you; reason as if solving it."
            + (NO_LABEL_FRQ if row["mode"] == "frq" else NO_LETTER_MCQ))


def reread_prompt(reasoning: str, row: dict) -> str:
    """Blind re-read: show ONLY the reasoning, ask which answer it commits to."""
    if row["mode"] == "mcq":
        return ("Below is reasoning about a multiple-choice question. Reply with ONLY the "
                "single option letter it concludes. No other text.\n\n" + reasoning)
    return ("Below is reasoning about a True/False/Uncertain-style logic question. Reply "
            "with ONLY one word — True, False, Uncertain, or Unknown — that it concludes. "
            "No other text.\n\n" + reasoning)


def strip_meta(text: str) -> str:
    """Remove any leaked 'the answer was given'/'for reference' asides."""
    out = [ln for ln in text.splitlines()
           if not re.search(r"\b(as (given|provided)|for reference|the answer was)\b", ln, re.I)]
    return "\n".join(out).strip()


def reword_prompt(row: dict) -> str:
    """For families whose completion is already a correct derivation (ProverQA's
    shipped chain): lightly reword into natural prose, same logic + same answer."""
    return ("Reword the following solution into clear, natural prose. Do NOT change its "
            "logic, the steps, or the final answer — only the phrasing. Keep it concise "
            "and end by stating the same final answer.\n\n--- solution ---\n"
            + row["completion"])


def process_row(row: dict, opus: str, sonnet: str, haiku: str, reread_model: str, timeout: int,
                folio_fol: dict | None = None):
    """Produce (record, stat) for one row. Routing (cheap model wherever a derivation
    exists, Opus only where reasoning must be generated from scratch):
      * ProverQA (shipped prover chain)     -> Haiku *reword*
      * FOLIO-val with a solver-verified verdict -> Haiku *reword of the solver result*
      * everything else (no-trace)          -> Opus *generate* (blind, else backfill+re-read)
    Any row that doesn't pass its gate keeps its deterministic completion."""
    completion = row["completion"]

    if row.get("trace_source") == "prover_chain":
        rw = S.call_agent(reword_prompt(row), haiku, timeout)
        if rw and verify(rw, row):
            return {**row, "completion": strip_meta(rw), "trace_source": "reword"}, "reword"
        return {**row, "completion": completion, "trace_source": "deterministic"}, (
            "no_teacher" if rw is None else "reword_failed")

    # FOLIO-val: solve symbolically, and only if the solver AGREES with gold, hand the
    # verified verdict to Haiku to verbalize (cheap). Solver disagreement/undecided -> Opus.
    if (fol_solver is not None and folio_fol and row.get("family") == "folio"
            and row["id"] in folio_fol):
        it = folio_fol[row["id"]]
        lab, detail = fol_solver.classify(it.get("premises-FOL", []),
                                          it.get("conclusion-FOL", "") or "")
        gold = S.canon_label(row["gold"]) or row["gold"]
        if lab and S.labels_equiv(lab, gold):
            gen = S.call_agent(folio_reason_prompt(it, lab, detail), haiku, timeout)
            if gen and verify(gen, row):
                return {**row, "completion": strip_meta(gen), "trace_source": "solver"}, "solver"
        # else: fall through to Opus generate below

    # no-trace families -> ESCALATION CASCADE (cheapest teacher that lands on gold wins):
    #   1) Sonnet blind  2) Opus blind  3) Opus answer-conditioned + blind re-read gate.
    # Each blind trace is kept only if the teacher commits to gold UNPROMPTED (faithful).
    bp = blind_prompt(row)
    s = S.call_agent(bp, sonnet, timeout)                       # tier 1: cheap
    if s and verify(s, row):
        return {**row, "completion": strip_meta(s), "trace_source": "blind", "teacher": "sonnet"}, "sonnet_blind"
    o = S.call_agent(bp, opus, timeout)                         # tier 2
    if o and verify(o, row):
        return {**row, "completion": strip_meta(o), "trace_source": "blind", "teacher": "opus"}, "opus_blind"
    if s is None and o is None:                                 # gateway down -> keep template
        return {**row, "completion": completion, "trace_source": "deterministic"}, "no_teacher"
    bf = S.call_agent(backfill_prompt(row), opus, timeout)      # tier 3: answer-conditioned
    if bf and verify(bf, row):
        reread = S.call_agent(reread_prompt(strip_meta(bf), row), reread_model, timeout)
        if reread and verify(reread, row):
            return {**row, "completion": strip_meta(bf), "trace_source": "backfill", "teacher": "opus"}, "backfill"
        return {**row, "completion": completion, "trace_source": "deterministic"}, "reread_failed"
    return {**row, "completion": completion, "trace_source": "deterministic"}, "backfill_failed"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default=str(DATA / "sft_train.jsonl"))
    ap.add_argument("--out", default=str(DATA / "sft_train_stepb.jsonl"))
    ap.add_argument("--opus-model", default="claude-opus-4-8",
                    help="strong model — tier 2 blind + tier 3 answer-conditioned")
    ap.add_argument("--sonnet-model", default="claude-sonnet-4-6",
                    help="cheaper model tried FIRST (tier 1 blind) in the escalation cascade")
    ap.add_argument("--haiku-model", default="claude-haiku-4-5",
                    help="cheap model that REWORDS shipped chains (ProverQA)")
    ap.add_argument("--reread-model", default=S.DEFAULT_JUDGE_MODEL,
                    help="cheap model for the blind re-read faithfulness gate")
    ap.add_argument("--concurrency", type=int, default=12,
                    help="parallel API calls (raise/lower per your rate-limit tier)")
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows")
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.inp).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    cache_path = Path(args.out).with_suffix(".cache.jsonl")
    cache: dict[str, dict] = {}
    if cache_path.exists():
        for l in cache_path.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                cache[r["id"]] = r
    pending = [r for r in rows if r["id"] not in cache]
    folio_fol = load_folio_fol() if fol_solver is not None else {}

    def _route(r):
        if r.get("trace_source") == "prover_chain":
            return "haiku-reword"
        if folio_fol and r.get("family") == "folio" and r["id"] in folio_fol:
            return "solver+haiku(FOLIO-val)"
        return "opus-generate"
    from collections import Counter as _C
    routing = _C(_route(r) for r in pending)
    print(f"[stepb] {len(rows)} rows | {len(cache)} cached | {len(pending)} to do "
          f"| routing {dict(routing)} | concurrency {args.concurrency}", flush=True)

    stats = Counter({"cached": len(cache)})
    lock = threading.Lock()
    t0 = time.time()
    try:                                          # warm the client once (avoid init race)
        S._client()
    except Exception:
        pass

    with cache_path.open("a", encoding="utf-8") as cache_fh:
        with cf.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
            futs = {ex.submit(process_row, r, args.opus_model, args.sonnet_model,
                              args.haiku_model, args.reread_model, args.timeout, folio_fol): r
                    for r in pending}
            done = 0
            for fut in cf.as_completed(futs):
                try:
                    rec, stat = fut.result()
                except Exception:
                    rec, stat = None, "error"
                with lock:
                    if stat:
                        stats[stat] += 1
                    if rec is not None:
                        cache_fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); cache_fh.flush()
                        cache[rec["id"]] = rec
                    done += 1
                    if done % 50 == 0 or done == len(pending):
                        rate = done / max(1e-6, time.time() - t0)
                        print(f"  {done}/{len(pending)}  {dict(stats)}  "
                              f"({rate:.1f}/s, {time.time()-t0:.0f}s)", flush=True)

    out_rows = [cache[r["id"]] for r in rows if r["id"] in cache]
    with Path(args.out).open("w", encoding="utf-8") as fh:
        for r in out_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    frontier = (stats["sonnet_blind"] + stats["opus_blind"] + stats["blind"] + stats["backfill"]
                + stats["reword"] + stats["solver"])
    print(f"\n[stepb] wrote {len(out_rows)} rows -> {args.out}")
    print(f"  frontier: {frontier} (Sonnet blind {stats['sonnet_blind']} + Opus blind "
          f"{stats['opus_blind'] + stats['blind']} + Opus backfill {stats['backfill']} "
          f"+ Haiku reword {stats['reword']} + solver+Haiku {stats['solver']})")
    print(f"  kept deterministic: {len(out_rows) - frontier - stats['cached']} (this run)  {dict(stats)}")
    if stats["no_teacher"]:
        print("  NOTE: teacher unreachable — set ANTHROPIC_API_KEY in colab/.env and re-run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
