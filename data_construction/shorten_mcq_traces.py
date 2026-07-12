#!/usr/bin/env python3
"""Compress the MCQ Step-B frontier traces to ~4 crisp sentences (SEPARATE artifact).

Diagnosis: the verbose MCQ frontier traces (avg ~165-233 tok) inverted the corpus token
mass (MCQ 63% vs entailment 37%), crowding out the entailment core and bleeding a
"weigh-the-options" style that raised MVR. This tool builds a *parallel* set of shortened
MCQ completions so a rebalanced corpus is ready to swap in IF the ablation confirms the
crowding-out. It NEVER modifies the current files — it reads mcq_stepb_review.jsonl
read-only and writes to data/mcq_short_traces.jsonl (name chosen so build_corpus's
``*_stepb_review.jsonl`` overlay glob does NOT pick it up).

Per MCQ frontier row (trace_source blind/backfill): ask a model to compress the ALREADY
VERIFIED trace to <=N sentences, keeping the logic and the exact committing final
sentence, with the letter isolated to that sentence (NO_LETTER_MCQ invariant). Then
re-verify: (a) parse_letter still lands on gold, (b) no option letter in the reasoning
body, (c) meaningfully shorter. If any check fails -> keep the ORIGINAL trace (never
degrade coverage). Resumable: per-id results cached to <out>.cache.jsonl.

    python shorten_mcq_traces.py --limit 15 --model claude-haiku-4-5   # sample
    python shorten_mcq_traces.py --concurrency 8                       # full run
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

HERE = Path(__file__).resolve().parent
_COLAB = HERE.parent
DATA = _COLAB / "data"
if str(_COLAB) not in sys.path:
    sys.path.insert(0, str(_COLAB))
import slm_core as S

MCQ_FAMILIES = {"lsat_lr", "logiqa", "arct"}


def gold_letter(gold: str) -> str | None:
    m = re.match(r"\(([A-Z])\)", gold or "")
    return m.group(1) if m else None


def n_options(prompt: str) -> int:
    opts = re.findall(r"^\(([A-Z])\)\s", prompt, re.M)
    return len(opts) if opts else 5


def letters_in_body(text: str) -> int:
    """Option-letter tokens anywhere EXCEPT the final sentence (should be 0)."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    body = " ".join(parts[:-1]) if len(parts) > 1 else ""
    return len(re.findall(r"\(([A-Ea-e])\)", body)) + len(re.findall(r"\boption\s+[A-Ea-e]\b", body, re.I))


def n_sentences(text: str) -> int:
    return len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()])


def shorten_prompt(row: dict, max_sentences: int) -> str:
    L = gold_letter(row["gold"])
    return (
        f"Compress the following solution to a multiple-choice logic question into AT MOST "
        f"{max_sentences} short sentences of reasoning, then the final committing sentence.\n"
        "Rules:\n"
        "- Keep the essential logic; delete restatement of the passage/options and any hedging.\n"
        "- Refer to options by their CONTENT, not their letter. Do NOT write any option letter "
        "(A, B, C, D, E) or the word \"option\" anywhere except the final sentence.\n"
        f"- End with EXACTLY this sentence, verbatim: \"Therefore, the correct option is ({L}).\"\n"
        "Output ONLY the compressed solution, nothing else.\n\n"
        "--- solution ---\n" + row["completion"]
    )


def verify_short(short: str, row: dict) -> bool:
    if not short:
        return False
    n = n_options(row["prompt"])
    idx = S.parse_letter(short, n)
    if idx is None or chr(65 + idx) != gold_letter(row["gold"]):
        return False                                   # must still commit to gold
    if letters_in_body(short) != 0:
        return False                                   # letter must stay in the final sentence
    if len(short) >= len(row["completion"]):
        return False                                   # must actually be shorter
    return True


def process_row(row: dict, model: str, max_sentences: int, timeout: int):
    """Return (record, stat). record keeps the row schema with a shortened (or original)
    completion + a 'shortened' flag."""
    r = S.call_agent(shorten_prompt(row, max_sentences), model, timeout, max_tokens=400)
    short = (r or "").strip()
    if verify_short(short, row):
        return {**row, "completion": short, "shortened": True}, "shortened"
    return {**row, "shortened": False}, ("no_teacher" if r is None else "verify_failed")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default=str(DATA / "mcq_stepb_review.jsonl"))
    ap.add_argument("--out", default=str(DATA / "mcq_short_traces.jsonl"))
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--max-sentences", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="0 = all frontier MCQ rows")
    ap.add_argument("--timeout", type=int, default=90)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.inp).read_text(encoding="utf-8").splitlines() if l.strip()]
    # only compress verified MCQ frontier traces; template / entailment stay as-is (not emitted)
    todo_all = [r for r in rows
                if r.get("family") in MCQ_FAMILIES and r.get("trace_source") in ("blind", "backfill")]
    if args.limit:
        todo_all = todo_all[:args.limit]

    cache_path = Path(args.out).with_suffix(".cache.jsonl")
    cache: dict[str, dict] = {}
    if cache_path.exists():
        for l in cache_path.read_text(encoding="utf-8").splitlines():
            if l.strip():
                c = json.loads(l)
                cache[c["id"]] = c
    pending = [r for r in todo_all if r["id"] not in cache]
    print(f"[shorten] {len(todo_all)} frontier MCQ traces | {len(cache)} cached | {len(pending)} to do "
          f"| model {args.model} | <= {args.max_sentences} sentences | concurrency {args.concurrency}",
          flush=True)

    stats = Counter({"cached": len(cache)})
    lock = threading.Lock()
    t0 = time.time()
    try:
        S._client()
    except Exception:
        pass

    with cache_path.open("a", encoding="utf-8") as cfh:
        with cf.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
            futs = {ex.submit(process_row, r, args.model, args.max_sentences, args.timeout): r
                    for r in pending}
            done = 0
            for fut in cf.as_completed(futs):
                try:
                    rec, stat = fut.result()
                except Exception:
                    rec, stat = None, "error"
                with lock:
                    stats[stat] += 1
                    if rec is not None:
                        cfh.write(json.dumps(rec, ensure_ascii=False) + "\n"); cfh.flush()
                        cache[rec["id"]] = rec
                    done += 1
                    if done % 50 == 0 or done == len(pending):
                        print(f"  {done}/{len(pending)}  {dict(stats)}  ({done/max(1e-6,time.time()-t0):.1f}/s)",
                              flush=True)

    # emit every frontier MCQ id (shortened where it verified, else original) for a clean swap
    out_rows = [cache[r["id"]] for r in todo_all if r["id"] in cache]
    with Path(args.out).open("w", encoding="utf-8") as fh:
        for r in out_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    sh = [r for r in out_rows if r.get("shortened")]
    if sh:
        import statistics
        orig_by_id = {r["id"]: r for r in todo_all}
        red = [1 - len(r["completion"]) / max(1, len(orig_by_id[r["id"]]["completion"])) for r in sh]
        sent = [n_sentences(r["completion"]) for r in sh]
        print(f"\n[shorten] wrote {len(out_rows)} rows -> {args.out}")
        print(f"  shortened {len(sh)}/{len(out_rows)} | kept original {len(out_rows)-len(sh)} "
              f"| avg length reduction {statistics.mean(red)*100:.0f}% | avg sentences {statistics.mean(sent):.1f}")
    print(f"  stats {dict(stats)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
