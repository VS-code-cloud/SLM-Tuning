#!/usr/bin/env python3
"""Phase 1 of the gap-closing plan (SEPARATE artifact — current files untouched).

The per-item diff vs Opus 4.8 showed the SLM loses lsat_lr/logiqa by committing to a
*plausible-but-wrong distractor*; Opus wins by explicitly **ruling out each option** then
committing. This re-distills lsat_lr + logiqa frontier traces from **Opus 4.8** in that
elimination structure (name the argument's gap -> rule out the other options by content ->
commit), verified to land on gold with the option letter isolated to the final sentence.

Reads the shortened corpus read-only for (prompt, gold); writes data/mcq_elim_traces.jsonl
(NOT matching build_corpus's `*_stepb_review.jsonl` overlay glob). Family-scoped: entailment
is never touched (protects the proverqa / neutral wins). Resumable per-id cache.

    python distill_mcq_elimination.py --limit 12    # sample
    python distill_mcq_elimination.py --concurrency 8
"""
from __future__ import annotations
import argparse, concurrent.futures as cf, json, re, sys, threading, time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
_COLAB = HERE.parent
DATA = _COLAB / "data"
if str(_COLAB) not in sys.path:
    sys.path.insert(0, str(_COLAB))
import slm_core as S

MCQ = {"lsat_lr", "logiqa"}


def gl(g):
    m = re.match(r"\(([A-Z])\)", g or ""); return m.group(1) if m else None


def n_opts(prompt):
    o = re.findall(r"^\(([A-Z])\)\s", prompt, re.M); return len(o) if o else 5


def letters_in_body(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    body = " ".join(parts[:-1]) if len(parts) > 1 else ""
    return len(re.findall(r"\(([A-Ea-e])\)", body)) + len(re.findall(r"\boption\s+[A-Ea-e]\b", body, re.I))


ELIM = ("\n\nExplain your answer in this structure, concisely: (1) name the argument's core "
        "gap / required assumption / flaw in one sentence; (2) briefly rule out the other "
        "candidate answers by their CONTENT — one short clause each on why it is wrong; "
        "(3) commit. Refer to options by CONTENT, never by letter, and do NOT write any option "
        "letter (A, B, C, D, E) or the word \"option\" until the very end. End with EXACTLY: "
        "\"Therefore, the correct option is ({L}).\"")


def blind_prompt(row):
    return row["prompt"] + ELIM.format(L="X")


def answer_prompt(row):
    L = gl(row["gold"])
    return (row["prompt"] + f"\n\n(For reference, the correct option is {L}.) Reason as if "
            "solving it — do not mention the answer was given." + ELIM.format(L=L))


def verify(text, row):
    if not text:
        return False
    idx = S.parse_letter(text, n_opts(row["prompt"]))
    if idx is None or chr(65 + idx) != gl(row["gold"]):
        return False
    if letters_in_body(text) != 0:            # letter isolated to the final sentence
        return False
    if len(text.strip()) < 250:               # proxy: elimination present (not a one-liner)
        return False
    return True


def strip_meta(text):
    out = [ln for ln in text.splitlines()
           if not re.search(r"\b(as (given|provided)|for reference|the answer was)\b", ln, re.I)]
    return "\n".join(out).strip()


def process_row(row, opus, timeout):
    b = S.call_agent(blind_prompt(row), opus, timeout, max_tokens=500)     # tier 1: blind elim
    if verify(b, row):
        return {**row, "completion": strip_meta(b), "trace_source": "elim", "teacher": "opus"}, "blind"
    a = S.call_agent(answer_prompt(row), opus, timeout, max_tokens=500)    # tier 2: answer-conditioned
    if verify(a, row):
        return {**row, "completion": strip_meta(a), "trace_source": "elim", "teacher": "opus"}, "answer"
    return {**row, "elim": False}, ("no_teacher" if b is None and a is None else "verify_failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(DATA / "sft_train_short.jsonl"))
    ap.add_argument("--out", default=str(DATA / "mcq_elim_traces.jsonl"))
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.inp).read_text().splitlines() if l.strip()]
    todo = [r for r in rows if r.get("family") in MCQ and r.get("trace_source") in ("blind", "backfill", "elim")]
    if args.limit:
        todo = todo[:args.limit]
    cache_path = Path(args.out).with_suffix(".cache.jsonl")
    cache = {}
    if cache_path.exists():
        for l in cache_path.read_text().splitlines():
            if l.strip():
                c = json.loads(l); cache[c["id"]] = c
    pending = [r for r in todo if r["id"] not in cache]
    print(f"[elim] {len(todo)} lsat_lr+logiqa frontier | {len(cache)} cached | {len(pending)} to do "
          f"| model {args.model} | concurrency {args.concurrency}", flush=True)
    stats = Counter({"cached": len(cache)}); lock = threading.Lock(); t0 = time.time()
    try: S._client()
    except Exception: pass
    with cache_path.open("a", encoding="utf-8") as fh:
        with cf.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
            futs = {ex.submit(process_row, r, args.model, args.timeout): r for r in pending}
            done = 0
            for fut in cf.as_completed(futs):
                try: rec, stat = fut.result()
                except Exception: rec, stat = None, "error"
                with lock:
                    stats[stat] += 1
                    if rec is not None:
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush(); cache[rec["id"]] = rec
                    done += 1
                    if done % 50 == 0 or done == len(pending):
                        print(f"  {done}/{len(pending)}  {dict(stats)}  ({done/max(1e-6,time.time()-t0):.2f}/s)", flush=True)
    out = [cache[r["id"]] for r in todo if r["id"] in cache]
    with Path(args.out).open("w", encoding="utf-8") as fh:
        for r in out: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    elim = sum(1 for r in out if r.get("trace_source") == "elim")
    print(f"\n[elim] wrote {len(out)} -> {args.out} | elim-distilled {elim} | kept-original {len(out)-elim} | {dict(stats)}")


if __name__ == "__main__":
    raise SystemExit(main())
