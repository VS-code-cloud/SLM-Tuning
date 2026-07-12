#!/usr/bin/env python3
"""Phase 3 (SEPARATE artifact). The SLM trails Opus on logicnli True/False (its neutral
already beats Opus), and those traces are terse (~262 chars). This EXPANDS each existing,
z3/forward-chain-VERIFIED derivation into explicit inference steps via a cheap model —
without changing the logic or the committed label. Low-risk: it's grounded in the already
-correct trace ("make each step explicit, do not change the logic or conclusion"), and any
result that doesn't still commit to gold is discarded (keep the original solver trace).

Reads the shortened corpus read-only; writes data/logicnli_deep_traces.jsonl (NOT the
`*_stepb_review.jsonl` glob). Only logicnli True/False (non-neutral) rows. Resumable.
"""
from __future__ import annotations
import argparse, concurrent.futures as cf, json, re, sys, threading, time
from collections import Counter
from pathlib import Path
HERE = Path(__file__).resolve().parent; _COLAB = HERE.parent; DATA = _COLAB / "data"
if str(_COLAB) not in sys.path: sys.path.insert(0, str(_COLAB))
import slm_core as S


def is_tf(r):
    return r["family"] == "logicnli" and (S.canon_label(r["gold"]) or r["gold"]) not in S.NEUTRAL


EXPAND = ("Rewrite the following CORRECT logical derivation to make each inference step "
          "explicit and easy to follow (e.g. 'From X and the rule that Y, it follows that Z; "
          "then ...'). Do NOT change the logic, the premises used, or the final conclusion. "
          "Keep it tight (no padding) and end with the SAME final sentence it already has.\n\n"
          "--- derivation ---\n")


def verify(text, row):
    if not text or len(text.strip()) <= len(row["completion"]):   # must be at least as explicit
        return len(text.strip()) >= len(row["completion"]) * 0.8 and _label_ok(text, row)
    return _label_ok(text, row)


def _label_ok(text, row):
    gold = S.canon_label(row["gold"]) or row["gold"]
    return S.labels_equiv(S.parse_label(text), gold)


def process_row(row, model, timeout):
    r = S.call_agent(EXPAND + row["completion"], model, timeout, max_tokens=400)
    if r and _label_ok(r, row) and len(r.strip()) >= 0.9 * len(row["completion"]):
        return {**row, "completion": r.strip(), "trace_source": "solver_deep"}, "deepened"
    return {**row, "deep": False}, ("no_teacher" if r is None else "verify_failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(DATA / "sft_train_short.jsonl"))
    ap.add_argument("--out", default=str(DATA / "logicnli_deep_traces.jsonl"))
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=90)
    args = ap.parse_args()
    rows = [json.loads(l) for l in Path(args.inp).read_text().splitlines() if l.strip()]
    todo = [r for r in rows if is_tf(r)]
    if args.limit: todo = todo[:args.limit]
    cache_path = Path(args.out).with_suffix(".cache.jsonl"); cache = {}
    if cache_path.exists():
        for l in cache_path.read_text().splitlines():
            if l.strip(): c = json.loads(l); cache[c["id"]] = c
    pending = [r for r in todo if r["id"] not in cache]
    print(f"[deepen] {len(todo)} logicnli T/F | {len(cache)} cached | {len(pending)} to do "
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
                    if done % 100 == 0 or done == len(pending):
                        print(f"  {done}/{len(pending)}  {dict(stats)}  ({done/max(1e-6,time.time()-t0):.2f}/s)", flush=True)
    out = [cache[r["id"]] for r in todo if r["id"] in cache]
    with Path(args.out).open("w", encoding="utf-8") as fh:
        for r in out: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    dp = sum(1 for r in out if r.get("trace_source") == "solver_deep")
    print(f"\n[deepen] wrote {len(out)} -> {args.out} | deepened {dp} | kept-original {len(out)-dp} | {dict(stats)}")


if __name__ == "__main__":
    raise SystemExit(main())
