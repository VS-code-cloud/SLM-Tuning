#!/usr/bin/env python3
"""Trim the lsat_lr elimination traces (~851 -> ~700 chars) to reduce MCQ token-mass while
keeping the distractor-elimination structure. Haiku compresses each; verified to still
commit to gold with the option letter isolated to the final sentence, else keep original.
Reads data/mcq_elim_traces.jsonl read-only; writes data/mcq_elim_traces_trim.jsonl (lsat_lr).
"""
from __future__ import annotations
import argparse, concurrent.futures as cf, json, re, sys, threading, time
from collections import Counter
from pathlib import Path
HERE = Path(__file__).resolve().parent; _COLAB = HERE.parent; DATA = _COLAB / "data"
if str(_COLAB) not in sys.path: sys.path.insert(0, str(_COLAB))
import slm_core as S

def gl(g): m = re.match(r"\(([A-Z])\)", g or ""); return m.group(1) if m else None
def nop(p): return len(re.findall(r"^\(([A-Z])\)\s", p, re.M)) or 5
def letters_in_body(t):
    parts = re.split(r"(?<=[.!?])\s+", t.strip()); body = " ".join(parts[:-1]) if len(parts) > 1 else ""
    return len(re.findall(r"\(([A-Ea-e])\)", body)) + len(re.findall(r"\boption\s+[A-Ea-e]\b", body, re.I))

TRIM = ("Tighten the following multiple-choice solution to about 700 characters (~5-6 short "
        "sentences). Keep the structure: state the argument's core gap/flaw, briefly rule out "
        "the other candidate answers by their CONTENT (one clause each), then commit. Do NOT "
        "write any option letter (A-E) or the word \"option\" until the very end, which must "
        "stay EXACTLY as it is. Output only the tightened solution.\n\n--- solution ---\n")

def verify(t, row):
    if not t: return False
    idx = S.parse_letter(t, nop(row["prompt"]))
    return (idx is not None and chr(65+idx) == gl(row["gold"]) and letters_in_body(t) == 0
            and 250 <= len(t.strip()) < len(row["completion"]))

def process_row(row, model, timeout):
    r = S.call_agent(TRIM + row["completion"], model, timeout, max_tokens=350)
    if verify(r, row):
        return {**row, "completion": r.strip()}, "trimmed"
    return {**row, "completion": row["completion"]}, ("no_teacher" if r is None else "kept_original")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(DATA / "mcq_elim_traces.jsonl"))
    ap.add_argument("--out", default=str(DATA / "mcq_elim_traces_trim.jsonl"))
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=90)
    args = ap.parse_args()
    rows = [json.loads(l) for l in Path(args.inp).read_text().splitlines() if l.strip()]
    todo = [r for r in rows if r.get("trace_source") == "elim" and r.get("family") == "lsat_lr"]
    if args.limit: todo = todo[:args.limit]
    cache_path = Path(args.out).with_suffix(".cache.jsonl"); cache = {}
    if cache_path.exists():
        for l in cache_path.read_text().splitlines():
            if l.strip(): c = json.loads(l); cache[c["id"]] = c
    pending = [r for r in todo if r["id"] not in cache]
    print(f"[trim] {len(todo)} lsat_lr elim | {len(cache)} cached | {len(pending)} to do | {args.model} | c={args.concurrency}", flush=True)
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
    tr = sum(1 for r in out if len(r["completion"]) < 851)
    avg = sum(len(r["completion"]) for r in out) // max(1, len(out))
    print(f"\n[trim] wrote {len(out)} -> {args.out} | avg len {avg} chars | {dict(stats)}")

if __name__ == "__main__":
    raise SystemExit(main())
