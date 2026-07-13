#!/usr/bin/env python3
"""Ensemble-disagreement noise vote for LogiQA (bad-gold filter).

LogiQA 2.0 is machine-translated from a Chinese civil-service exam and carries gold-label
noise. This gates the AMBIGUOUS rows — the ones no frontier model committed to blind in
Step-B (trace_source grounded_template/deterministic) — with an explicit 3-model blind vote:
Haiku 4.5 + Sonnet 4.6 + Opus 4.8 each answer the MCQ blind and rate confidence. A row is
DROPPED only if a MAJORITY (>=2 of 3) CONFIDENTLY disagree with the gold letter (confidence
!= low) — i.e. multiple strong models independently pick a different answer, the signature of
bad gold. Conservative by construction: any row where >=2 models can't confidently agree on a
non-gold answer is KEPT. (Rows a frontier model already solved-to-gold blind are clean by that
fact and are NOT sent here.)

Read-only w.r.t. the corpus; writes only the vote cache + a decisions file. Resumable.

    python noise_vote_logiqa.py --in data/_noise_candidates.jsonl --out data/logiqa_noise_votes.jsonl
"""
import argparse, concurrent.futures as cf, json, re, sys, threading, time
from collections import Counter
from pathlib import Path
HERE = Path(__file__).resolve().parent; _COLAB = HERE.parent; DATA = _COLAB / "data"
sys.path.insert(0, str(_COLAB)); import slm_core as S

MODELS = [("haiku", "claude-haiku-4-5"), ("sonnet", "claude-sonnet-4-6"), ("opus", "claude-opus-4-8")]


def gold_letter(gold): m = re.match(r"\(([A-Z])\)", gold or ""); return m.group(1) if m else None
def n_options(prompt): o = re.findall(r"^\(([A-Z])\)\s", prompt, re.M); return len(o) if o else 5


def vote_prompt(row):
    # cheaper models ignore a "letter only" instruction and start reasoning; let them reason
    # briefly then commit on a parseable final line (mirrors the Step-B blind solve that works).
    return (row["prompt"] + "\n\nReason in AT MOST 2 sentences, then end with EXACTLY one final "
            "line formatted: 'FINAL: (X) | CONFIDENCE: high|medium|low' — where X is the option "
            "letter you believe is correct.")


def parse_vote(text, n):
    """-> (letter or None, confidence in {high,medium,low}). Prefer the 'FINAL: (X)' line;
    fall back to the last committing letter. A produced-but-unrated answer defaults to medium
    (NOT low) so a clear pick still counts; only an explicit 'low' or no-answer is low."""
    if not text:
        return None, "low"
    m = re.search(r"FINAL:\s*\(?\s*([A-Za-z])\s*\)?", text)
    letter = m.group(1).upper() if m else None
    if letter is None:
        idx = S.parse_letter(text, n)
        letter = chr(65 + idx) if idx is not None else None
    if letter is not None and not (0 <= ord(letter) - 65 < n):
        letter = None
    conf = "medium" if letter else "low"
    cm = re.search(r"CONFIDENCE[:\s]*([A-Za-z]+)", text, re.I) or re.search(r"\b(high|medium|low)\b", text, re.I)
    if cm and cm.group(1).lower() in ("high", "medium", "low"):
        conf = cm.group(1).lower()
    return letter, conf


def process_row(row, timeout):
    gl = gold_letter(row["gold"]); n = n_options(row["prompt"]); p = vote_prompt(row)
    votes = {}
    for name, model in MODELS:
        raw = S.call_agent(p, model, timeout, max_tokens=600)
        letter, conf = parse_vote(raw, n)
        votes[name] = {"letter": letter, "conf": conf, "raw": (raw or "")[-160:],
                       "disagree": (letter is not None and letter != gl),
                       "confident_disagree": (letter is not None and letter != gl and conf != "low")}
    n_conf_dis = sum(1 for v in votes.values() if v["confident_disagree"])
    n_agree = sum(1 for v in votes.values() if v["letter"] == gl)
    drop = n_conf_dis >= 2
    return {"id": row["id"], "gold_letter": gl, "votes": votes,
            "n_confident_disagree": n_conf_dis, "n_agree": n_agree, "drop": drop}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default=str(DATA / "logiqa_noise_votes.jsonl"))
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.inp).read_text().splitlines() if l.strip()]
    cache_path = Path(args.out).with_suffix(".cache.jsonl")
    cache = {}
    if cache_path.exists():
        for l in cache_path.read_text().splitlines():
            if l.strip():
                r = json.loads(l); cache[r["id"]] = r
    pending = [r for r in rows if r["id"] not in cache]
    print(f"[noise-vote] {len(rows)} rows | {len(cache)} cached | {len(pending)} to do | "
          f"models {[m[1] for m in MODELS]} | concurrency {args.concurrency}", flush=True)
    stats = Counter(); lock = threading.Lock(); t0 = time.time()
    try: S._client()
    except Exception: pass
    with cache_path.open("a", encoding="utf-8") as cfh:
        with cf.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
            futs = {ex.submit(process_row, r, args.timeout): r for r in pending}
            done = 0
            for fut in cf.as_completed(futs):
                try: rec = fut.result()
                except Exception: rec = None
                with lock:
                    if rec is not None:
                        cfh.write(json.dumps(rec, ensure_ascii=False) + "\n"); cfh.flush()
                        cache[rec["id"]] = rec; stats["drop" if rec["drop"] else "keep"] += 1
                    done += 1
                    if done % 25 == 0 or done == len(pending):
                        print(f"  {done}/{len(pending)}  keep={stats['keep']} drop={stats['drop']}  "
                              f"({done/max(1e-6,time.time()-t0):.1f}/s)", flush=True)
    out = [cache[r["id"]] for r in rows if r["id"] in cache]
    with Path(args.out).open("w", encoding="utf-8") as fh:
        for r in out: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    kept = sum(1 for r in out if not r["drop"]); dropped = sum(1 for r in out if r["drop"])
    print(f"\n[noise-vote] wrote {len(out)} decisions -> {args.out}")
    print(f"  KEEP {kept} | DROP {dropped} ({100*dropped/max(1,len(out)):.1f}% dropped as bad-gold)")
    print(f"  agreement dist (n models = gold): {dict(Counter(r['n_agree'] for r in out))}")


if __name__ == "__main__":
    main()
