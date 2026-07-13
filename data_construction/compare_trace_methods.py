#!/usr/bin/env python3
"""A/B trace-quality ablation for concise MCQ traces, judged by Opus.

Two ways to produce a SHORT MCQ trace from what we already have:
  (A) Haiku SHORTENS the verbose frontier Step-B trace (real reasoning, compressed).
  (B) Haiku REWORDS the deterministic grounded template (grounded, but shallow).
Both are normalized to the same committing final sentence + letter-isolation invariant, so the
only thing that differs is the reasoning body. Opus then judges, blinded and order-randomized,
which is (1) more logically SOUND and (2) less FLUFFy — and an OVERALL winner.

Purpose: decide which source makes the better concise MCQ trace before rebuilding the corpus's
MCQ completions. Read-only; writes data/trace_ab_results.jsonl. Resumable (per-id cache).

    python compare_trace_methods.py --n 20                 # 20 per MCQ family
    python compare_trace_methods.py --families logiqa --n 40
"""
import argparse, concurrent.futures as cf, hashlib, json, re, sys, threading, time
from collections import Counter, defaultdict
from pathlib import Path
HERE = Path(__file__).resolve().parent; _COLAB = HERE.parent; DATA = _COLAB / "data"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(_COLAB))
import build_corpus as B
import slm_core as S

HAIKU = "claude-haiku-4-5"; OPUS = "claude-opus-4-8"
FINAL = "Therefore, the correct option is ({})."


def gold_letter(g): m = re.match(r"\(([A-Z])\)", g or ""); return m.group(1) if m else None
def n_options(p): o = re.findall(r"^\(([A-Z])\)\s", p, re.M); return len(o) if o else 5
def letters_in_body(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    body = " ".join(parts[:-1]) if len(parts) > 1 else ""
    return len(re.findall(r"\(([A-Ea-e])\)", body)) + len(re.findall(r"\boption\s+[A-Ea-e]\b", body, re.I))
def grade_ok(text, row):
    idx = S.parse_letter(text, n_options(row["prompt"]))
    return text and idx is not None and chr(65 + idx) == gold_letter(row["gold"]) and letters_in_body(text) == 0


def shorten_prompt(frontier, L):
    return ("Compress the following solution to a multiple-choice logic question into AT MOST 4 short "
            "sentences of reasoning, then the final committing sentence.\nRules:\n"
            "- Keep the essential logic; delete restatement of the passage/options and any hedging.\n"
            "- Refer to options by CONTENT, not letter. Do NOT write any option letter or the word "
            "\"option\" anywhere except the final sentence.\n"
            f"- End with EXACTLY: \"{FINAL.format(L)}\"\nOutput ONLY the compressed solution.\n\n"
            "--- solution ---\n" + frontier)


def reword_prompt(template, L):
    return ("Rewrite the following solution to a multiple-choice logic question as a natural, concise "
            "chain of reasoning (AT MOST 4 short sentences), then the final committing sentence.\nRules:\n"
            "- Keep the logic and the same final answer; do not invent facts not in the solution.\n"
            "- Refer to options by CONTENT, not letter. Do NOT write any option letter or the word "
            "\"option\" anywhere except the final sentence.\n"
            f"- End with EXACTLY: \"{FINAL.format(L)}\"\nOutput ONLY the rewritten solution.\n\n"
            "--- solution ---\n" + template)


def judge_prompt(row, t1, t2):
    L = gold_letter(row["gold"])
    q = row["prompt"].split("choose the SINGLE best option.\n\n", 1)[-1]
    return ("You are comparing two candidate written solutions to the SAME multiple-choice logic "
            "question. Both reach the same, correct answer. Judge which is the better solution on:\n"
            "(1) LOGICAL SOUNDNESS — does the reasoning actually justify the credited answer with valid, "
            "on-point logic (not just assert it)?\n"
            "(2) FLUFF — penalize hedging, restating the passage/options, filler, vagueness.\n\n"
            f"=== QUESTION ===\n{q}\n\nCredited answer: ({L})\n\n"
            f"=== Trace 1 ===\n{t1}\n\n=== Trace 2 ===\n{t2}\n\n"
            "Reply EXACTLY in this format (no other text):\n"
            "SOUND: <1|2|tie>\nFLUFF: <1|2|tie>   (which is LESS fluffy)\nOVERALL: <1|2|tie>\n"
            "REASON: <one sentence>")


def parse_judge(text):
    def g(key):
        m = re.search(rf"{key}:\s*(1|2|tie)", text or "", re.I); return m.group(1).lower() if m else None
    rm = re.search(r"REASON:\s*(.+)", text or "", re.I | re.S)
    return g("SOUND"), g("FLUFF"), g("OVERALL"), (rm.group(1).strip()[:240] if rm else "")


def process(item, frontier_row, timeout):
    L = gold_letter(frontier_row["gold"])
    A = (S.call_agent(shorten_prompt(frontier_row["completion"], L), HAIKU, timeout, max_tokens=400) or "").strip()
    B_ = (S.call_agent(reword_prompt(B.verbalize(item), L), HAIKU, timeout, max_tokens=400) or "").strip()
    if not A or not B_:
        return None
    gA, gB = grade_ok(A, frontier_row), grade_ok(B_, frontier_row)
    # blinded, order-randomized by id hash (no RNG -> reproducible)
    a_is_1 = int(hashlib.sha1(item["id"].encode()).hexdigest()[:8], 16) % 2 == 0
    t1, t2 = (A, B_) if a_is_1 else (B_, A)
    sound, fluff, overall, reason = parse_judge(S.call_agent(judge_prompt(frontier_row, t1, t2), OPUS, timeout, max_tokens=300))
    def unmap(v):  # judge's 1/2 -> 'A'/'B'
        if v not in ("1", "2"): return v
        return ("A" if v == "1" else "B") if a_is_1 else ("B" if v == "1" else "A")
    return {"id": item["id"], "family": item["family"], "gradeA": gA, "gradeB": gB,
            "lenA": len(A), "lenB": len(B_), "sound": unmap(sound), "fluff": unmap(fluff),
            "overall": unmap(overall), "reason": reason, "A": A, "B": B_}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="items per family")
    ap.add_argument("--families", default="lsat_lr,logiqa,arct")
    ap.add_argument("--out", default=str(DATA / "trace_ab_results.jsonl"))
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()
    fams = args.families.split(",")

    items = {it["id"]: it for it in (B.parse_reclor() + B.parse_logiqa() + B.parse_arct())}
    rev = [json.loads(l) for l in (DATA / "mcq_stepb_review.jsonl").read_text().splitlines() if l.strip()]
    frontier = {r["id"]: r for r in rev if r.get("trace_source") in ("blind", "backfill") and r["id"] in items}
    # deterministic sample n per family
    sample = []
    for fam in fams:
        ids = sorted([i for i, r in frontier.items() if r["family"] == fam],
                     key=lambda i: hashlib.sha1(f"ab:{i}".encode()).hexdigest())[:args.n]
        sample += ids
    cache_path = Path(args.out).with_suffix(".cache.jsonl"); cache = {}
    if cache_path.exists():
        for l in cache_path.read_text().splitlines():
            if l.strip(): r = json.loads(l); cache[r["id"]] = r
    pending = [i for i in sample if i not in cache]
    print(f"[trace-ab] sample {len(sample)} ({dict(Counter(frontier[i]['family'] for i in sample))}) | "
          f"{len(cache)} cached | {len(pending)} to do | concurrency {args.concurrency}", flush=True)
    lock = threading.Lock(); t0 = time.time(); done = 0
    try: S._client()
    except Exception: pass
    with cache_path.open("a", encoding="utf-8") as cfh:
        with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {ex.submit(process, items[i], frontier[i], args.timeout): i for i in pending}
            for fut in cf.as_completed(futs):
                try: rec = fut.result()
                except Exception: rec = None
                with lock:
                    if rec: cfh.write(json.dumps(rec, ensure_ascii=False) + "\n"); cfh.flush(); cache[rec["id"]] = rec
                    done += 1
                    if done % 20 == 0 or done == len(pending):
                        print(f"  {done}/{len(pending)} ({done/max(1e-6,time.time()-t0):.1f}/s)", flush=True)

    out = [cache[i] for i in sample if i in cache]
    with Path(args.out).open("w", encoding="utf-8") as fh:
        for r in out: fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    def tally(key, rows):
        c = Counter(r[key] for r in rows); return f"A={c['A']} B={c['B']} tie={c['tie']}"
    print(f"\n[trace-ab] {len(out)} judged. A = Haiku-shortened-frontier | B = Haiku-reworded-template")
    print(f"  OVERALL : {tally('overall', out)}")
    print(f"  SOUND   : {tally('sound', out)}")
    print(f"  FLUFF(less): {tally('fluff', out)}")
    for fam in fams:
        fr = [r for r in out if r["family"] == fam]
        if fr: print(f"    {fam:8s} overall {tally('overall', fr)}  sound {tally('sound', fr)}  fluff {tally('fluff', fr)}")
    import statistics
    print(f"  avg chars: A {int(statistics.mean([r['lenA'] for r in out]))}  B {int(statistics.mean([r['lenB'] for r in out]))}")
    print(f"  grade-ok: A {sum(r['gradeA'] for r in out)}/{len(out)}  B {sum(r['gradeB'] for r in out)}/{len(out)}")
    print("\n  sample reasons:")
    for r in out[:6]:
        print(f"    [{r['family']}] overall={r['overall']} sound={r['sound']} fluff={r['fluff']}: {r['reason']}")


if __name__ == "__main__":
    main()
