#!/usr/bin/env python3
"""Rigorously confirm which UNANIMOUS-quick-vote eval-logiqa flags are genuinely bad gold.

The quick vote over-flags (~1/3 false positives), so for the 26 items where all 3 models
picked the SAME non-gold letter, run TWO independent careful Opus passes (solve + audit).
An item is CONFIRMED BAD only if BOTH passes conclude the key is wrong AND agree on the same
alternative letter. Writes data/eval_confirmed_bad.jsonl. No eval mutation here.
"""
import concurrent.futures as cf, json, re, sys, threading, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; _COLAB = HERE.parent; DATA = _COLAB / "data"
sys.path.insert(0, str(_COLAB)); import slm_core as S
OPUS = "claude-opus-4-8"

votes = {json.loads(l)["id"]: json.loads(l) for l in (DATA/"eval_logiqa_votes.jsonl").read_text().splitlines() if l.strip()}
ev = {it["id"]: it for it in json.load(open(DATA/"eval_items.json")) if it.get("family") == "logiqa"}
def letters(v): return [v["votes"][m]["letter"] for m in ("haiku","sonnet","opus")]
unanimous = [i for i, v in votes.items()
             if all(letters(v)) and len(set(letters(v))) == 1 and letters(v)[0] != v["gold_letter"]]
print(f"unanimous-flagged eval logiqa: {len(unanimous)}", flush=True)

def n_opt(p): o = re.findall(r"^\(([A-Z])\)\s", p, re.M); return len(o) if o else 5
def gl(g): m = re.match(r"\(([A-Z])\)", g or ""); return m.group(1) if m else None

def adjudicate(iid, timeout=120):
    it = ev[iid]; prompt = S.build_prompt(it, it.get("mode","mcq")); gold = gl(S.credited_answer(it, it.get("mode","mcq")))
    def one(instr):
        r = S.call_agent(prompt + instr, OPUS, timeout, max_tokens=900) or ""
        bm = re.search(r"BEST:\s*\(?\s*([A-Za-z])", r); kw = re.search(r"KEY_WRONG:\s*(yes|no)", r, re.I)
        return (bm.group(1).upper() if bm else None), (kw.group(1).lower() if kw else None), r[-200:]
    p1 = one(f"\n\nSolve this INDEPENDENTLY. Reason step by step, then output exactly:\nBEST: (X)\n"
             f"KEY_WRONG: yes|no   (the published key says ({gold}); say yes only if you are confident it is wrong)")
    p2 = one(f"\n\nCritically AUDIT whether the published answer key ({gold}) is correct, or whether a "
             f"different option is clearly better. Reason step by step, then output exactly:\nBEST: (X)\nKEY_WRONG: yes|no")
    confirmed = (p1[1] == "yes" and p2[1] == "yes" and p1[0] and p1[0] == p2[0] and p1[0] != gold)
    return {"id": iid, "gold": gold, "pass1": {"best": p1[0], "key_wrong": p1[1]},
            "pass2": {"best": p2[0], "key_wrong": p2[1]}, "confirmed_bad": confirmed,
            "alt": p1[0] if confirmed else None}

lock = threading.Lock(); out = []; t0 = time.time(); done = 0
with cf.ThreadPoolExecutor(max_workers=6) as ex:
    futs = {ex.submit(adjudicate, i): i for i in unanimous}
    for fut in cf.as_completed(futs):
        try: rec = fut.result()
        except Exception: rec = None
        with lock:
            if rec: out.append(rec)
            done += 1
            if done % 10 == 0 or done == len(unanimous):
                print(f"  {done}/{len(unanimous)} ({done/max(1e-6,time.time()-t0):.1f}/s)", flush=True)

out.sort(key=lambda r: r["id"])
with (DATA/"eval_confirmed_bad.jsonl").open("w", encoding="utf-8") as fh:
    for r in out: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
conf = [r for r in out if r["confirmed_bad"]]
print(f"\nCONFIRMED BAD (both Opus passes agree key wrong, same alt): {len(conf)} / {len(out)}")
for r in conf:
    it = ev[r["id"]]
    print(f"  {r['id']}: gold ({r['gold']}) -> Opus ({r['alt']})  | {it['stimulus'][:70]}...")
print(f"\nRescued (>=1 pass says key OK, or passes disagree): {len(out)-len(conf)}")
