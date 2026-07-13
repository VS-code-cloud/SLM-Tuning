#!/usr/bin/env python3
"""Full-20-MR LGMT consistency eval on a fixed 300-case subset, for cross-model comparison.

Builds data/lgmt300.json once (300 metamorphic groups sampled from the sibling
critical-reasoning-eval corpus of 590, deterministic) + the source items they reference, so
the SLM (Colab), Opus, and Sonnet all run the IDENTICAL cases. Scores a gateway model here
(slm_core.call_agent); the SLM runs the same file on Colab (lgmt300_slm_colab.py).

MVR = fraction of cases whose label flips vs its source under a label-preserving transform
(oracle-free). HDR = flips among source-correct items. Method mirrors lgmt_full.do_eval.

    python lgmt300_eval.py --build                       # build data/lgmt300.json
    python lgmt300_eval.py --model claude-opus-4-8        # score a gateway model
"""
import argparse, concurrent.futures as cf, hashlib, json, re, sys, threading
from collections import defaultdict
from pathlib import Path
HERE = Path(__file__).resolve().parent; DATA = HERE / "data"
CORPUS = HERE.parent.parent / "critical-reasoning-eval" / "lgmt_full_corpus.json"
sys.path.insert(0, str(HERE)); import slm_core as S

def solve_prompt(premises, conclusion):
    prem = "\n".join(f"- {p}" for p in premises)
    return ("You are an expert in logic. Based ONLY on the premises, decide whether the conclusion "
            "is True, False, or Unknown (does not deductively follow either way).\n\n"
            f"Premises:\n{prem}\n\nConclusion: {conclusion}\n\nReason briefly, then end with a line "
            "exactly:\nAnswer: <True|False|Unknown>")

def parse_label(text):
    m = list(re.finditer(r"answer\s*[:\-]\s*\*?\*?\s*(true|false|unknown|uncertain)", text or "", re.I))
    if not m: return None
    v = m[-1].group(1).lower(); return "Unknown" if v in ("unknown", "uncertain") else v.capitalize()

def build_300():
    c = json.loads(CORPUS.read_text()); items = {it["id"]: it for it in c["items"]}
    ranked = sorted(c["mgs"], key=lambda m: hashlib.sha1(f"{m['mr']}|{m['item_id']}|{m['conclusion']}".encode()).hexdigest())
    sel = ranked[:300]; sids = sorted({m["item_id"] for m in sel})
    out = {"cases": sel, "sources": {i: {"premises": items[i]["premises"], "conclusion": items[i]["conclusion"], "gold": items[i]["gold"]} for i in sids}}
    (DATA/"lgmt300.json").write_text(json.dumps(out, ensure_ascii=False))
    from collections import Counter
    print(f"[build] 300 cases from {len(sids)} sources -> data/lgmt300.json | by category {dict(Counter(m['category'] for m in sel))}")
    return out

def score(model, concurrency, timeout):
    d = json.loads((DATA/"lgmt300.json").read_text()); src, cases = d["sources"], d["cases"]
    jobs = [("src", i, src[i]["premises"], src[i]["conclusion"]) for i in src] + \
           [("mg", k, m["premises"], m["conclusion"]) for k, m in enumerate(cases)]
    res = {}; lock = threading.Lock(); done = [0]
    def run(j):
        kind, key, prem, concl = j
        lab = parse_label(S.call_agent(solve_prompt(prem, concl), model, timeout, max_tokens=700))
        with lock:
            res[(kind, key)] = lab; done[0] += 1
            if done[0] % 50 == 0: print(f"  {done[0]}/{len(jobs)}", flush=True)
    with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
        list(ex.map(run, jobs))
    srclab = {i: res.get(("src", i)) for i in src}
    per_cat = defaultdict(lambda: [0, 0]); V = H = N = accs = accn = 0
    for k, m in enumerate(cases):
        ys, yf, g = srclab.get(m["item_id"]), res.get(("mg", k)), m["gold"]
        if ys is None or yf is None: continue
        N += 1; viol = ys != yf; V += viol; H += (ys == g and viol)
        per_cat[m["category"]][0] += 1; per_cat[m["category"]][1] += viol
    # static accuracy over sources
    for i in src:
        if srclab[i] is not None: accn += 1; accs += (srclab[i] == src[i]["gold"])
    mvr = round(100*V/N, 1); hdr = round(100*H/N, 1); acc = round(100*accs/max(1, accn), 1)
    out = {"model": model, "n": N, "MVR": mvr, "HDR": hdr, "Acc_static": acc,
           "by_category": {c: {"n": v[0], "MVR": round(100*v[1]/v[0], 1)} for c, v in sorted(per_cat.items())}}
    (DATA/f"lgmt300_{model.replace('/','_')}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n[{model}] n={N} | Acc_static {acc} | MVR {mvr} | HDR {hdr}")
    for c, s in out["by_category"].items(): print(f"    {c}: n={s['n']} MVR {s['MVR']}")
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--build", action="store_true")
    ap.add_argument("--model"); ap.add_argument("--concurrency", type=int, default=6); ap.add_argument("--timeout", type=int, default=120)
    a = ap.parse_args()
    if a.build or not (DATA/"lgmt300.json").exists(): build_300()
    if a.model: score(a.model, a.concurrency, a.timeout)
