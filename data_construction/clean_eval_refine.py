#!/usr/bin/env python3
"""Refine the confirmed-bad set with a FAIR (steelman) non-blind Opus pass.

The first adjudication used a fault-seeking 'audit' prompt that over-confirmed debatable
items (e.g. logiqa-9472, a subjective 'best title' item whose gold is defensible). Here we
re-judge the 16 with a NEUTRAL prompt that steelmans the key, two passes. An item stays
CONFIRMED (replacement kept) only if BOTH fair passes conclude key-wrong; otherwise it is
RESCUED (original restored). Rebuilds data/eval_items_clean.json accordingly.
"""
import concurrent.futures as cf, json, re, sys, threading
from pathlib import Path
HERE = Path(__file__).resolve().parent; _COLAB = HERE.parent; DATA = _COLAB / "data"
sys.path.insert(0, str(_COLAB)); import slm_core as S
OPUS = "claude-opus-4-8"

orig = {it["id"]: it for it in json.load(open(DATA/"eval_items.json"))}
clean_now = {it["id"]: it for it in json.load(open(DATA/"eval_items_clean.json"))}
manifest = json.load(open(DATA/"eval_clean_manifest.json"))          # [{removed, added, ...}]
repl_by_removed = {m["removed"]: clean_now[m["added"]] for m in manifest}   # replacement item objects
confirmed_ids = [m["removed"] for m in manifest]
print(f"re-judging {len(confirmed_ids)} confirmed-bad items with a fair steelman pass", flush=True)

def gl(g): m = re.match(r"\(([A-Z])\)", g or ""); return m.group(1) if m else None

def fair_judge(iid, timeout=150):
    it = orig[iid]; prompt = S.build_prompt(it, it.get("mode","mcq")); gold = gl(S.credited_answer(it, it.get("mode","mcq")))
    base = (prompt + f"\n\nThe published answer key says ({gold}). STEELMAN the key: give the strongest case "
            f"that ({gold}) is a defensible/correct answer, then the case against. For subjective item types "
            f"(best title / main point / most appropriate), weigh how such questions are conventionally graded. "
            f"Only judge the key WRONG if a different option is CLEARLY better, not merely arguable. End with:\n"
            f"VERDICT: key-defensible | key-wrong\nBEST: (X)")
    def one():
        r = S.call_agent(base, OPUS, timeout, max_tokens=1000) or ""
        vm = re.search(r"VERDICT:\s*(key-defensible|key-wrong)", r, re.I)
        bm = re.search(r"BEST:\s*\(?\s*([A-Za-z])", r)
        return (vm.group(1).lower() if vm else None), (bm.group(1).upper() if bm else None)
    v1, b1 = one(); v2, b2 = one()
    still_bad = (v1 == "key-wrong" and v2 == "key-wrong")
    return {"id": iid, "gold": gold, "p1": [v1, b1], "p2": [v2, b2], "still_confirmed": still_bad}

lock = threading.Lock(); res = []
with cf.ThreadPoolExecutor(max_workers=6) as ex:
    for r in ex.map(fair_judge, confirmed_ids):
        with lock: res.append(r)
res.sort(key=lambda r: r["id"])

still = [r for r in res if r["still_confirmed"]]; rescued = [r for r in res if not r["still_confirmed"]]
print(f"\nAfter fair re-judge: STILL confirmed {len(still)} | RESCUED {len(rescued)}")
print("STILL confirmed (replacement kept):")
for r in still: print(f"  {r['id']} gold({r['gold']}) p1{r['p1']} p2{r['p2']}")
print("RESCUED (original restored):")
for r in rescued: print(f"  {r['id']} gold({r['gold']}) p1{r['p1']} p2{r['p2']}")

# rebuild: start from originals; apply replacement ONLY for still-confirmed
still_ids = {r["id"] for r in still}
new_ev = [ (repl_by_removed[it["id"]] if it["id"] in still_ids else it) for it in json.load(open(DATA/"eval_items.json")) ]
Path(DATA/"eval_items_clean.json").write_text(json.dumps(new_ev, ensure_ascii=False, indent=0))
new_manifest = [m for m in manifest if m["removed"] in still_ids]
Path(DATA/"eval_clean_manifest.json").write_text(json.dumps(new_manifest, ensure_ascii=False, indent=2))
print(f"\nrebuilt data/eval_items_clean.json: {len(still_ids)} swaps kept, {len(rescued)} reverted to original")
