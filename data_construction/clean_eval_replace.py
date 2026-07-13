#!/usr/bin/env python3
"""Replace the confirmed-bad eval-logiqa items with fresh, verified-clean questions.

For each id in eval_confirmed_bad.jsonl (confirmed_bad=True), pull a replacement from
LogiQA-2.0 train/dev that is (a) valid 4-option, (b) NOT in ReClor, (c) NOT in any corpus
(base test, the v6 +300 new pull, or the current eval), and (d) verified-clean gold: all
three models (Haiku+Sonnet+Opus) independently agree with the gold (n_agree==3). Preserves
eval position/schema. Writes a NEW file data/eval_items_clean.json (original untouched) +
a manifest. Re-running frontier + SLM evals on the new file is required for comparability.
"""
import concurrent.futures as cf, hashlib, json, sys, threading
from pathlib import Path
HERE = Path(__file__).resolve().parent; _COLAB = HERE.parent; RAW = _COLAB/"data"/"_raw"; DATA = _COLAB/"data"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(_COLAB))
import build_corpus as B, slm_core as S
from noise_vote_logiqa import process_row  # 3-model vote -> n_agree

confirmed = [json.loads(l)["id"] for l in (DATA/"eval_confirmed_bad.jsonl").read_text().splitlines()
             if l.strip() and json.loads(l)["confirmed_bad"]]
N = len(confirmed)
print(f"confirmed-bad to replace: {N}", flush=True)

def load(fn):
    return [json.loads(l) for l in (RAW/fn).read_text(encoding="utf-8").splitlines() if l.strip()]
def valid(d):
    o, a = d.get("options") or [], d.get("answer")
    t, q = B.norm_ws(d.get("text","")), B.norm_ws(d.get("question",""))
    return bool(t) and bool(q) and isinstance(a, int) and len(o) == 4 and a < 4

# exclusion set: ReClor + base test + v6 new pull + current eval stimuli
rc = B._reclor_keys()
excl = set(rc)
for d in load("logiqa2_test.txt"): excl.add(B._norm_key(B.norm_ws(d.get("text",""))))
ev_all = json.load(open(DATA/"eval_items.json"))
for it in ev_all:
    if it.get("family") == "logiqa": excl.add(B._norm_key(B.norm_ws(it.get("stimulus",""))))
for l in (DATA/"logiqa_new_raw.jsonl").read_text().splitlines():
    if l.strip(): excl.add(B._norm_key(B.norm_ws(json.loads(l).get("prompt","").split("choose the SINGLE best option.\n\n",1)[-1].split("\n\nQuestion:",1)[0])))
# (also exclude by raw text of the v6 pull, robustly, from logiqa_new_short stimuli)

# clean candidates, deterministic order
cand = []; seen = set()
for split, fn in (("tr","logiqa2_train.txt"), ("dv","logiqa2_dev.txt")):
    for d in load(fn):
        if not valid(d): continue
        k = B._norm_key(B.norm_ws(d.get("text","")))
        if k in excl or k in seen: continue
        seen.add(k)
        text, q = B.norm_ws(d.get("text","")), B.norm_ws(d.get("question",""))
        opts, ans = d["options"], int(d["answer"])
        cand.append({"raw_id": d.get("id","x"), "split": split, "text": text, "q": q, "opts": opts, "ans": ans,
                     "id": f"logiqa-evalclean-{split}-{d.get('id','x')}",
                     "prompt": None, "gold": None})
cand.sort(key=lambda c: hashlib.sha1(f"evc:{c['id']}".encode()).hexdigest())
print(f"clean candidates available: {len(cand)}", flush=True)

def as_item(c):
    return {"id": c["id"], "group_id": f"logiqa-ctx-{hashlib.sha1(c['text'].encode()).hexdigest()[:12]}",
            "family": "logiqa", "task_type": "inference", "difficulty": "hard", "mode": "mcq",
            "stimulus": c["text"], "question": c["q"], "mc_question": c["q"],
            "mc_choices": [B.norm_ws(str(o)) for o in c["opts"]], "mc_credited_index": c["ans"],
            "reference_answer": B.norm_ws(str(c["opts"][c["ans"]])), "source": f"LogiQA 2.0 {c['split']} [eval-clean repl]"}

# vote candidates in deterministic order until we have N with n_agree==3
clean_repls = []; lock = threading.Lock(); idx = 0; BATCH = 8
def vote_one(c):
    it = as_item(c)
    row = {"id": it["id"], "prompt": S.build_prompt(it, "mcq"), "gold": S.credited_answer(it, "mcq")}
    rec = process_row(row, 120)
    return c, it, rec["n_agree"]

while len(clean_repls) < N and idx < len(cand):
    batch = cand[idx: idx+BATCH]; idx += BATCH
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        for c, it, nag in ex.map(vote_one, batch):
            if nag == 3 and len(clean_repls) < N:
                clean_repls.append(it)
    print(f"  voted {idx} candidates -> {len(clean_repls)}/{N} clean replacements", flush=True)

if len(clean_repls) < N:
    print(f"WARNING: only found {len(clean_repls)} clean replacements for {N} bad items"); N = len(clean_repls)

# swap by position in the full eval list
repl_map = dict(zip(confirmed[:N], clean_repls[:N]))
new_ev = []
for it in ev_all:
    if it.get("id") in repl_map:
        new_ev.append(repl_map[it["id"]])
    else:
        new_ev.append(it)
Path(DATA/"eval_items_clean.json").write_text(json.dumps(new_ev, ensure_ascii=False, indent=0))
manifest = [{"removed": rid, "added": repl_map[rid]["id"], "new_gold": repl_map[rid]["reference_answer"][:60]}
            for rid in confirmed[:N]]
Path(DATA/"eval_clean_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
print(f"\nwrote data/eval_items_clean.json ({len(new_ev)} items; {N} logiqa swapped) + eval_clean_manifest.json")
print("swaps:")
for m in manifest: print(f"  - {m['removed']} -> {m['added']}")
