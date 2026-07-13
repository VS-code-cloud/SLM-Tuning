#!/usr/bin/env python3
"""Pull fresh clean items to backfill the bad-gold drops: ~arct + ~logiqa.

arct: fresh arguments whose claim-id (group) is NOT already in the v8 train corpus or eval
(prevents the annotator-duplicate leakage parse_arct guards against). logiqa: fresh clean
NON-RC not in any corpus/eval/prior-pull. Emits pipeline-native template rows (to_row) to
data/_badgold_backfill_raw.jsonl. No gateway.
"""
import argparse, hashlib, json, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; _COLAB = HERE.parent; RAW = _COLAB/"data"/"_raw"; DATA = _COLAB/"data"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(_COLAB)); import build_corpus as B

RC = re.compile(r'best title|main idea|central idea|\btheme\b|mainly about|primarily about|meant to emphasize|intended to (indicate|express|convey|illustrate|emphasize)|best summar|most accurate summary|general idea|main topic|purpose of (this|the) (text|passage|article)|title for this', re.I)
EXC = re.compile(r'point (at issue|in dispute)|at issue between', re.I)
def is_rc(q): return bool(RC.search(q or "")) and not EXC.search(q or "")

ap = argparse.ArgumentParser(); ap.add_argument("--arct", type=int, default=65); ap.add_argument("--logiqa", type=int, default=25)
args = ap.parse_args()

v8 = [json.loads(l) for l in (DATA/"sft_train_v7_base.jsonl").read_text().splitlines() if l.strip()]
ev = json.load(open(DATA/"eval_items_clean.json"))

# --- arct: reconstruct used claim-ids from v8 ids + eval group_ids ---
def arct_group_from_id(i):
    m = re.match(r"arct-\w+?-(\d+_\d+)", i); return f"arct-{m.group(1)}" if m else None
used_arct = {arct_group_from_id(r["id"]) for r in v8 if r["family"]=="arct"}
used_arct |= {it.get("group_id") for it in ev if it.get("family")=="arct"}
seen_g=set(); arct_fresh=[]
for it in B.parse_arct():
    g=it["group_id"]
    if g in used_arct or g in seen_g: continue
    seen_g.add(g); arct_fresh.append(it)
arct_fresh.sort(key=lambda it: hashlib.sha1(f"bg:{it['id']}".encode()).hexdigest())
arct_work = arct_fresh[: args.arct]

# --- logiqa: fresh clean non-RC, exclude everything ---
excl = set(B._reclor_keys())
for d in (json.loads(l) for l in (RAW/"logiqa2_test.txt").read_text().splitlines() if l.strip()):
    excl.add(B._norm_key(B.norm_ws(d.get("text",""))))
for fn in ("eval_items.json","eval_items_clean.json"):
    for it in json.load(open(DATA/fn)):
        if it.get("family")=="logiqa": excl.add(B._norm_key(B.norm_ws(it.get("stimulus",""))))
for src in ("logiqa_new_raw.jsonl","logiqa_repl_items.jsonl"):
    for l in (DATA/src).read_text().splitlines():
        if not l.strip(): continue
        o=json.loads(l)
        st=o.get("stimulus") or o.get("prompt","").split("choose the SINGLE best option.\n\n",1)[-1].split("\n\nQuestion:",1)[0]
        excl.add(B._norm_key(B.norm_ws(st)))
def valid(d):
    o,a=d.get("options") or [],d.get("answer"); t,q=B.norm_ws(d.get("text","")),B.norm_ws(d.get("question",""))
    return bool(t) and bool(q) and isinstance(a,int) and len(o)==4 and a<4
lq_fresh=[]; seen=set()
for split,fn in (("tr","logiqa2_train.txt"),("dv","logiqa2_dev.txt")):
    for d in (json.loads(l) for l in (RAW/fn).read_text().splitlines() if l.strip()):
        if not valid(d): continue
        text,q=B.norm_ws(d.get("text","")),B.norm_ws(d.get("question",""))
        k=B._norm_key(text)
        if k in excl or k in seen or is_rc(q): continue
        seen.add(k); opts,ans=d["options"],int(d["answer"])
        lq_fresh.append({"id":f"logiqa-bg-{split}-{d.get('id','x')}",
            "group_id":f"logiqa-ctx-{hashlib.sha1(text.encode()).hexdigest()[:12]}","family":"logiqa",
            "task_type":"inference","difficulty":"hard","mode":"mcq","stimulus":text,"question":q,"mc_question":q,
            "mc_choices":[B.norm_ws(str(o)) for o in opts],"mc_credited_index":ans,
            "reference_answer":B.norm_ws(str(opts[ans])),"source":f"LogiQA 2.0 {split} [bad-gold backfill]"})
lq_fresh.sort(key=lambda it: hashlib.sha1(f"bg:{it['id']}".encode()).hexdigest())
lq_work = lq_fresh[: args.logiqa]

rows=[B.to_row(it) for it in arct_work]+[B.to_row(it) for it in lq_work]
Path(DATA/"_badgold_backfill_raw.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows))
print(f"arct fresh available {len(arct_fresh)} -> working {len(arct_work)} | logiqa fresh {len(lq_fresh)} -> working {len(lq_work)}")
print(f"wrote {len(rows)} template rows -> data/_badgold_backfill_raw.jsonl")
