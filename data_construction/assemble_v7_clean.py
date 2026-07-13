#!/usr/bin/env python3
"""Finalize sft_train_v7.jsonl (the MCQ-quality cleanup stage of v7) + eval_items_clean.json.

Reads the pre-cleanup base (sft_train_v7_base.jsonl = v6 + P4=600) and applies:
  TRAIN:
   - remove reading-comprehension (non-logic) MCQ rows (title/main-idea/theme; keeps point-at-issue)
   - drop the 105 rows careful Opus judged bad-gold (re-adjudicated 92-drops + hollow templates)
   - upgrade the 102 GOOD hollow-template rows + restore the 38 GOOD drops -> real Haiku/Sonnet
     shortened careful-Opus traces (data/mcq_readj_final_traces.jsonl)
   - add 46 fresh clean non-RC logiqa (Step-B traced + shortened) to replace the removed RC
  EVAL:
   - replace the 7 remaining RC eval items with fresh clean non-RC items (n_agree==3 verified)

Deterministic given the produced artifacts. Full verify + leakage check vs the FINAL eval.
"""
import json, re, sys, hashlib
from collections import Counter, defaultdict
from pathlib import Path
HERE = Path(__file__).resolve().parent; _COLAB = HERE.parent; DATA = _COLAB / "data"
sys.path.insert(0, str(_COLAB)); import slm_core as S
from transformers import AutoTokenizer
_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B", trust_remote_code=True)

MCQ = {"logiqa","arct","lsat_lr"}
BASE_KEYS = ["id","family","mode","task_type","difficulty","prompt","completion","gold","trace_source","is_lgmt","lgmt_mr"]
RC = re.compile(r'best title|main idea|central idea|\btheme\b|mainly about|primarily about|meant to emphasize|intended to (indicate|express|convey|illustrate|emphasize)|best summar|most accurate summary|general idea|main topic|purpose of (this|the) (text|passage|article)|title for this', re.I)
EXC = re.compile(r'point (at issue|in dispute)|at issue between', re.I)
def is_rc(q): return bool(RC.search(q or "")) and not EXC.search(q or "")
def qof(p): m = re.search(r"\nQuestion:\s*(.+?)\n", p); return m.group(1) if m else ""
def gl(g): m = re.match(r"\(([A-Z])\)", g or ""); return m.group(1) if m else None
def nop(p): return len(re.findall(r"^\(([A-Z])\)\s", p, re.M)) or 5
def grade_ok(r):
    if r["mode"]=="mcq":
        i=S.parse_letter(r["completion"], nop(r["prompt"])); return i is not None and chr(65+i)==gl(r["gold"])
    return S.labels_equiv(S.parse_label(r["completion"]), S.canon_label(r["gold"]) or r["gold"])
def fl(p,c):
    pre=_tok.apply_chat_template([{"role":"user","content":p}], tokenize=False, add_generation_prompt=True)
    return len(_tok(pre+c+(_tok.eos_token or ""), add_special_tokens=False)["input_ids"])
def norm(s): return re.sub(r"\s+"," ",(s or "").strip()).lower()
def loadl(fn): return [json.loads(l) for l in (DATA/fn).read_text().splitlines() if l.strip()]
def project(r): d={k:r.get(k) for k in BASE_KEYS}; return d

v7 = loadl("sft_train_v7_base.jsonl")
base = {r["id"]: r for r in loadl("sft_train_short.jsonl")}
readj = {r["id"]: r for r in loadl("mcq_readjudicated.jsonl")}
final_traces = {r["id"]: r["completion"] for r in loadl("mcq_readj_final_traces.jsonl")}
repl_train = [r for r in loadl("repl_train_short.jsonl") if r.get("shortened")]
bad_ids = {i for i,r in readj.items() if r["verdict"]=="bad"}
# drop traces the QC judge flagged as prevaricating (hedge/no clean commit)
prevaricate = {json.loads(l)["id"] for l in (DATA/"qc_trace_flags.jsonl").read_text().splitlines()
               if l.strip() and json.loads(l)["verdict"]=="prevaricates"}
good_drop_ids = {i for i,r in readj.items() if r["verdict"]=="good" and r["src"]=="drop" and i in final_traces}

out=[]; seen=set(); n_rc=n_bad=n_upg=0
n_prev=0
for r in v7:
    if r["family"] in MCQ and is_rc(qof(r["prompt"])): n_rc+=1; continue
    if r["id"] in bad_ids: n_bad+=1; continue
    if r["id"] in prevaricate: n_prev+=1; continue   # drop prevaricating (don't keep its hollow template)
    r=dict(r)
    if r["id"] in final_traces and (r.get("trace_source") or "").split("|")[0]=="grounded_template":
        r["completion"]=final_traces[r["id"]]; r["trace_source"]="reasoned"; n_upg+=1
    out.append(r); seen.add(r["id"])
# restore GOOD drops (add rows with real traces)
n_restore=0
for i in good_drop_ids:
    if i in seen or i not in base or i in prevaricate: continue
    nr=project(base[i]); nr["completion"]=final_traces[i]; nr["trace_source"]="reasoned"; nr["is_lgmt"]=False; nr["lgmt_mr"]=None
    out.append(nr); seen.add(i); n_restore+=1
# add 46 fresh RC-replacements (skip any QC-flagged prevaricators, pull next clean)
repl_add=[project(r) for r in repl_train if r["id"] not in prevaricate][:46]
for r in repl_add: r["is_lgmt"]=False; r["lgmt_mr"]=None
out += repl_add

# backfill the bad-gold drops: +46 arct, +16 logiqa (fresh clean, Step-B traced + shortened)
bg=[r for r in loadl("badgold_backfill_short.jsonl") if r.get("shortened") and r["id"] not in prevaricate]
bg_arct=[project(r) for r in bg if r["family"]=="arct"][:46]
bg_lq=[project(r) for r in bg if r["family"]=="logiqa"][:16]
for r in bg_arct+bg_lq: r["is_lgmt"]=False; r["lgmt_mr"]=None
out += bg_arct + bg_lq
n_bg_arct=len(bg_arct); n_bg_lq=len(bg_lq)

# ---- LGMT REBALANCE: rebuild a balanced ~3300 set using the EVAL's exact MR transforms — MR-P/C/S
# deterministic across all 3 entailment families; MR-E from real FOL (folio+logicnli). ----
sys.path.insert(0, str(HERE))
from lgmt_augment import gen_variants, parse_src, MR_CAT, load_fol_sources, base_id, save_cache
n_lgmt_old = sum(1 for r in out if r.get("lgmt_mr"))
out = [r for r in out if not r.get("lgmt_mr")]        # strip old LGMT augmentation
def _sig(prems, concl): return norm(" ".join(prems) + " || " + (concl or ""))
lg300 = json.loads((DATA/"lgmt300.json").read_text())
banned = {_sig(s["premises"], s["conclusion"]) for s in lg300["sources"].values()}   # never morph an eval source
fol_src = load_fol_sources()                          # base_id -> FOL (folio+logicnli), for MR-E
ENT = ("folio","logicnli","proverqa"); FOLFAM = {"folio","logicnli"}
def cats_for(fam): return ("MR-P","MR-C","MR-S","MR-E") if fam in FOLFAM else ("MR-P","MR-C","MR-S")
PER = 300                                             # ~3300 target (MR-E only folio+logicnli)
new_lgmt=[]; skip_leak=0
for fam in ENT:
    srcs = sorted((r for r in out if r["family"]==fam and r["mode"]=="frq" and not r.get("lgmt_mr")
                   and not any(t in r["id"] for t in ("-lgmt-","-irr","-up"))),
                  key=lambda r: hashlib.sha1(r["id"].encode()).hexdigest())
    cats = cats_for(fam)
    buckets={c:[] for c in cats}
    for r in srcs:
        if all(len(buckets[c])>=PER for c in cats): break
        ps = parse_src(r)
        if not ps: continue
        if _sig(ps["prems"], ps["concl"]) in banned: skip_leak+=1; continue
        src = fol_src.get(base_id(r["id"]))
        need_mre = ("MR-E" in cats) and len(buckets["MR-E"])<PER and src is not None
        seen=set()
        for v in gen_variants(r, src=src, call_agent=S.call_agent, enable_mre=need_mre):
            c = MR_CAT[v["lgmt_mr"]]
            if c in cats and c not in seen and len(buckets[c])<PER:   # 1 variant/source/category
                buckets[c].append(v); seen.add(c)
    for c in cats: new_lgmt += buckets[c]
save_cache()                                          # persist FOL->NL translations (offline rebuilds)
out += new_lgmt
print(f"[LGMT] dropped {n_lgmt_old} old | added {len(new_lgmt)} balanced "
      f"(leakage-skipped {skip_leak}) | by MR-cat {dict(Counter(MR_CAT[r['lgmt_mr']] for r in new_lgmt))} "
      f"| by family {dict(Counter(r['family'] for r in new_lgmt))} "
      f"| by (fam,cat) {dict(sorted(Counter((r['family'],MR_CAT[r['lgmt_mr']]) for r in new_lgmt).items()))}")

with (DATA/"sft_train_v7.jsonl").open("w",encoding="utf-8") as fh:
    for r in out: fh.write(json.dumps(r,ensure_ascii=False)+"\n")

# ---- finalize eval ----
ev = json.load(open(DATA/"eval_items_clean.json"))
rc_eval = [it["id"] for it in ev if it.get("family")=="logiqa" and is_rc(it.get("question",""))]
repl_items = {it["id"]: it for it in loadl("logiqa_repl_items.jsonl")}
evotes = loadl("repl_eval_votes.jsonl")
clean_ids = [v["id"] for v in evotes if v.get("n_agree")==3 and v["id"] in repl_items][:len(rc_eval)]
repl_ev = [repl_items[i] for i in clean_ids]
ri = iter(repl_ev); new_ev=[]
for it in ev:
    new_ev.append(next(ri) if it["id"] in set(rc_eval) else it)
Path(DATA/"eval_items_clean.json").write_text(json.dumps(new_ev, ensure_ascii=False, indent=0))

# ---- verify ----
ids=[r["id"] for r in out]; mm=sum(0 if grade_ok(r) else 1 for r in out)
over=sum(1 for r in out if fl(r["prompt"],r["completion"])>1024)
def bts(r): return (r.get("trace_source") or "").split("|")[0]
tmpl_mcq=sum(1 for r in out if r["family"] in MCQ and bts(r)=="grounded_template")
rc_left=sum(1 for r in out if r["family"] in MCQ and is_rc(qof(r["prompt"])))
print(f"[assemble_v7-clean] wrote {len(out)} -> data/sft_train_v7.jsonl")
print(f"  removed: RC {n_rc} | bad-gold {n_bad} | prevaricating {n_prev}   upgraded templates {n_upg} | restored drops {n_restore} | RC-replacements +{len(repl_add)} | bad-gold backfill +{n_bg_arct} arct +{n_bg_lq} logiqa")
print(f"  unique ids {len(set(ids))==len(ids)} | grade-mismatch {mm} | over-1024 {over}")
print(f"  MCQ hollow-templates left {tmpl_mcq} (was 169) | RC left in MCQ {rc_left} (must be 0)")
print(f"  by family {dict(Counter(r['family'] for r in out))}")
# leakage vs FINAL eval
def stim(r,is_item=False):
    if is_item: return norm(r.get("stimulus",""))
    b=r["prompt"].split("choose the SINGLE best option.\n\n",1)[-1] if r["mode"]=="mcq" else r["prompt"].split("Be specific and concise.\n\n",1)[-1]
    return norm(b.split("\n\nQuestion:",1)[0])
ev_stim=set(stim(it,True) for it in new_ev); ev_ids=set(it["id"] for it in new_ev)
tr_clean=set(stim(r) for r in out if "|irr" not in (r.get("trace_source") or ""))
print(f"\n=== EVAL finalize ===")
print(f"  eval items {len(new_ev)} | RC swapped {len(rc_eval)} -> clean {len(repl_ev)} | eval logiqa RC left {sum(1 for it in new_ev if it.get('family')=='logiqa' and is_rc(it.get('question','')))} (must be 0)")
print(f"  id overlap train<->eval {len(set(ids)&ev_ids)} | stimulus overlap {len(tr_clean&ev_stim)} (must be 0)")
