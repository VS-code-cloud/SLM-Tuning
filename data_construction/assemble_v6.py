#!/usr/bin/env python3
"""Assemble sft_train_v6.jsonl (SEPARATE): v5 recipe + LogiQA anti-decline package.

The logiqa family declines (shortened 78.3 -> v4/v5 ~67) because the P2/P4 augmentation
rows dilute its SHARE of the training signal (15.1% -> 13.2%), NOT because of trace quality
(v5's logiqa-elim swap did not recover it). v6 targets the mechanism, three ways:

  (1) +300 FRESH clean logiqa  — pulled from LogiQA-2.0 train/dev (not-in-ReClor,
      not-in-corpus), Step-B traced + shortened (data/logiqa_new_short.jsonl). Restores share.
  (2) NOISE FILTER            — ensemble-disagreement drop (Haiku+Sonnet+Opus vote) on the
      AMBIGUOUS base logiqa (the grounded_template rows no model solved blind); drop rows
      where >=2/3 confidently disagree with gold (data/logiqa_noise_votes.jsonl). Removes
      bad-gold noise. Blind-agreed logiqa are clean by that fact and are kept untouched.
  (3) P4 irrelevant-aug 600 -> 300 — halves the diluting augmentation.

Everything else is v5: logiqa Opus-elim swap + P2 folio-neutral up-sample. Deterministic
given the (already-produced) gateway artifacts. Includes a full leakage/verify check.
"""
import json, re, sys, hashlib
from collections import Counter, defaultdict
from pathlib import Path
HERE = Path(__file__).resolve().parent; _COLAB = HERE.parent; DATA = _COLAB / "data"
sys.path.insert(0, str(_COLAB)); import slm_core as S
from transformers import AutoTokenizer
_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B", trust_remote_code=True)

BASE_KEYS = ["id", "family", "mode", "task_type", "difficulty", "prompt", "completion",
             "gold", "trace_source", "is_lgmt", "lgmt_mr"]
NEW_LOGIQA_TARGET = 300
P4_TARGET = 300                                          # was 600 in v4/v5 (halve the dilution)

def gl(g): m = re.match(r"\(([A-Z])\)", g or ""); return m.group(1) if m else None
def nop(p): return len(re.findall(r"^\(([A-Z])\)\s", p, re.M)) or 5
def neutral(r): return r["family"] in S.ENTAILMENT_FAMILIES and (S.canon_label(r["gold"]) or r["gold"]) in S.NEUTRAL
def fl(p, c):
    pre = _tok.apply_chat_template([{"role":"user","content":p}], tokenize=False, add_generation_prompt=True)
    return len(_tok(pre+c+(_tok.eos_token or ""), add_special_tokens=False)["input_ids"])
def grade_ok(r):
    if r["mode"] == "mcq":
        i = S.parse_letter(r["completion"], nop(r["prompt"])); return i is not None and chr(65+i) == gl(r["gold"])
    return S.labels_equiv(S.parse_label(r["completion"]), S.canon_label(r["gold"]) or r["gold"])
def norm(s): return re.sub(r"\s+"," ",(s or "").strip()).lower()
def rank(iid): return int(hashlib.sha1(f"v6:{iid}".encode()).hexdigest()[:8], 16)
def project(r): return {k: r.get(k) for k in BASE_KEYS}

def loadl(fn): return [json.loads(l) for l in (DATA/fn).read_text().splitlines() if l.strip()]

base = loadl("sft_train_short.jsonl")

# ---- v5 change: logiqa Opus-elim swap ----
elim = {r["id"]: r["completion"] for r in loadl("mcq_elim_traces.jsonl")
        if r.get("trace_source") == "elim" and r.get("family") == "logiqa"}
out = []; swapped = 0
for r in base:
    r = dict(r)
    if r["family"] == "logiqa" and r["id"] in elim:
        r["completion"] = elim[r["id"]]; r["trace_source"] = "elim"; swapped += 1
    out.append(r)

# ---- (2) NOISE FILTER: drop bad-gold base logiqa (Policy C — all 3 models produced a
# letter, UNANIMOUS on the SAME letter, and it is NOT the gold; i.e. the whole ensemble
# independently agrees on one specific better answer than the key = clearest bad-gold). ----
votes = loadl("logiqa_noise_votes.jsonl")
def _letters(v): return [v["votes"][m]["letter"] for m in ("haiku", "sonnet", "opus")]
noise_drop = {v["id"] for v in votes
              if all(_letters(v)) and len(set(_letters(v))) == 1 and _letters(v)[0] != v["gold_letter"]}
before = len(out)
out = [r for r in out if not (r["family"] == "logiqa" and r["id"] in noise_drop)]
n_noise_dropped = before - len(out)

# ---- (1) ADD 300 fresh clean logiqa (shortened Step-B blind traces) ----
new_all = loadl("logiqa_new_short.jsonl")
new_blind = [r for r in new_all if r.get("trace_source") == "blind"]
pool = new_blind if len(new_blind) >= NEW_LOGIQA_TARGET else \
    new_blind + [r for r in new_all if r.get("trace_source") == "backfill"]
pool = sorted(pool, key=lambda r: rank(r["id"]))[:NEW_LOGIQA_TARGET]
new_rows = []
for r in pool:
    nr = project(r); nr["is_lgmt"] = False; nr["lgmt_mr"] = None
    new_rows.append(nr)
out += new_rows
n_new_added = len(new_rows)

# ---- v5: P2 folio-neutral up-sample ----
fneu = [r for r in out if r["family"] == "folio" and neutral(r) and not str(r["id"]).endswith("-up")]
for r in fneu:
    out.append({**r, "id": r["id"] + "-up", "trace_source": (r.get("trace_source") or "") + "|up"})
p2 = len(fneu)

# ---- (3) P4 irrelevant-premise aug, HALVED to 300 ----
def stim_of(r):
    b = r["prompt"].split("Be specific and concise.\n\n", 1)
    return b[1].split("\n\nQuestion:", 1)[0] if len(b) == 2 else None
ent = [r for r in base if r["mode"] == "frq" and not r.get("lgmt_mr") and stim_of(r)]
donor = [s.strip() for r in ent for s in re.split(r'(?<=[.!?])\s+', stim_of(r)) if 20 < len(s) < 160]
p4 = 0
for r in sorted(ent, key=lambda r: hashlib.sha1(r["id"].encode()).hexdigest())[:P4_TARGET]:
    st = stim_of(r); inj = donor[int(hashlib.sha1((r["id"]+"irr").encode()).hexdigest()[:8],16) % len(donor)]
    if inj in st: continue
    out.append({**r, "id": r["id"]+"-irr", "prompt": r["prompt"].replace(st, st.rstrip()+" "+inj, 1),
                "trace_source": (r.get("trace_source") or "")+"|irr", "lgmt_mr": "P3-irrelevant-aug"})
    p4 += 1

with (DATA/"sft_train_v6.jsonl").open("w", encoding="utf-8") as fh:
    for r in out: fh.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---- standard verify ----
ids = [r["id"] for r in out]; mm = sum(0 if grade_ok(r) else 1 for r in out)
over = sum(1 for r in out if fl(r["prompt"], r["completion"]) > 1024)
def massp(rows):
    m=defaultdict(int); mcq={"lsat_lr","arct","logiqa"}
    for r in rows: m[r["family"]]+=len(_tok(r["completion"],add_special_tokens=False)["input_ids"])
    t=sum(m.values()); return 100*sum(v for f,v in m.items() if f in mcq)/t, 100*sum(v for f,v in m.items() if f not in mcq)/t
nlq = sum(1 for r in out if r["family"] == "logiqa")
print(f"[assemble_v6] wrote {len(out)} -> data/sft_train_v6.jsonl")
print(f"  logiqa-elim swapped {swapped} | noise-DROPPED {n_noise_dropped} | NEW logiqa +{n_new_added} "
      f"(blind pool {len(new_blind)}) | P2 folio +{p2} | P4 irr +{p4} (target {P4_TARGET})")
print(f"  logiqa rows now {nlq} ({100*nlq/len(out):.1f}% of corpus)  [was 1122 / 13.2% in v5]")
print(f"  unique ids {len(set(ids))==len(ids)} | grade-mismatch {mm} | over-1024 {over}")
print(f"  by family {dict(Counter(r['family'] for r in out))}")
mc,en=massp(out); print(f"  token mass MCQ {mc:.1f}% / entailment {en:.1f}%")

# ---- LEAKAGE CHECK (train stimulus vs eval stimulus), incl. the new pulled items ----
ev = json.load(open(DATA/"eval_items.json"))
def stim_tr(r):
    b = r["prompt"].split("choose the SINGLE best option.\n\n",1)[-1] if r["mode"]=="mcq" else r["prompt"].split("Be specific and concise.\n\n",1)[-1]
    return norm(b.split("\n\nQuestion:",1)[0])
tr_clean = set(stim_tr(r) for r in out if "|irr" not in (r.get("trace_source") or ""))
tr_all = set(stim_tr(r) for r in out)
ev_stim = set(norm(it.get("stimulus","")) for it in ev)
ev_ids = set(it["id"] for it in ev); tr_ids = set(r["id"] for r in out)
new_ids = set(r["id"] for r in new_rows)
print("\n=== LEAKAGE CHECK ===")
print(f"  id overlap train<->eval: {len(tr_ids & ev_ids)}  (must be 0)")
print(f"  stimulus overlap train(clean)<->eval: {len(tr_clean & ev_stim)}  (must be 0)")
print(f"  stimulus overlap train(incl irr-aug)<->eval: {len(tr_all & ev_stim)}  (must be 0)")
# new pulled logiqa: not in ReClor, not colliding with existing corpus ids
import zipfile
z=zipfile.ZipFile(DATA/"_raw"/"reclor_data.zip"); rc=set()
for sp in ("train.json","val.json","test.json"):
    for e in json.loads(z.read(sp, pwd=__import__("build_corpus").RECLOR_PW)):
        if e.get("context"): rc.add(norm(e["context"]))
new_stim = [stim_tr(r) for r in new_rows]
print(f"  NEW logiqa in ReClor (must be 0): {sum(1 for s in new_stim if s in rc)}")
print(f"  NEW logiqa stimulus in eval (must be 0): {len(set(new_stim) & ev_stim)}")
print(f"  NEW logiqa id collision with base (must be 0): {len(new_ids & set(r['id'] for r in base))}")

# ---- diff vs v5 ----
v5 = {json.loads(l)["id"]: json.loads(l) for l in open(DATA/"sft_train_v5.jsonl") if l.strip()}
v6_ids = set(r["id"] for r in out)
added = v6_ids - set(v5); removed = set(v5) - v6_ids
print("\n=== DIFF vs v5 ===")
print(f"  rows v5 {len(v5)} -> v6 {len(out)}")
print(f"  added ids {len(added)} (by family {dict(Counter(next((r['family'] for r in out if r['id']==i),'?') for i in added))})")
print(f"  removed ids {len(removed)} (by family {dict(Counter(v5[i]['family'] for i in removed))})")
