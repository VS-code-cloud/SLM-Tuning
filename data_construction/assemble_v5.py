#!/usr/bin/env python3
"""Assemble sft_train_v5.jsonl (SEPARATE): v4 + logiqa elimination traces.

v5 = shortened base + P2(folio up-sample) + P4(irrelevant-aug) + **logiqa Opus-elim**.
The only change vs v4 is the logiqa completions (shortened -> Opus distractor-elimination),
family-specifically — lsat_lr stays shortened (elim hurt it), logicnli stays shortened
(deepen hurt it). Rationale: logiqa-elim helped the noisy family (v2 80 vs v4 66.7).
Deterministic; reuses data/mcq_elim_traces.jsonl (no gateway). Includes a full leakage check.
"""
import json, re, sys, hashlib
from collections import Counter, defaultdict
from pathlib import Path
HERE = Path(__file__).resolve().parent; _COLAB = HERE.parent; DATA = _COLAB / "data"
sys.path.insert(0, str(_COLAB)); import slm_core as S
from transformers import AutoTokenizer
_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B", trust_remote_code=True)

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

base = [json.loads(l) for l in (DATA/"sft_train_short.jsonl").read_text().splitlines() if l.strip()]

# logiqa Opus-elim swap (the v5 change vs v4)
elim = {r["id"]: r["completion"]
        for r in (json.loads(l) for l in (DATA/"mcq_elim_traces.jsonl").read_text().splitlines() if l.strip())
        if r.get("trace_source") == "elim" and r.get("family") == "logiqa"}
out = []; swapped = 0
for r in base:
    r = dict(r)
    if r["family"] == "logiqa" and r["id"] in elim:
        r["completion"] = elim[r["id"]]; r["trace_source"] = "elim"; swapped += 1
    out.append(r)

# P2 folio-neutral up-sample
fneu = [r for r in out if r["family"] == "folio" and neutral(r)]
for r in fneu:
    out.append({**r, "id": r["id"] + "-up", "trace_source": r.get("trace_source","") + "|up"})
p2 = len(fneu)

# P4 irrelevant-premise aug
def stim_of(r):
    b = r["prompt"].split("Be specific and concise.\n\n", 1)
    return b[1].split("\n\nQuestion:", 1)[0] if len(b) == 2 else None
ent = [r for r in base if r["mode"] == "frq" and not r.get("lgmt_mr") and stim_of(r)]
donor = [s.strip() for r in ent for s in re.split(r'(?<=[.!?])\s+', stim_of(r)) if 20 < len(s) < 160]
p4 = 0
for r in sorted(ent, key=lambda r: hashlib.sha1(r["id"].encode()).hexdigest())[:600]:
    st = stim_of(r); inj = donor[int(hashlib.sha1((r["id"]+"irr").encode()).hexdigest()[:8],16) % len(donor)]
    if inj in st: continue
    out.append({**r, "id": r["id"]+"-irr", "prompt": r["prompt"].replace(st, st.rstrip()+" "+inj, 1),
                "trace_source": r.get("trace_source","")+"|irr", "lgmt_mr": "P3-irrelevant-aug"})
    p4 += 1

with (DATA/"sft_train_v5.jsonl").open("w", encoding="utf-8") as fh:
    for r in out: fh.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---- standard verify ----
ids = [r["id"] for r in out]; mm = sum(0 if grade_ok(r) else 1 for r in out)
over = sum(1 for r in out if fl(r["prompt"], r["completion"]) > 1024)
def massp(rows):
    m=defaultdict(int); mcq={"lsat_lr","arct","logiqa"}
    for r in rows: m[r["family"]]+=len(_tok(r["completion"],add_special_tokens=False)["input_ids"])
    t=sum(m.values()); return 100*sum(v for f,v in m.items() if f in mcq)/t, 100*sum(v for f,v in m.items() if f not in mcq)/t
print(f"[assemble_v5] wrote {len(out)} -> data/sft_train_v5.jsonl")
print(f"  logiqa-elim swapped {swapped} | P2 folio +{p2} | P4 irr +{p4}")
print(f"  unique ids {len(set(ids))==len(ids)} | grade-mismatch {mm} | over-1024 {over}")
print(f"  by family {dict(Counter(r['family'] for r in out))}")
mc,en=massp(out); print(f"  token mass MCQ {mc:.1f}% / entailment {en:.1f}%")

# ---- diff vs v4: only logiqa completions should differ ----
v4={json.loads(l)["id"]:json.loads(l) for l in open(DATA/"sft_train_v4.jsonl") if l.strip()}
diff=Counter(out_r["family"] for out_r in out if out_r["id"] in v4 and out_r["completion"]!=v4[out_r["id"]]["completion"])
print(f"  diff vs v4 (completions), by family: {dict(diff)}  (should be logiqa-only)")

# ---- LEAKAGE CHECK (train stimulus vs eval stimulus) ----
ev = json.load(open(DATA/"eval_items.json"))
def stim_tr(r):
    b = r["prompt"].split("choose the SINGLE best option.\n\n",1)[-1] if r["mode"]=="mcq" else r["prompt"].split("Be specific and concise.\n\n",1)[-1]
    return norm(b.split("\n\nQuestion:",1)[0])
tr_stim_all = set(stim_tr(r) for r in out)                      # incl. irr-augmented
tr_stim_clean = set(stim_tr(r) for r in out if "|irr" not in r.get("trace_source",""))
ev_stim = set(norm(it.get("stimulus","")) for it in ev)
ev_ids = set(it["id"] for it in ev); tr_ids = set(r["id"] for r in out)
print("\n=== LEAKAGE CHECK ===")
print(f"  id overlap train<->eval: {len(tr_ids & ev_ids)}  (must be 0)")
print(f"  stimulus overlap train(clean)<->eval: {len(tr_stim_clean & ev_stim)}  (must be 0)")
print(f"  stimulus overlap train(incl. irr-aug)<->eval: {len(tr_stim_all & ev_stim)}  (must be 0)")
# logiqa passages still in ReClor? (the dedup must hold in v5's logiqa set)
import zipfile
z=zipfile.ZipFile(DATA/"_raw"/"reclor_data.zip"); rc=set()
for sp in ("train.json","val.json","test.json"):
    for e in json.loads(z.read(sp,pwd=S.__dict__.get("_x") or __import__("data_construction.build_corpus",fromlist=["RECLOR_PW"]).RECLOR_PW)):
        if e.get("context"): rc.add(norm(e["context"]))
lq=[r for r in out if r["family"]=="logiqa"]
print(f"  logiqa passages present in ReClor (must be 0): {sum(1 for r in lq if stim_tr(r) in rc)}")
