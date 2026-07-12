#!/usr/bin/env python3
"""Assemble sft_train_v3.jsonl (SEPARATE — current corpus untouched): the WINNERS from v2.

v2 stacked 4 phases; the eval showed 2 helped and 2 hurt. v3 keeps only the winners:
  KEEP  P1 elimination distillation for **lsat_lr only** (lsat_lr +3.3 in v2)
  KEEP  P4 irrelevant-premise robustness aug (MVR 20->5, beat Opus)
  DROP  P1 for logiqa (logiqa -10, noisy gold), P2 folio up-sample (folio -8.3,
        over-committed), P3 logicnli deepen (logicnli -3.3, added noise)

Base = sft_train_short.jsonl. Deterministic, reuses data/mcq_elim_traces.jsonl (no gateway).
"""
import json, re, sys, hashlib
from collections import Counter, defaultdict
from pathlib import Path
HERE = Path(__file__).resolve().parent; _COLAB = HERE.parent; DATA = _COLAB / "data"
sys.path.insert(0, str(_COLAB)); import slm_core as S
try:
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B", trust_remote_code=True)
except Exception:
    _tok = None

def gl(g): m = re.match(r"\(([A-Z])\)", g or ""); return m.group(1) if m else None
def nop(p): return len(re.findall(r"^\(([A-Z])\)\s", p, re.M)) or 5
def fl(p, c):
    if _tok is None: return (len(p)+len(c))//4
    pre = _tok.apply_chat_template([{"role":"user","content":p}], tokenize=False, add_generation_prompt=True)
    return len(_tok(pre+c+(_tok.eos_token or ""), add_special_tokens=False)["input_ids"])
def grade_ok(r):
    if r["mode"] == "mcq":
        i = S.parse_letter(r["completion"], nop(r["prompt"])); return i is not None and chr(65+i) == gl(r["gold"])
    return S.labels_equiv(S.parse_label(r["completion"]), S.canon_label(r["gold"]) or r["gold"])

base = [json.loads(l) for l in (DATA/"sft_train_short.jsonl").read_text().splitlines() if l.strip()]

# WINNER P1 — lsat_lr ONLY (drop logiqa elim); prefer the trimmed (~700 char) traces,
# fall back to the original elim trace where a trim wasn't produced.
elim = {}
for fn in ("mcq_elim_traces.jsonl", "mcq_elim_traces_trim.jsonl"):   # trim overrides original
    p = DATA / fn
    if not p.exists():
        continue
    for r in (json.loads(l) for l in p.read_text().splitlines() if l.strip()):
        if r.get("trace_source") == "elim" and r.get("family") == "lsat_lr":
            elim[r["id"]] = r["completion"]

out = []; p1 = 0
for r in base:
    r = dict(r)
    if r["family"] == "lsat_lr" and r["id"] in elim:
        r["completion"] = elim[r["id"]]; r["trace_source"] = "elim"; p1 += 1
    out.append(r)                                   # logiqa/folio/logicnli kept as-is (drop P1-logiqa, P2, P3)

# WINNER P4 — irrelevant-premise robustness variants (deterministic; borrow a sentence from a
# DIFFERENT entailment problem -> guaranteed unrelated -> gold unchanged, reasoning ignores it)
def stim_of(r):
    b = r["prompt"].split("Be specific and concise.\n\n", 1)
    return b[1].split("\n\nQuestion:", 1)[0] if len(b) == 2 else None
ent = [r for r in base if r["mode"] == "frq" and not r.get("lgmt_mr") and stim_of(r)]
donor_sents = []
for r in ent:
    for sent in re.split(r'(?<=[.!?])\s+', stim_of(r)):
        if 20 < len(sent) < 160: donor_sents.append(sent.strip())
p4 = 0; TARGET = 600
for r in sorted(ent, key=lambda r: hashlib.sha1(r["id"].encode()).hexdigest())[:TARGET]:
    st = stim_of(r)
    h = int(hashlib.sha1((r["id"]+"irr").encode()).hexdigest()[:8], 16)
    inj = donor_sents[h % len(donor_sents)]
    if inj in st: continue
    new_prompt = r["prompt"].replace(st, st.rstrip() + " " + inj, 1)
    out.append({**r, "id": r["id"] + "-irr", "prompt": new_prompt,
                "trace_source": (r.get("trace_source","") + "|irr"), "lgmt_mr": "P3-irrelevant-aug"})
    p4 += 1

with (DATA/"sft_train_v3.jsonl").open("w", encoding="utf-8") as fh:
    for r in out: fh.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---- verify ----
ids = [r["id"] for r in out]
mm = sum(0 if grade_ok(r) else 1 for r in out)
over = sum(1 for r in out if fl(r["prompt"], r["completion"]) > 1024)
ev = json.load(open(DATA/"eval_items.json"))
def norm(s): return re.sub(r"\s+"," ",(s or "").strip()).lower()
def stim_tr(r):
    b = r["prompt"].split("choose the SINGLE best option.\n\n",1)[-1] if r["mode"]=="mcq" else r["prompt"].split("Be specific and concise.\n\n",1)[-1]
    return norm(b.split("\n\nQuestion:",1)[0])
leak = len(set(stim_tr(r) for r in out if "|irr" not in r.get("trace_source","")) & set(norm(it.get("stimulus","")) for it in ev))
def mass(rows):
    m = defaultdict(int); mcq = {"lsat_lr","arct","logiqa"}
    for r in rows: m[r["family"]] += (len(_tok(r["completion"],add_special_tokens=False)["input_ids"]) if _tok else len(r["completion"])//4)
    t = sum(m.values()); return 100*sum(v for f,v in m.items() if f in mcq)/t, 100*sum(v for f,v in m.items() if f not in mcq)/t
print(f"\n[assemble_v3] wrote {len(out)} rows -> data/sft_train_v3.jsonl")
print(f"  KEEP: P1 lsat_lr-elim {p1} | P4 irrelevant-aug +{p4}   (dropped: logiqa-elim, P2 folio-upsample, P3 logicnli-deepen)")
print(f"  unique ids: {len(set(ids))==len(ids)} | grade-mismatch: {mm} | over-1024: {over} | leakage vs eval: {leak}")
print(f"  by family: {dict(Counter(r['family'] for r in out))}")
m_,e_ = mass(out); print(f"  token mass: MCQ {m_:.1f}%  entailment {e_:.1f}%  (shortened base was 43.7/56.3)")
