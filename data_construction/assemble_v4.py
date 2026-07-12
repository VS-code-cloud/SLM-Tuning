#!/usr/bin/env python3
"""Assemble sft_train_v4.jsonl (SEPARATE — current corpus untouched): the CORRECTED winners.

The real-v2 eval showed the two data changes that actually help are P2 (folio-neutral
up-sample -> folio-neutral 66.7->86.7) and P4 (irrelevant-premise aug -> robustness). The
P1 elimination distill and P3 logicnli-deepen HURT (lsat_lr/logicnli down) and are dropped.
So v4 layers ONLY P2 + P4 onto the shortened base (the current accuracy leader, 85.3).

  KEEP  P2 folio-neutral up-sample (duplicate each once)
  KEEP  P4 irrelevant-premise robustness aug (+~600)
  DROP  P1 elimination distill (lsat_lr & logiqa), P3 logicnli-deepen

Base = sft_train_short.jsonl. Fully deterministic, no gateway.
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
def neutral(r): return r["family"] in S.ENTAILMENT_FAMILIES and (S.canon_label(r["gold"]) or r["gold"]) in S.NEUTRAL
def fl(p, c):
    if _tok is None: return (len(p)+len(c))//4
    pre = _tok.apply_chat_template([{"role":"user","content":p}], tokenize=False, add_generation_prompt=True)
    return len(_tok(pre+c+(_tok.eos_token or ""), add_special_tokens=False)["input_ids"])
def grade_ok(r):
    if r["mode"] == "mcq":
        i = S.parse_letter(r["completion"], nop(r["prompt"])); return i is not None and chr(65+i) == gl(r["gold"])
    return S.labels_equiv(S.parse_label(r["completion"]), S.canon_label(r["gold"]) or r["gold"])

base = [json.loads(l) for l in (DATA/"sft_train_short.jsonl").read_text().splitlines() if l.strip()]
out = [dict(r) for r in base]                                  # no P1/P3 swaps -> base MCQ/logicnli kept

# WINNER P2 — up-sample folio-neutral (duplicate once)
fneu = [r for r in out if r["family"] == "folio" and neutral(r)]
for r in fneu:
    out.append({**r, "id": r["id"] + "-up", "trace_source": r.get("trace_source","") + "|up"})
p2 = len(fneu)

# WINNER P4 — irrelevant-premise robustness variants (borrow a sentence from a DIFFERENT
# entailment problem -> guaranteed unrelated -> gold unchanged, reasoning ignores it)
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

with (DATA/"sft_train_v4.jsonl").open("w", encoding="utf-8") as fh:
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
print(f"\n[assemble_v4] wrote {len(out)} rows -> data/sft_train_v4.jsonl")
print(f"  KEEP: P2 folio-neutral +{p2} | P4 irrelevant-aug +{p4}   (dropped: P1 elim, P3 deepen)")
print(f"  unique ids: {len(set(ids))==len(ids)} | grade-mismatch: {mm} | over-1024: {over} | leakage vs eval: {leak}")
print(f"  by family: {dict(Counter(r['family'] for r in out))}")
m_,e_ = mass(out); print(f"  token mass: MCQ {m_:.1f}%  entailment {e_:.1f}%  (shortened base 43.7/56.3)")
