#!/usr/bin/env python3
"""Assemble sft_train_v2.jsonl (SEPARATE — current corpus untouched) from the 4 phases.

Base = data/sft_train_short.jsonl (current best). Applies:
  P1  swap lsat_lr+logiqa frontier completions <- data/mcq_elim_traces.jsonl (elim rows)
  P3  swap logicnli T/F completions            <- data/logicnli_deep_traces.jsonl (solver_deep)
  P2  up-sample folio-neutral (duplicate each once, id '-up')      [deterministic]
  P4  irrelevant-premise robustness variants for entailment rows   [deterministic]
Then verifies: 0 grade-mismatch, 0 over-1024, unique ids, 0 leakage vs eval_items,
per-family counts, token-mass balance. Read-only on every current file.
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
def load_map(fn, ts):
    p = DATA/fn
    if not p.exists(): print(f"  WARN {fn} missing — phase skipped"); return {}
    return {r["id"]: r["completion"] for r in (json.loads(l) for l in p.read_text().splitlines() if l.strip())
            if r.get("trace_source") == ts}
elim = load_map("mcq_elim_traces.jsonl", "elim")           # P1
deep = load_map("logicnli_deep_traces.jsonl", "solver_deep")  # P3

out = []; p1 = p3 = 0
for r in base:
    r = dict(r)
    if r["id"] in elim: r["completion"] = elim[r["id"]]; r["trace_source"] = "elim"; p1 += 1
    elif r["id"] in deep: r["completion"] = deep[r["id"]]; r["trace_source"] = "solver_deep"; p3 += 1
    out.append(r)

# P2 — up-sample folio-neutral (duplicate once)
fneu = [r for r in out if r["family"] == "folio" and neutral(r)]
for r in fneu:
    out.append({**r, "id": r["id"] + "-up", "trace_source": r.get("trace_source","") + "|up"})
p2 = len(fneu)

# P4 — irrelevant-premise robustness variants (deterministic; borrow a sentence from a
# DIFFERENT entailment problem so it's guaranteed unrelated -> gold unchanged, reasoning ignores it)
def stim_of(r):
    b = r["prompt"].split("Be specific and concise.\n\n", 1)
    return b[1].split("\n\nQuestion:", 1)[0] if len(b) == 2 else None
ent = [r for r in base if r["mode"] == "frq" and not (r.get("lgmt_mr")) and stim_of(r)]
donors = [s for s in (stim_of(r) for r in ent) if s]
donor_sents = []
for s in donors:
    for sent in re.split(r'(?<=[.!?])\s+', s):
        if 20 < len(sent) < 160: donor_sents.append(sent.strip())
p4 = 0; TARGET = 600
for r in sorted(ent, key=lambda r: hashlib.sha1(r["id"].encode()).hexdigest())[:TARGET]:
    st = stim_of(r);
    if not st: continue
    h = int(hashlib.sha1((r["id"]+"irr").encode()).hexdigest()[:8], 16)
    inj = donor_sents[h % len(donor_sents)]
    if inj in st: continue                                  # skip accidental self-overlap
    new_stim = st.rstrip() + " " + inj                      # append irrelevant premise
    new_prompt = r["prompt"].replace(st, new_stim, 1)
    out.append({**r, "id": r["id"] + "-irr", "prompt": new_prompt,
                "trace_source": (r.get("trace_source","") + "|irr"), "lgmt_mr": "P3-irrelevant-aug"})
    p4 += 1

# ---- write ----
with (DATA/"sft_train_v2.jsonl").open("w", encoding="utf-8") as fh:
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
# leakage check ignores the P4 aug rows (their stimulus is intentionally altered)
leak = len(set(stim_tr(r) for r in out if "|irr" not in r.get("trace_source","")) & set(norm(it.get("stimulus","")) for it in ev))
def mass(rows):
    m = defaultdict(int); mcq = {"lsat_lr","arct","logiqa"}
    for r in rows: m[r["family"]] += (len(_tok(r["completion"],add_special_tokens=False)["input_ids"]) if _tok else len(r["completion"])//4)
    t = sum(m.values()); return 100*sum(v for f,v in m.items() if f in mcq)/t, 100*sum(v for f,v in m.items() if f not in mcq)/t

print(f"\n[assemble_v2] wrote {len(out)} rows -> data/sft_train_v2.jsonl")
print(f"  applied: P1 elim {p1} | P3 deep {p3} | P2 folio-neutral +{p2} | P4 irrelevant-aug +{p4}")
print(f"  unique ids: {len(set(ids))==len(ids)} | grade-mismatch: {mm} | over-1024: {over} | leakage vs eval: {leak}")
print(f"  by family: {dict(Counter(r['family'] for r in out))}")
mm_,en = mass(out); print(f"  token mass: MCQ {mm_:.1f}%  entailment {en:.1f}%")
print(f"  trace_source: {dict(Counter(r.get('trace_source') for r in out))}")
