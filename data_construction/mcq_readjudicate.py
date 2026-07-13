#!/usr/bin/env python3
"""Careful blind+non-blind Opus re-adjudication of low-confidence MCQ rows.

Universe = the 81 non-RC Policy-C train drops + the 165 non-RC hollow-template MCQ rows
(logiqa/arct/lsat_lr). For each:
  1. BLIND careful Opus (no gold): reason step by step -> answer + full reasoning.
     If it lands on the gold, the item is GOOD and its blind reasoning IS a faithful trace.
  2. If it misses, a FAIR steelman (2 passes, refined bar) decides: if either pass finds the
     key defensible -> GOOD-hard (generate an answer-conditioned careful trace committing to
     gold); if BOTH passes say key-wrong -> BAD (drop).
Emits data/mcq_readjudicated.jsonl {id, family, gold, verdict, method, trace_raw}. A later
Haiku pass shortens the GOOD trace_raw for stylistic consistency. Resumable (per-id cache).
"""
import concurrent.futures as cf, json, re, sys, threading, time
from collections import Counter
from pathlib import Path
HERE = Path(__file__).resolve().parent; _COLAB = HERE.parent; DATA = _COLAB / "data"
sys.path.insert(0, str(_COLAB)); import slm_core as S
OPUS = "claude-opus-4-8"

RC = re.compile(r'best title|main idea|central idea|\btheme\b|mainly about|primarily about|meant to emphasize|intended to (indicate|express|convey|illustrate|emphasize)|best summar|most accurate summary|general idea|main topic|purpose of (this|the) (text|passage|article)|title for this', re.I)
EXC = re.compile(r'point (at issue|in dispute)|at issue between', re.I)
def is_rc(q): return bool(RC.search(q or "")) and not EXC.search(q or "")
def qof(p): m = re.search(r"\nQuestion:\s*(.+?)\n", p); return m.group(1) if m else ""
def gl(g): m = re.match(r"\(([A-Z])\)", g or ""); return m.group(1) if m else None
def n_opt(p): o = re.findall(r"^\(([A-Z])\)\s", p, re.M); return len(o) if o else 5

base = {r["id"]: r for r in (json.loads(l) for l in (DATA/"sft_train_short.jsonl").read_text().splitlines() if l.strip())}
v6 = [json.loads(l) for l in (DATA/"sft_train_v6.jsonl").read_text().splitlines() if l.strip()]
votes = {json.loads(l)["id"]: json.loads(l) for l in (DATA/"logiqa_noise_votes.jsonl").read_text().splitlines() if l.strip()}
def lets(v): return [v["votes"][m]["letter"] for m in ("haiku","sonnet","opus")]
dropped = [i for i, v in votes.items() if all(lets(v)) and len(set(lets(v)))==1 and lets(v)[0]!=v["gold_letter"]]
def bts(r): return (r.get("trace_source") or "").split("|")[0]

universe = []
for i in dropped:
    r = base.get(i)
    if r and not is_rc(qof(r["prompt"])): universe.append({**r, "_src": "drop"})
for r in v6:
    if r["family"] in ("logiqa","arct","lsat_lr") and bts(r)=="grounded_template" and not is_rc(qof(r["prompt"])):
        universe.append({**r, "_src": "template"})
seen=set(); uni=[]
for r in universe:
    if r["id"] not in seen: seen.add(r["id"]); uni.append(r)
print(f"universe {len(uni)} ({dict(Counter(r['_src'] for r in uni))}, {dict(Counter(r['family'] for r in uni))})", flush=True)

def opus(p, mt=900): return S.call_agent(p, OPUS, 150, max_tokens=mt) or ""
def ans_of(t): m = re.search(r"ANSWER:\s*\(?\s*([A-Za-z])", t); return m.group(1).upper() if m else None

def steelman(prompt, gold):
    q = (prompt + f"\n\nThe key says ({gold}). Steelman the key first, then the case against. Only judge WRONG "
         f"if another option is CLEARLY better, not merely arguable. End:\nVERDICT: key-defensible | key-wrong")
    r = opus(q); m = re.search(r"VERDICT:\s*(key-defensible|key-wrong)", r, re.I)
    return m.group(1).lower() if m else "key-defensible"   # default to defensible (conservative)

def process(r):
    gold = gl(r["gold"]); prompt = r["prompt"]
    blind = opus(prompt + "\n\nReason step by step from the passage, then end with exactly: ANSWER: (X)")
    if ans_of(blind) == gold:
        return {"id": r["id"], "family": r["family"], "gold": r["gold"], "src": r["_src"],
                "verdict": "good", "method": "blind", "trace_raw": blind}
    v1 = steelman(prompt, gold)
    if v1 == "key-wrong" and steelman(prompt, gold) == "key-wrong":
        return {"id": r["id"], "family": r["family"], "gold": r["gold"], "src": r["_src"],
                "verdict": "bad", "method": "steelman", "trace_raw": ""}
    trace = opus(prompt + f"\n\n(The correct answer is {gold}.) Give a rigorous step-by-step justification "
                f"that derives it from the passage; reason as if solving it. End with exactly: ANSWER: ({gold})")
    return {"id": r["id"], "family": r["family"], "gold": r["gold"], "src": r["_src"],
            "verdict": "good", "method": "nonblind", "trace_raw": trace}

cache_path = DATA/"mcq_readjudicated.cache.jsonl"; cache = {}
if cache_path.exists():
    for l in cache_path.read_text().splitlines():
        if l.strip(): c = json.loads(l); cache[c["id"]] = c
pending = [r for r in uni if r["id"] not in cache]
print(f"{len(cache)} cached | {len(pending)} to do", flush=True)
lock = threading.Lock(); t0 = time.time(); done = 0
with cache_path.open("a", encoding="utf-8") as cfh:
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(process, r): r for r in pending}
        for fut in cf.as_completed(futs):
            try: rec = fut.result()
            except Exception: rec = None
            with lock:
                if rec: cfh.write(json.dumps(rec, ensure_ascii=False)+"\n"); cfh.flush(); cache[rec["id"]] = rec
                done += 1
                if done % 20 == 0 or done == len(pending):
                    print(f"  {done}/{len(pending)} ({done/max(1e-6,time.time()-t0):.1f}/s)", flush=True)

out = [cache[r["id"]] for r in uni if r["id"] in cache]
Path(DATA/"mcq_readjudicated.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out)+"\n")
g = [r for r in out if r["verdict"]=="good"]; b = [r for r in out if r["verdict"]=="bad"]
print(f"\n[readjudicate] {len(out)} judged -> data/mcq_readjudicated.jsonl")
print(f"  GOOD {len(g)} (blind {sum(1 for r in g if r['method']=='blind')} + nonblind {sum(1 for r in g if r['method']=='nonblind')}) | BAD(drop) {len(b)}")
print(f"  GOOD by src {dict(Counter(r['src'] for r in g))} | BAD by src {dict(Counter(r['src'] for r in b))}")
print(f"  GOOD by family {dict(Counter(r['family'] for r in g))}")
