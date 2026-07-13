#!/usr/bin/env python3
"""QC the newly-generated MCQ traces for PREVARICATION (hedging / not committing).

Judges every trace produced by this cleanup — restored/upgraded (mcq_readj_final_traces),
RC-replacements (repl_train_short), and bad-gold backfill (badgold_backfill_short) — with a
Sonnet judge: does the reasoning waffle / treat multiple options as viable without decisively
committing? (Decisively rejecting a distractor is NOT prevarication.) Writes flags to
data/qc_trace_flags.jsonl. No corpus mutation.
"""
import concurrent.futures as cf, json, re, sys, threading
from collections import Counter
from pathlib import Path
HERE = Path(__file__).resolve().parent; _COLAB = HERE.parent; DATA = _COLAB / "data"
sys.path.insert(0, str(_COLAB)); import slm_core as S
JUDGE = "claude-sonnet-4-6"

def loadl(fn): return [json.loads(l) for l in (DATA/fn).read_text().splitlines() if l.strip()]
new = []
for r in loadl("mcq_readj_final_traces.jsonl"): new.append({"id": r["id"], "family": r["family"], "src": "restored/upgraded", "completion": r["completion"]})
for r in loadl("repl_train_short.jsonl"):
    if r.get("shortened"): new.append({"id": r["id"], "family": r["family"], "src": "RC-replacement", "completion": r["completion"]})
for r in loadl("badgold_backfill_short.jsonl"):
    if r.get("shortened"): new.append({"id": r["id"], "family": r["family"], "src": "backfill", "completion": r["completion"]})
print(f"QC {len(new)} new traces ({dict(Counter(r['src'] for r in new))})", flush=True)

JP = ("Below is a written solution to a multiple-choice logic question. Judge whether it PREVARICATES: "
      "does the reasoning HEDGE, waffle, or treat more than one option as genuinely correct/viable "
      "without decisively committing to a single answer? Decisively rejecting a distractor "
      "(\"X is tempting but wrong because...\") is NOT prevarication — that is good reasoning. "
      "Prevarication is genuine indecision, self-contradiction, or endorsing multiple answers.\n\n"
      "Reply EXACTLY:\nVERDICT: clean | prevaricates\nREASON: <one short clause, only if prevaricates>\n\n"
      "--- solution ---\n")
def judge(r):
    resp = S.call_agent(JP + r["completion"], JUDGE, 90, max_tokens=120) or ""
    m = re.search(r"VERDICT:\s*(clean|prevaricates)", resp, re.I)
    rm = re.search(r"REASON:\s*(.+)", resp, re.I)
    return {**r, "verdict": (m.group(1).lower() if m else "clean"), "reason": (rm.group(1).strip()[:160] if rm else "")}

lock = threading.Lock(); out = []
with cf.ThreadPoolExecutor(max_workers=8) as ex:
    for rec in ex.map(judge, new):
        with lock: out.append(rec)
Path(DATA/"qc_trace_flags.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in out))
prev = [r for r in out if r["verdict"] == "prevaricates"]
print(f"\n[qc] clean {len(out)-len(prev)} | PREVARICATES {len(prev)} ({100*len(prev)/max(1,len(out)):.1f}%)")
print(f"  prevaricates by src {dict(Counter(r['src'] for r in prev))} | by family {dict(Counter(r['family'] for r in prev))}")
for r in prev[:12]:
    print(f"\n  [{r['src']}/{r['family']}] {r['id']}: {r['reason']}")
    print(f"    {r['completion'][:260]}")
