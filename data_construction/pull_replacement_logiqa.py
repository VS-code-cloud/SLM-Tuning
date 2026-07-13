#!/usr/bin/env python3
"""Pull fresh clean NON-RC LogiQA items to replace the removed reading-comprehension items.

Excludes: ReClor, base test corpus, the v6 +300 pull, current eval, eval-clean replacements,
and reading-comprehension (title/main-idea/theme) questions. Emits normalized items to
data/logiqa_repl_items.jsonl (deterministic working set). No gateway.
"""
import argparse, hashlib, json, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; _COLAB = HERE.parent; RAW = _COLAB/"data"/"_raw"; DATA = _COLAB/"data"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(_COLAB)); import build_corpus as B

RC = re.compile(r'best title|main idea|central idea|\btheme\b|mainly about|primarily about|meant to emphasize|intended to (indicate|express|convey|illustrate|emphasize)|best summar|most accurate summary|general idea|main topic|purpose of (this|the) (text|passage|article)|title for this', re.I)
EXC = re.compile(r'point (at issue|in dispute)|at issue between', re.I)
def is_rc(q): return bool(RC.search(q or "")) and not EXC.search(q or "")

def load(fn):
    return [json.loads(l) for l in (RAW/fn).read_text(encoding="utf-8").splitlines() if l.strip()]
def valid(d):
    o, a = d.get("options") or [], d.get("answer")
    t, q = B.norm_ws(d.get("text","")), B.norm_ws(d.get("question",""))
    return bool(t) and bool(q) and isinstance(a, int) and len(o) == 4 and a < 4

ap = argparse.ArgumentParser(); ap.add_argument("--working", type=int, default=90); args = ap.parse_args()

excl = set(B._reclor_keys())
for d in load("logiqa2_test.txt"): excl.add(B._norm_key(B.norm_ws(d.get("text",""))))
for fn in ("eval_items.json", "eval_items_clean.json"):
    for it in json.load(open(DATA/fn)):
        if it.get("family") == "logiqa": excl.add(B._norm_key(B.norm_ws(it.get("stimulus",""))))
for l in (DATA/"logiqa_new_raw.jsonl").read_text().splitlines():
    if l.strip():
        st = json.loads(l)["prompt"].split("choose the SINGLE best option.\n\n",1)[-1].split("\n\nQuestion:",1)[0]
        excl.add(B._norm_key(B.norm_ws(st)))

items = []; seen = set()
for split, fn in (("tr","logiqa2_train.txt"), ("dv","logiqa2_dev.txt")):
    for d in load(fn):
        if not valid(d): continue
        text, q = B.norm_ws(d.get("text","")), B.norm_ws(d.get("question",""))
        k = B._norm_key(text)
        if k in excl or k in seen or is_rc(q): continue
        seen.add(k); opts, ans = d["options"], int(d["answer"])
        items.append({"id": f"logiqa-repl2-{split}-{d.get('id','x')}",
                      "group_id": f"logiqa-ctx-{hashlib.sha1(text.encode()).hexdigest()[:12]}",
                      "family": "logiqa", "task_type": "inference", "difficulty": "hard", "mode": "mcq",
                      "stimulus": text, "question": q, "mc_question": q,
                      "mc_choices": [B.norm_ws(str(o)) for o in opts], "mc_credited_index": ans,
                      "reference_answer": B.norm_ws(str(opts[ans])), "source": f"LogiQA 2.0 {split} [RC-replacement]"})
items.sort(key=lambda it: hashlib.sha1(f"repl2:{it['id']}".encode()).hexdigest())
work = items[: args.working]
Path(DATA/"logiqa_repl_items.jsonl").write_text("".join(json.dumps(it, ensure_ascii=False)+"\n" for it in work))
print(f"clean non-RC candidates {len(items)} | working set {len(work)} -> data/logiqa_repl_items.jsonl")
