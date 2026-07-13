#!/usr/bin/env python3
"""Pull fresh, clean LogiQA-2.0 items to counter the mix-shift dilution of the logiqa family.

Sources LogiQA-2.0 *train* + *dev* (never the test split we already use), keeps only items
that are (a) valid 4-option format, (b) NOT present in ReClor (the same cross-family dedup
build_corpus applies — ReClor owns shared LSAT passages), (c) NOT already in our corpus
(current logiqa2_test.txt, incl. the 478 repl), (d) passage-unique within the pull. Then
deterministically samples a WORKING set (default 420) and emits pipeline-native template rows
(via build_corpus.to_row) so they run through the EXACT same Step-B -> shorten -> noise-gate
path as the existing logiqa. Distinct ids (`logiqa-new-{split}-{origid}`) so nothing collides.

Read-only w.r.t. the shared corpus; writes only data/logiqa_new_raw.jsonl. No gateway.

    python pull_more_logiqa.py --working 420          # sample 420 clean candidates to trace
"""
import argparse, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; _COLAB = HERE.parent; RAW = _COLAB / "data" / "_raw"; DATA = _COLAB / "data"
sys.path.insert(0, str(HERE)); import build_corpus as B
sys.path.insert(0, str(_COLAB)); import slm_core as S  # noqa: F401 (imported by build_corpus)


def load_jsonl(fn):
    out = []
    for line in (RAW / fn).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def valid(d):
    opts, ans = d.get("options") or [], d.get("answer")
    text, q = B.norm_ws(d.get("text", "")), B.norm_ws(d.get("question", ""))
    return bool(text) and bool(q) and isinstance(ans, int) and len(opts) == 4 and ans < 4


def to_item(d, split):
    """parse_logiqa-style normalized item, with a distinct new id."""
    text, q = B.norm_ws(d.get("text", "")), B.norm_ws(d.get("question", ""))
    opts, ans = d["options"], int(d["answer"])
    import hashlib
    return {
        "id": f"logiqa-new-{split}-{d.get('id', 'x')}",
        "group_id": f"logiqa-ctx-{hashlib.sha1(text.encode()).hexdigest()[:12]}",
        "family": "logiqa", "task_type": "inference", "difficulty": "hard", "mode": "mcq",
        "stimulus": text, "question": q, "mc_question": q,
        "mc_choices": [B.norm_ws(str(o)) for o in opts], "mc_credited_index": ans,
        "reference_answer": B.norm_ws(str(opts[ans])),
        "source": f"LogiQA 2.0 (English) {split} [v6 pull]",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--working", type=int, default=420, help="working-set size to trace (net target lands lower)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(DATA / "logiqa_new_raw.jsonl"))
    args = ap.parse_args()

    rc = B._reclor_keys()
    cur_keys = {B._norm_key(B.norm_ws(d.get("text", ""))) for d in load_jsonl("logiqa2_test.txt")}
    seen = set(); clean = []
    for split, fn in (("tr", "logiqa2_train.txt"), ("dv", "logiqa2_dev.txt")):
        for d in load_jsonl(fn):
            if not valid(d):
                continue
            k = B._norm_key(B.norm_ws(d.get("text", "")))
            if k in rc or k in cur_keys or k in seen:
                continue
            seen.add(k); clean.append(to_item(d, split))
    # deterministic sample of the working set (same hashed-rank shuffle as build_corpus.cap)
    clean.sort(key=lambda it: B._rank(it["id"], args.seed))
    work = clean[: args.working]

    rows = [B.to_row(it) for it in work]
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # sanity: grade-consistent templates, unique ids, no eval leakage
    ev = json.load(open(DATA / "eval_items.json"))
    ev_stim = {B._norm_key(it.get("stimulus", "")) for it in ev}
    leak = sum(1 for it in work if B._norm_key(it["stimulus"]) in ev_stim)
    ids_uniq = len({r["id"] for r in rows}) == len(rows)
    from collections import Counter
    print(f"[pull] clean-available {len(clean)} | working set {len(work)} -> {args.out}")
    print(f"  by split: {dict(Counter(it['id'].split('-')[2] for it in work))}")
    print(f"  unique ids: {ids_uniq} | eval-stimulus leakage: {leak} (must be 0)")
    print(f"  gold-letter dist: {dict(Counter(r['gold'][:3] for r in rows))}")


if __name__ == "__main__":
    main()
