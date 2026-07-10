#!/usr/bin/env python3
"""Score a frontier API model (default Opus 4.8) on the EXACT same held-out eval
as run_sft.py's base-vs-tuned eval, so it slots into the README as an apples-to-apples
reference row. Reuses slm_core: same build_prompt, same deterministic grade(), same
stratified even-stride pick per family, same LGMT probe — the only difference is the
answer comes from the Anthropic API (slm_core.call_agent) instead of a local model.

Needs ANTHROPIC_API_KEY in colab/.env (same mechanism as the Haiku judge).

    python opus_eval.py --model claude-opus-4-8 --eval-per-family 10 --lgmt-eval 8
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import time
from collections import defaultdict
from pathlib import Path

import slm_core as S

HERE = Path(__file__).resolve().parent
_IRR = "Additionally, an unrelated fact holds: Quennell owns a teal kite."

# satellite open-ended (FRQ) probe — mirrors run_sft.py so all 4 models get the same suite
_SAT_OPEN_Q = {
    "flaw": "Identify the flaw in the argument's reasoning. State it specifically.",
    "assumption": "State the unstated assumption the argument depends on.",
    "weaken": "State what would most weaken this argument.",
    "warrant": "State the implicit warrant — the unstated assumption that makes the reason support the claim.",
    "inference": "State the single conclusion that can be validly drawn from the information.",
}


def _frq_prompt_sat(it):
    q = _SAT_OPEN_Q.get(it.get("task_type", "")) or (it.get("question") or "Answer the question.")
    return (f"{it['stimulus'].strip()}\n\n{q}\n\nReason briefly from the argument, then state "
            "your answer in a final sentence.")


def _judge_frq(candidate, reference, model):
    if not candidate or not candidate.strip():
        return None
    q = ("Grade an open-ended answer to a critical-reasoning question against a reference "
         "answer. Reply with ONLY 'YES' if the candidate makes essentially the same point as "
         "the reference (a defensible paraphrase counts), otherwise 'NO'.\n\n"
         f"Reference:\n{reference}\n\nCandidate:\n{candidate}\n\nSame point? (YES/NO)")
    out = S.call_agent(q, model, timeout=60, max_tokens=8)
    return None if not out else out.strip().upper().startswith("Y")


def _stride(items, k):
    k = min(k, len(items))
    if k <= 0:
        return []
    s = len(items) / k
    return [items[int(i * s)] for i in range(k)]


def pick(eval_items, per_family, neutral_frac=0.25):
    """Same stratified even-stride pick as run_sft.py (must match to compare on the
    identical items): even stride per family, plus over-sample neutral within the
    entailment families so ~neutral_frac of the picked set is neutral gold."""
    by_fam = defaultdict(list)
    for it in eval_items:
        by_fam[it["family"]].append(it)
    fams = list(by_fam)
    ent = [f for f in fams if f in S.ENTAILMENT_FAMILIES]
    neu_quota = 0
    if ent and neutral_frac > 0 and per_family > 0:
        neu_quota = min(per_family, round(neutral_frac * len(fams) * per_family / len(ent)))
    picked = []
    for fam, items in by_fam.items():
        n = min(per_family, len(items))
        if n <= 0:
            continue
        if fam in ent and neu_quota > 0:
            neu = [it for it in items if S.is_neutral_gold(it, "frq")]
            non = [it for it in items if not S.is_neutral_gold(it, "frq")]
            kneu = min(neu_quota, len(neu), n)
            picked += _stride(neu, kneu) + _stride(non, n - kneu)
        else:
            picked += _stride(items, n)
    return picked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--eval-per-family", type=int, default=10)
    ap.add_argument("--lgmt-eval", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--neutral-frac", type=float, default=0.25,
                    help="target neutral/non-entailment fraction (matches run_sft --neutral-frac)")
    ap.add_argument("--frq-eval", type=int, default=0,
                    help="ALSO run the open-ended satellite FRQ probe on N items/family "
                         "(the model under test answers open-ended; a fixed --frq-judge grades)")
    ap.add_argument("--frq-judge", default="claude-haiku-4-5",
                    help="fixed judge model for the FRQ probe (kept constant across all "
                         "evaluated models so scores are comparable and no model self-judges)")
    ap.add_argument("--out", default=str(HERE / "runs" / "eval_opus48"))
    args = ap.parse_args()

    eval_items = S.load_eval_items()
    picked = pick(eval_items, args.eval_per_family, args.neutral_frac)
    graded = []
    for it in picked:
        mode = "frq" if it["family"] in S.ENTAILMENT_FAMILIES else "mcq"
        if mode == "mcq" and not S.has_mc(it):
            continue
        graded.append((it, mode))
    print(f"[opus_eval] model={args.model} items={len(graded)} "
          f"(eval_per_family={args.eval_per_family}) concurrency={args.concurrency}", flush=True)
    try:
        S._client()                       # warm the client / fail fast if no key
    except Exception as e:
        print(f"[opus_eval] no API client ({type(e).__name__}: {e}); need ANTHROPIC_API_KEY "
              f"in colab/.env", flush=True)
        return 2

    def answer(it, mode):
        return S.call_agent(S.build_prompt(it, mode), args.model, args.timeout, args.max_tokens)

    # ---- base-vs-tuned-style single condition (the frontier reference) ----
    t0 = time.time()
    b = defaultdict(lambda: {"n": 0, "parse": 0, "correct": 0, "neu_n": 0, "neu_c": 0})
    done = 0
    failed = 0
    samples = []
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(answer, it, m): (it, m) for it, m in graded}
        for fut in cf.as_completed(futs):
            it, mode = futs[fut]
            try:
                resp = fut.result()
            except Exception:
                resp = None
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{len(graded)} ({time.time()-t0:.0f}s)", flush=True)
            if resp is None:
                # API call failed/timed out. Exclude it from the denominator — exactly as
                # the LGMT and FRQ probes below do — so a transient gateway failure does not
                # get graded as a wrong/unparseable answer and deflate the frontier model's
                # acc/parse. Tracked separately and reported for transparency (no silent cap).
                failed += 1
                samples.append({"model": args.model, "id": it.get("id"), "family": it["family"],
                                "mode": mode, "gold": S.credited_answer(it, mode),
                                "neutral_gold": S.is_neutral_gold(it, mode),
                                "parseable": False, "correct": False, "needs_judge": False,
                                "api_failed": True, "completion": None})
                continue
            g = S.grade(it, resp, mode, judge_model=None)
            samples.append({"model": args.model, "id": it.get("id"), "family": it["family"],
                            "mode": mode, "gold": S.credited_answer(it, mode),
                            "neutral_gold": S.is_neutral_gold(it, mode),
                            "parseable": bool(g["parseable"]), "correct": bool(g["correct"]),
                            "needs_judge": bool(g["needs_judge"]), "completion": resp})
            if g["needs_judge"]:
                continue
            k = b[it["family"]]
            k["n"] += 1; k["parse"] += int(g["parseable"]); k["correct"] += int(bool(g["correct"]))
            if S.is_neutral_gold(it, mode):
                k["neu_n"] += 1; k["neu_c"] += int(bool(g["correct"]))
    tot = defaultdict(int)
    for v in b.values():
        for kk, vv in v.items():
            tot[kk] += vv
    pct = lambda a, d: round(100 * a / d, 1) if d else None
    fail_note = f" api_failed={failed} (excluded from denominator)" if failed else ""
    print(f"[{args.model}] n={tot['n']} parse={pct(tot['parse'],tot['n'])}% "
          f"acc={pct(tot['correct'],tot['n'])}% neutral_recall={pct(tot['neu_c'],tot['neu_n'])}%"
          f"{fail_note}",
          flush=True)
    print("\nby family (acc, n) [neutral]")
    for fam in sorted(b):
        k = b[fam]
        neu = f"  neutral {pct(k['neu_c'],k['neu_n'])}%" if k["neu_n"] else ""
        print(f"  {fam:9} {pct(k['correct'],k['n'])}% (n={k['n']}){neu}")

    # ---- LGMT consistency (same probe/sample as run_sft) ----
    ent = [it for it in eval_items if it["family"] in S.ENTAILMENT_FAMILIES]
    n = min(args.lgmt_eval, len(ent))
    lgmt = None
    if n > 0:
        stepn = len(ent) / n
        sample = [ent[int(i * stepn)] for i in range(n)]
        def lg(it):
            hint = S.neutral_variant(it)
            src = S.parse_label(answer(it, "frq") or "", neutral_hint=hint)
            fu = dict(it); fu["stimulus"] = it["stimulus"].rstrip() + " " + _IRR
            flab = S.parse_label(answer(fu, "frq") or "", neutral_hint=hint)
            gold = S.canon_label(it["reference_answer"]) or it["reference_answer"]
            return src, flab, gold
        viol = hidden = usable = 0
        with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            for src, flab, gold in ex.map(lg, sample):
                if src is None or flab is None:
                    continue
                usable += 1
                if not S.labels_equiv(src, flab):
                    viol += 1
                    if S.labels_equiv(src, gold):
                        hidden += 1
        lgmt = {"n": usable, "mvr": pct(viol, usable), "hdr": pct(hidden, usable)}
        print(f"[{args.model}] LGMT consistency (n={usable}): MVR {lgmt['mvr']}%  "
              f"HDR {hidden}/{usable}={lgmt['hdr']}%", flush=True)

    # ---- optional satellite FRQ probe: model-under-test answers open-ended; a FIXED
    # judge (constant across all evaluated models, so scores compare and no model
    # self-judges) grades semantic correctness. Same probe as run_sft. ----
    frq = None
    if args.frq_eval:
        byf = defaultdict(list)
        for it in eval_items:
            if it["family"] in S.MCQ_FAMILIES:
                byf[it["family"]].append(it)
        fpick = []
        for fam, its in byf.items():
            fpick += _stride(its, min(args.frq_eval, len(its)))
        print(f"[{args.model}] FRQ probe: {len(fpick)} open-ended items, judge={args.frq_judge}", flush=True)
        def _one_frq(it):
            o = S.call_agent(_frq_prompt_sat(it), args.model, args.timeout, args.max_tokens)
            return it, o, _judge_frq(o, S.credited_answer(it, "mcq"), args.frq_judge)
        fb = defaultdict(lambda: {"n": 0, "c": 0})
        with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            for it, o, v in ex.map(_one_frq, fpick):
                samples.append({"model": args.model, "probe": "frq", "id": it.get("id"),
                                "family": it["family"], "task_type": it.get("task_type"),
                                "reference": S.credited_answer(it, "mcq"),
                                "judged_correct": v, "completion": o})
                if v is not None:
                    k = fb[it["family"]]; k["n"] += 1; k["c"] += int(v)
        ftot = defaultdict(int)
        for v in fb.values():
            for kk, vv in v.items(): ftot[kk] += vv
        print(f"[{args.model}] FRQ correctness n={ftot['n']} acc={pct(ftot['c'],ftot['n'])}%", flush=True)
        frq = {"judge": args.frq_judge, "acc": pct(ftot['c'], ftot['n']), "n": ftot['n'],
               "by_family": {fam: {"acc": pct(fb[fam]['c'], fb[fam]['n']), "n": fb[fam]['n']}
                             for fam in sorted(fb)}}

    metrics = {"model": args.model, "eval_per_family": args.eval_per_family,
               "neutral_frac": args.neutral_frac, "neutral_n": tot["neu_n"], "n": tot["n"],
               "parse": pct(tot["parse"], tot["n"]), "acc": pct(tot["correct"], tot["n"]),
               "neutral": pct(tot["neu_c"], tot["neu_n"]),
               "by_family": {fam: {"acc": pct(b[fam]["correct"], b[fam]["n"]), "n": b[fam]["n"],
                                   "neutral": pct(b[fam]["neu_c"], b[fam]["neu_n"])}
                             for fam in sorted(b)},
               "lgmt": lgmt, "frq": frq}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with (Path(args.out) / "eval_samples.jsonl").open("w", encoding="utf-8") as fh:
        for r in samples:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n[opus_eval] metrics -> {Path(args.out)/'metrics.json'}  "
          f"| raw answers -> {Path(args.out)/'eval_samples.jsonl'} (neutral n={tot['neu_n']})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
