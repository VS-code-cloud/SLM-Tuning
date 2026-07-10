#!/usr/bin/env python3
"""Colab smoke test — prove the REAL loop on the pre-built data (no GPU needed).

Runs the exact notebook path on a tiny slice: load pre-built SFT rows -> tokenize
with completion-only masking -> a few LoRA steps on a small Qwen3 -> base-vs-tuned
deterministic eval on a handful of held-out items. Green = the wiring is sound
(data schema, chat template, masking, LoRA attach, save, generate, grading).

Unlike the parent ``run_smoke.py`` (which fabricates junk), this trains on the
actual house-style traces, so it also sanity-checks the corpus format end-to-end.
It does NOT prove learning (too few steps) — real numbers come from the full Colab
run on Qwen3.5-0.8B.

Usage:
    python smoke.py                      # Qwen3-0.6B on CPU, 3 steps
    python smoke.py --model Qwen/Qwen3.5-0.8B --steps 8 --n-train 200
"""
from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(HERE.parent / ".hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import slm_core as S  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--n-train", type=int, default=48)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--eval-n", type=int, default=6)
    ap.add_argument("--max-len", type=int, default=384)
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    t0 = time.time()
    rows = S.load_sft_rows()[: args.n_train]
    eval_items = S.load_eval_items()
    print(f"[smoke] {len(rows)} train rows | {len(eval_items)} eval items | model={args.model}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def fmt(prompt, completion):
        try:
            prefix = tok.apply_chat_template([{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            prefix = tok.apply_chat_template([{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True)
        full = prefix + completion + (tok.eos_token or "")
        pids = tok(prefix, add_special_tokens=False)["input_ids"]
        fids = tok(full, add_special_tokens=False)["input_ids"][: args.max_len]
        labels = list(fids)
        for i in range(min(len(pids), len(labels))):
            labels[i] = -100
        return fids, labels

    enc = [fmt(r["prompt"], r["completion"]) for r in rows]
    enc = [(x, y) for x, y in enc if any(t != -100 for t in y)]
    assert enc, "no trainable examples after masking"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float32)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.print_trainable_parameters()

    def collate(batch):
        maxlen = max(len(x) for x, _ in batch); pad = tok.pad_token_id
        ii, ll, aa = [], [], []
        for x, y in batch:
            n = maxlen - len(x)
            ii.append(x + [pad]*n); ll.append(y + [-100]*n); aa.append([1]*len(x) + [0]*n)
        return torch.tensor(ii), torch.tensor(ll), torch.tensor(aa)

    dl = DataLoader(enc, batch_size=2, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    model.train(); losses = []
    for step, (ii, ll, aa) in enumerate(dl, 1):
        out = model(input_ids=ii, attention_mask=aa, labels=ll)
        out.loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        losses.append(float(out.loss.detach()))
        print(f"  step {step} loss {losses[-1]:.3f}")
        if step >= args.steps:
            break

    out_dir = HERE / "runs" / "smoke" / "adapter"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir)); tok.save_pretrained(str(out_dir))

    # deterministic eval on a few held-out items (both regimes)
    def gen(it, mode):
        prompt = S.build_prompt(it, mode)
        try:
            text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True)
        e = tok(text, return_tensors="pt")
        o = model.generate(**e, max_new_tokens=48, do_sample=False,
                           pad_token_id=(tok.pad_token_id or tok.eos_token_id))
        return tok.decode(o[0][e["input_ids"].shape[1]:], skip_special_tokens=True)

    model.eval()
    by = defaultdict(lambda: {"n": 0, "parse": 0})
    picked = []
    seen = set()
    for it in eval_items:
        if it["family"] not in seen or len([p for p in picked if p["family"] == it["family"]]) < 1:
            picked.append(it); seen.add(it["family"])
        if len(picked) >= args.eval_n:
            break
    import torch as _t
    with _t.no_grad():
        for it in picked:
            mode = "frq" if it["family"] in S.ENTAILMENT_FAMILIES else "mcq"
            g = S.grade(it, gen(it, mode), mode)
            k = by[(it["family"], mode)]
            k["n"] += 1; k["parse"] += int(g["parseable"])

    print("\n[smoke] eval parse-rate by family:")
    for k, v in sorted(by.items()):
        print(f"  {k[0]}:{k[1]}  parse {v['parse']}/{v['n']}")
    print(f"\nSMOKE PASSED ✓  load→tokenize→train({len(losses)} steps, "
          f"loss {losses[0]:.2f}→{losses[-1]:.2f})→save→eval in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
