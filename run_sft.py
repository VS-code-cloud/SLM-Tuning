#!/usr/bin/env python3
"""Headless SFT runner — the notebook's train+eval as a script (GPU or CPU).

Same pipeline as ``critical_reasoning_sft.ipynb`` (load pre-built data ->
completion-masked LoRA -> base-vs-tuned deterministic eval), with a few
quality-preserving optimizations for a real run:

  * **token-budget, length-sorted batching** — pack shortest-first up to a token
    budget (rows x longest-row) so long ProverQA rows land in small batches ->
    bounded peak VRAM; batch ORDER is shuffled for stochasticity.
  * **token-weighted gradient accumulation** — each micro-batch contributes in
    proportion to its supervised-token count and the window is normalized by total
    tokens, so variable batch sizes don't bias the update; the trailing partial
    window is flushed (no dropped batches).
  * **bf16 on GPU** + SDPA attention + **gradient checkpointing** (with
    ``preserve_rng_state=False`` — the RNG fork throws cudaErrorUnknown on RTX 5050
    / sm_120); fp32 on CPU. LoRA (not 4-bit): a 0.6-0.8B base is tiny, so this
    avoids the bitsandbytes/Blackwell dependency with no quality loss.
  * **cosine LR schedule with warmup**.
  * **eval** re-enables the KV cache and disables gradient checkpointing (both are
    training-only), and samples each family's held-out slice by an even stride so
    the sample is representative across difficulty (not the first-N, which is
    ordered easy->hard for ProverQA).

Usage:
    python run_sft.py                              # auto device, Qwen3-0.6B, 3 epochs
    python run_sft.py --model Qwen/Qwen3.5-0.8B --batch-size 8 --grad-accum 2
    python run_sft.py --max-steps 40 --eval-per-family 8   # quick real check
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(HERE.parent / ".hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import slm_core as S  # noqa: E402


def build_batches(encoded, max_tokens, max_bs, seed=0):
    """Token-budget, length-sorted batches: pack examples (shortest first) until
    padded size (rows x longest-row) would exceed max_tokens or max_bs rows. Long
    sequences therefore land in SMALL batches automatically -> bounded peak VRAM
    (fixes the batch-4 OOM on long ProverQA rows). Batch order shuffled."""
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i][0]))
    batches, cur, curmax = [], [], 0
    for i in order:
        L = len(encoded[i][0])
        nmax = max(curmax, L)
        if cur and (nmax * (len(cur) + 1) > max_tokens or len(cur) >= max_bs):
            batches.append(cur); cur, curmax = [], 0; nmax = L
        cur.append(i); curmax = nmax
    if cur:
        batches.append(cur)
    batches.sort(key=lambda b: (hash((seed, b[0])) & 0xFFFF))
    return batches


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8, help="max rows per batch (token budget also caps it)")
    ap.add_argument("--max-tokens", type=int, default=3072, help="token budget per batch (rows x longest row)")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = full epochs")
    ap.add_argument("--save-every", type=int, default=0,
                    help="checkpoint the adapter every N optimizer updates (overwrites "
                         "runs/<out>/checkpoint); 0 = only save at the end")
    ap.add_argument("--train-file", default="",
                    help="path to the SFT JSONL to train on (default data/sft_train.jsonl; "
                         "point at data/sft_train_stepb.jsonl to use Step B traces)")
    ap.add_argument("--eval-per-family", type=int, default=40)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--eval-batch", type=int, default=8,
                    help="eval generation batch size: left-pad this many prompts into ONE "
                         "model.generate call. Single-sequence decoding is memory-bound at "
                         "low GPU utilization; batching fills the GPU for a several-x faster "
                         "eval. Lower it if generation OOMs (KV cache grows with the batch).")
    ap.add_argument("--neutral-frac", type=float, default=0.25,
                    help="target fraction of the base-vs-tuned eval that is neutral/"
                         "non-entailment gold, by over-sampling neutral within the entailment "
                         "families (they carry the hard 'does-not-follow' class). 0 = pure "
                         "even-stride. ~0.2-0.3 keeps the neutral recall statistically meaningful.")
    ap.add_argument("--frq-eval", type=int, default=0,
                    help="ALSO run an open-ended (FRQ) probe on N satellite items/family: pose "
                         "the generative question (state the flaw/warrant/assumption) and grade "
                         "with the gateway judge. Eval-only (training stays MCQ). 0 = off; needs "
                         "a judge model (--judge-model or gateway creds).")
    ap.add_argument("--lgmt-eval", type=int, default=0,
                    help="also report LGMT consistency (MVR/HDR) on N entailment eval "
                         "items per condition; 0 = off")
    ap.add_argument("--judge-model", default="",
                    help="cheap model id for the non-deterministic answer-extraction "
                         "fallback when parsing fails (needs cursor-agent auth); off if empty")
    ap.add_argument("--out", default=str(HERE / "runs" / "gpu_v1"))
    ap.add_argument("--load-4bit", action="store_true",
                    help="load the base in 4-bit nf4 (QLoRA) so bigger models fit 8 GB")
    ap.add_argument("--resume", default="",
                    help="resume training: load the LoRA adapter from this dir (e.g. a "
                         "prior --save-every checkpoint) instead of a fresh adapter")
    ap.add_argument("--skip-updates", type=int, default=0,
                    help="with --resume: N optimizer updates already done -> skip that many "
                         "micro-batch windows (same fixed batch order) and start the LR "
                         "schedule + step counter at N, so a crashed run continues its epoch")
    ap.add_argument("--eval-only", action="store_true",
                    help="skip ALL training and just run the base-vs-tuned eval suite on the "
                         "adapter loaded via --resume (does NOT save a final adapter, so it "
                         "won't falsely signal epoch completion to a supervisor)")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
    from peft import LoraConfig, get_peft_model

    use_cuda = torch.cuda.is_available() and not args.cpu
    device = "cuda" if use_cuda else "cpu"
    dtype = torch.bfloat16 if use_cuda else torch.float32
    print(f"[run_sft] device={device} model={args.model} "
          f"{'('+torch.cuda.get_device_name(0)+')' if use_cuda else ''}", flush=True)

    rows = S.load_sft_rows(args.train_file or None)
    eval_items = S.load_eval_items()
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

    t_enc = time.time()
    encoded = [fmt(r["prompt"], r["completion"]) for r in rows]
    dropped = sum(1 for _, y in encoded if not any(t != -100 for t in y))
    encoded = [(x, y) for x, y in encoded if any(t != -100 for t in y)]
    print(f"[run_sft] encoded {len(encoded)} rows ({dropped} dropped: prompt>={args.max_len}) "
          f"in {time.time()-t_enc:.0f}s", flush=True)

    # SDPA attention: memory-efficient (no O(seq^2) matrix), unlike eager which OOMs
    # 8 GB VRAM on long ProverQA sequences. Gradient checkpointing (below) is the
    # other half of fitting an 8 GB laptop GPU. --load-4bit adds nf4 QLoRA so bigger
    # bases (e.g. Qwen3.5-4B: ~2 GB in 4-bit) fit 8 GB.
    quant_config = None
    if args.load_4bit and use_cuda:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=dtype, quantization_config=quant_config,
        attn_implementation="sdpa", device_map=(device if use_cuda else None))
    if not use_cuda:
        model = model.to(device)
    if quant_config is not None:
        from peft import prepare_model_for_kbit_training
        # GC handled below with our sm_120-safe kwargs, not by prepare_* (reentrant)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model.config.use_cache = False
    if use_cuda:
        # preserve_rng_state=False: skips fork_rng/set_rng_state during recompute,
        # which throws cudaErrorUnknown on this RTX 5050 (sm_120) build. Safe because
        # lora_dropout=0 below -> recompute is deterministic anyway.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={
            "use_reentrant": False, "preserve_rng_state": False})
        model.enable_input_require_grads()   # required for GC + LoRA
    if args.resume and (Path(args.resume) / "adapter_config.json").exists():
        from peft import PeftModel
        # load the prior adapter onto the (already kbit-prepared, GC-enabled) base and
        # keep it trainable -> continue the same run from a checkpoint after a crash.
        model = PeftModel.from_pretrained(model, args.resume, is_trainable=True)
        print(f"[run_sft] RESUMED adapter from {args.resume} "
              f"(skip {args.skip_updates} updates)", flush=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    pad = tok.pad_token_id

    def make_tensor(idxs):
        batch = [encoded[i] for i in idxs]
        maxlen = max(len(x) for x, _ in batch)
        ii, ll, aa = [], [], []
        for x, y in batch:
            n = maxlen - len(x)
            ii.append(x + [pad]*n); ll.append(y + [-100]*n); aa.append([1]*len(x) + [0]*n)
        return (torch.tensor(ii, device=device), torch.tensor(ll, device=device),
                torch.tensor(aa, device=device))

    batches = build_batches(encoded, args.max_tokens, args.batch_size)
    print(f"[run_sft] {len(batches)} token-budget batches "
          f"(<= {args.max_tokens} tok or {args.batch_size} rows each)", flush=True)
    updates_per_epoch = max(1, len(batches) // args.grad_accum)   # real optimizer steps
    total_updates = (args.max_steps or args.epochs * updates_per_epoch)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    # scheduler counts one step() per OPTIMIZER update (called every grad_accum micro-batches)
    sched = get_cosine_schedule_with_warmup(
        opt, max(1, int(0.03 * total_updates)), total_updates)
    # --resume: fast-forward the LR schedule to where the crashed run stopped so the
    # decay continues smoothly (Adam moments restart fresh — a benign warm restart).
    skip_updates = min(args.skip_updates, total_updates) if args.resume else 0
    for _ in range(skip_updates):
        sched.step()

    model.train()
    model.config.use_cache = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    t0, upd, micro, accum_tokens, losses = time.time(), skip_updates, 0, 0, []

    def optimizer_step(epoch):
        # normalize the token-weighted accumulated grads to a true token-mean,
        # then clip/step. Called every grad_accum micro-batches + once at the end
        # to flush the trailing partial window (so no batches are dropped).
        nonlocal accum_tokens, upd
        if accum_tokens == 0:
            return
        inv = 1.0 / accum_tokens
        for p in trainable:
            if p.grad is not None:
                p.grad.mul_(inv)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        accum_tokens = 0
        upd += 1
        if upd % 25 == 0 or upd == 1:
            print(f"  upd {upd}/{total_updates} ep{epoch} loss {losses[-1]:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        # periodic checkpoint: overwrite one adapter dir so a killed run leaves a
        # recent, usable adapter behind. Cheap (LoRA is tens of MB) — see --save-every.
        if args.save_every and upd % args.save_every == 0:
            ck = Path(args.out) / "checkpoint"; ck.mkdir(parents=True, exist_ok=True)
            ts = time.time()
            model.save_pretrained(str(ck)); tok.save_pretrained(str(ck))
            # record progress so a supervisor can --resume ... --skip-updates N exactly
            (ck / "progress.json").write_text(json.dumps(
                {"updates_done": upd, "total": total_updates, "loss": losses[-1]}))
            print(f"    [checkpoint] upd {upd} -> {ck} ({time.time()-ts:.1f}s)", flush=True)

    if args.eval_only:
        train_secs = 0.0
        print("[run_sft] --eval-only: skipping training; evaluating the loaded adapter "
              f"(resume={args.resume or 'NONE — WARNING: fresh random adapter'})", flush=True)
    else:
        stop = False
        skip_micro = skip_updates * args.grad_accum   # micro-batches already trained pre-crash
        seen = 0
        for epoch in range(args.epochs):
            for idxs in batches:
                if seen < skip_micro:                 # fast-skip done windows (same batch order)
                    seen += 1
                    continue
                ii, ll, aa = make_tensor(idxs)
                out = model(input_ids=ii, attention_mask=aa, labels=ll)
                n_tok = int((ll != -100).sum().item())            # supervised tokens
                (out.loss * n_tok).backward()                      # token-weighted accum
                accum_tokens += n_tok
                micro += 1
                losses.append(float(out.loss.detach()))
                if micro % args.grad_accum == 0:
                    optimizer_step(epoch)
                    if args.max_steps and upd >= args.max_steps:
                        stop = True; break
            if stop:
                break
        if not stop:
            optimizer_step(args.epochs - 1)                        # flush trailing window

        out_dir = Path(args.out) / "adapter"
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(out_dir)); tok.save_pretrained(str(out_dir))
        train_secs = time.time() - t0
        # losses can be empty when a resume skips every micro-batch (e.g. --skip-updates
        # equals total_updates and grad_accum divides len(batches)) -> guard the summary
        # so the run proceeds to eval instead of crashing with IndexError.
        loss_str = f"{losses[0]:.2f}->{losses[-1]:.2f}" if losses else "n/a (no micro-batches)"
        print(f"[run_sft] trained {upd} updates, loss {loss_str}, "
              f"{train_secs:.0f}s -> {out_dir}", flush=True)

    # ---- base-vs-tuned eval on the held-out split (deterministic) ----
    # Batched generation: single-sequence decoding is memory-bound (the weights are
    # re-read from VRAM for every token, GPU mostly idle); left-padding a batch of
    # prompts into ONE generate() call amortizes those reads -> much higher throughput.
    def _chat(it, mode):
        prompt = S.build_prompt(it, mode)
        try:
            return tok.apply_chat_template([{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template([{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def gen_many(items, modes):
        """Generate for a batch of (item, mode). Decoder-only models must LEFT-pad so
        every sequence's generated tokens begin at the same offset (plen)."""
        texts = [_chat(it, m) for it, m in zip(items, modes)]
        prev_side = tok.padding_side
        tok.padding_side = "left"
        enc = tok(texts, return_tensors="pt", padding=True).to(device)
        tok.padding_side = prev_side
        o = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                           pad_token_id=(tok.pad_token_id or tok.eos_token_id))
        plen = enc["input_ids"].shape[1]
        return [tok.decode(o[i][plen:], skip_special_tokens=True) for i in range(len(texts))]

    def gen_chunked(items, modes):
        """Batched generation over --eval-batch-sized chunks; preserves input order."""
        out, bs = [], max(1, args.eval_batch)
        for i in range(0, len(items), bs):
            out.extend(gen_many(items[i:i+bs], modes[i:i+bs]))
        return out

    # representative sample: even stride across each family's (ordered) eval slice
    # so we don't take only the first N (ProverQA is ordered easy->medium->hard,
    # so first-N would be all-easy and inflate its accuracy). For the entailment
    # families we ALSO label-stratify to hit ~--neutral-frac neutral overall, so the
    # hard 'does-not-follow' class isn't a tiny (statistically useless) denominator.
    def _stride(items, k):
        k = min(k, len(items))
        if k <= 0:
            return []
        s = len(items) / k
        return [items[int(i * s)] for i in range(k)]

    by_fam = defaultdict(list)
    for it in eval_items:
        by_fam[it["family"]].append(it)
    fams = list(by_fam)
    ent_fams = [f for f in fams if f in S.ENTAILMENT_FAMILIES]
    per = args.eval_per_family
    # per-entailment-family neutral quota s.t. overall neutral ~= neutral_frac
    # (satellite families have no neutral gold; overall = ent_share * within-ent neutral)
    neu_quota = 0
    if ent_fams and args.neutral_frac > 0 and per > 0:
        neu_quota = min(per, round(args.neutral_frac * len(fams) * per / len(ent_fams)))
    picked = []
    for fam, fam_items in by_fam.items():
        n = min(per, len(fam_items))
        if n <= 0:
            continue
        if fam in ent_fams and neu_quota > 0:
            neu = [it for it in fam_items if S.is_neutral_gold(it, "frq")]
            non = [it for it in fam_items if not S.is_neutral_gold(it, "frq")]
            kneu = min(neu_quota, len(neu), n)
            picked += _stride(neu, kneu) + _stride(non, n - kneu)
        else:
            picked += _stride(fam_items, n)

    judge = args.judge_model or None
    # gradeable (item, mode) list, computed once and shared by both conditions
    graded_items = []
    for it in picked:
        mode = "frq" if it["family"] in S.ENTAILMENT_FAMILIES else "mcq"
        if mode == "mcq" and not S.has_mc(it):
            continue
        graded_items.append((it, mode))

    def run_condition(label):
        b = defaultdict(lambda: {"n":0,"parse":0,"correct":0,"neu_n":0,"neu_c":0,"judged":0})
        _t = time.time()
        items = [it for it, _ in graded_items]
        modes = [m for _, m in graded_items]
        comps, bs = [], max(1, args.eval_batch)
        for i in range(0, len(items), bs):
            comps.extend(gen_many(items[i:i+bs], modes[i:i+bs]))
            done = min(i + bs, len(items))
            print(f"  [{label}] {done}/{len(items)} generations "
                  f"({time.time()-_t:.0f}s, {(time.time()-_t)/max(1,done):.1f}s/gen, bs={bs})",
                  flush=True)
        samples = []
        for (it, mode), comp in zip(graded_items, comps):
            g = S.grade(it, comp, mode, judge_model=judge)
            gold = S.credited_answer(it, mode)
            samples.append({"condition": label.strip(), "id": it.get("id"),
                            "family": it["family"], "mode": mode, "gold": gold,
                            "neutral_gold": S.is_neutral_gold(it, mode),
                            "parseable": bool(g["parseable"]), "correct": bool(g["correct"]),
                            "needs_judge": bool(g["needs_judge"]), "completion": comp})
            if g["needs_judge"]:
                continue
            k = b[it["family"]]
            k["n"] += 1; k["parse"] += int(g["parseable"]); k["correct"] += int(bool(g["correct"]))
            k["judged"] += int(g.get("used_judge", False))
            if S.is_neutral_gold(it, mode):
                k["neu_n"] += 1; k["neu_c"] += int(bool(g["correct"]))
        tot = defaultdict(int)
        for v in b.values():
            for kk, vv in v.items(): tot[kk] += vv
        pct = lambda a, d: round(100*a/d, 1) if d else None
        print(f"[{label}] n={tot['n']} parse={pct(tot['parse'],tot['n'])}% "
              f"acc={pct(tot['correct'],tot['n'])}% neutral_recall={pct(tot['neu_c'],tot['neu_n'])}%",
              flush=True)
        return b, tot, samples

    model.eval()
    model.config.use_cache = True                    # KV cache -> fast generation
    try:
        model.gradient_checkpointing_disable()       # training-only; slows generate
    except Exception:
        pass
    print(f"\n[run_sft] base-vs-tuned eval on {len(picked)} held-out items "
          f"({args.eval_per_family}/family)")
    with model.disable_adapter():
        base_b, base_t, base_s = run_condition("base ")
    tuned_b, tuned_t, tuned_s = run_condition("tuned")
    # raw-answer log: every item's completion + gold + verdict, for auditing (e.g.
    # is a 100% neutral recall real, or a tiny/unbalanced denominator?)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with (Path(args.out) / "eval_samples.jsonl").open("w", encoding="utf-8") as fh:
        for r in base_s + tuned_s:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[run_sft] raw eval answers -> {Path(args.out)/'eval_samples.jsonl'} "
          f"({len(base_s)+len(tuned_s)} rows)", flush=True)
    pct = lambda a, d: round(100*a/d, 1) if d else None
    print("\nby family (base_acc -> tuned_acc, n) [neutral]")
    for fam in sorted(set(base_b) | set(tuned_b)):
        bb, tt = base_b[fam], tuned_b[fam]
        neu = f"  neutral {pct(bb['neu_c'],bb['neu_n'])}%->{pct(tt['neu_c'],tt['neu_n'])}%" if tt["neu_n"] else ""
        print(f"  {fam:9} {pct(bb['correct'],bb['n'])}% -> {pct(tt['correct'],tt['n'])}% (n={tt['n']}){neu}")

    # ---- LGMT consistency (MVR / HDR): does the label survive a logic-preserving
    # reformulation? Append an irrelevant premise (the P3 over-inference probe) and
    # check the entailment label is unchanged. MVR = label-flip rate; HDR = flips among
    # answers that were correct on the source (the defects static accuracy hides). ----
    _IRR = "Additionally, an unrelated fact holds: Quennell owns a teal kite."
    def lgmt_consistency(label):
        ent = [it for it in eval_items if it["family"] in S.ENTAILMENT_FAMILIES]
        n = min(args.lgmt_eval, len(ent))
        if n <= 0:
            return None
        stepn = len(ent) / n
        sample = [ent[int(i * stepn)] for i in range(n)]
        fu_items = []
        for it in sample:
            fu = dict(it); fu["stimulus"] = it["stimulus"].rstrip() + " " + _IRR
            fu_items.append(fu)
        # batched: all source prompts, then all irrelevant-premise follow-ups
        src_out = gen_chunked(sample, ["frq"] * len(sample))
        fu_out = gen_chunked(fu_items, ["frq"] * len(sample))
        viol = hidden = usable = 0
        for it, s_txt, f_txt in zip(sample, src_out, fu_out):
            gold = S.canon_label(it["reference_answer"]) or it["reference_answer"]
            hint = S.neutral_variant(it)
            src = S.parse_label(s_txt, neutral_hint=hint)
            flab = S.parse_label(f_txt, neutral_hint=hint)
            if src is None or flab is None:
                continue
            usable += 1
            if not S.labels_equiv(src, flab):
                viol += 1
                if S.labels_equiv(src, gold):
                    hidden += 1
        mvr, hdr = pct(viol, usable), pct(hidden, usable)
        print(f"[{label}] LGMT consistency (n={usable}): MVR {mvr}%  HDR {hidden}/{usable}={hdr}%",
              flush=True)
        return {"n": usable, "mvr": mvr, "hdr": hdr}

    lgmt = {}
    if args.lgmt_eval:
        print(f"\n[run_sft] LGMT consistency probe ({args.lgmt_eval} entailment items/condition)")
        with model.disable_adapter():
            lgmt["base"] = lgmt_consistency("base ")
        lgmt["tuned"] = lgmt_consistency("tuned")

    # ---- optional satellite FRQ probe (EVAL-ONLY; training stays MCQ). Pose the
    # OPEN-ENDED question (state the flaw / warrant / assumption) instead of MCQ and
    # judge semantic correctness with the gateway. This measures the generation /
    # identifies-target weakness that MCQ recognition hides; expect low absolute
    # scores (train-MCQ/eval-FRQ mismatch) — it's a diagnostic, read base-vs-tuned. ----
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

    @torch.no_grad()
    def gen_prompts_chunked(prompts):   # generate from RAW prompt strings (not S.build_prompt)
        texts = []
        for p in prompts:
            try:
                texts.append(tok.apply_chat_template([{"role": "user", "content": p}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False))
            except TypeError:
                texts.append(tok.apply_chat_template([{"role": "user", "content": p}],
                    tokenize=False, add_generation_prompt=True))
        out, bs = [], max(1, args.eval_batch)
        for i in range(0, len(texts), bs):
            chunk = texts[i:i+bs]
            prev = tok.padding_side; tok.padding_side = "left"
            enc = tok(chunk, return_tensors="pt", padding=True).to(device); tok.padding_side = prev
            o = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                               pad_token_id=(tok.pad_token_id or tok.eos_token_id))
            plen = enc["input_ids"].shape[1]
            out += [tok.decode(o[j][plen:], skip_special_tokens=True) for j in range(len(chunk))]
        return out

    def _judge_frq(candidate, reference, model):
        if not candidate or not candidate.strip():
            return None
        q = ("Grade an open-ended answer to a critical-reasoning question against a reference "
             "answer. Reply with ONLY 'YES' if the candidate makes essentially the same point "
             "as the reference (a defensible paraphrase counts), otherwise 'NO'.\n\n"
             f"Reference:\n{reference}\n\nCandidate:\n{candidate}\n\nSame point? (YES/NO)")
        out = S.call_agent(q, model, timeout=60, max_tokens=8)
        return None if not out else out.strip().upper().startswith("Y")

    frq = {}
    if args.frq_eval:
        jm = args.judge_model or S.DEFAULT_JUDGE_MODEL
        byf = defaultdict(list)
        for it in eval_items:
            if it["family"] in S.MCQ_FAMILIES:
                byf[it["family"]].append(it)
        fpick = []
        for fam, its in byf.items():
            fpick += _stride(its, min(args.frq_eval, len(its)))
        prompts = [_frq_prompt_sat(it) for it in fpick]
        print(f"\n[run_sft] satellite FRQ probe: {len(fpick)} open-ended items, judge={jm}", flush=True)
        def frq_condition(label):
            outs = gen_prompts_chunked(prompts)
            b = defaultdict(lambda: {"n": 0, "c": 0})
            fsamp = []
            for it, o in zip(fpick, outs):
                ref = S.credited_answer(it, "mcq")
                v = _judge_frq(o, ref, jm)
                fsamp.append({"condition": label.strip(), "probe": "frq", "id": it.get("id"),
                              "family": it["family"], "task_type": it.get("task_type"),
                              "reference": ref, "judged_correct": v, "completion": o})
                if v is not None:
                    k = b[it["family"]]; k["n"] += 1; k["c"] += int(v)
            tot = defaultdict(int)
            for v in b.values():
                for kk, vv in v.items(): tot[kk] += vv
            print(f"[{label}] FRQ correctness n={tot['n']} acc={pct(tot['c'],tot['n'])}%", flush=True)
            return ({fam: {"acc": pct(b[fam]['c'], b[fam]['n']), "n": b[fam]['n']} for fam in b},
                    pct(tot['c'], tot['n']), tot['n'], fsamp)
        if fpick:
            with model.disable_adapter():
                fb_by, fb_acc, fb_n, fb_s = frq_condition("base ")
            ft_by, ft_acc, ft_n, ft_s = frq_condition("tuned")
            with (Path(args.out) / "eval_samples.jsonl").open("a", encoding="utf-8") as fh:
                for r in fb_s + ft_s:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            frq = {"judge": jm, "base": {"acc": fb_acc, "n": fb_n, "by_family": fb_by},
                   "tuned": {"acc": ft_acc, "n": ft_n, "by_family": ft_by}}
            if fb_n == 0 and ft_n == 0:
                print("[run_sft] FRQ probe: judge returned nothing (no gateway creds?)", flush=True)

    metrics = {"model": args.model, "device": device, "updates": upd,
               "train_seconds": round(train_secs, 1),
               "eval_only": args.eval_only, "resumed_from": args.resume or None,
               "loss_first": round(losses[0], 3) if losses else None,
               "loss_last": round(losses[-1], 3) if losses else None,
               "eval_per_family": args.eval_per_family, "neutral_frac": args.neutral_frac,
               "neutral_n": base_t['neu_n'],
               "base": {"parse": pct(base_t['parse'], base_t['n']),
                        "acc": pct(base_t['correct'], base_t['n']),
                        "neutral": pct(base_t['neu_c'], base_t['neu_n'])},
               "tuned": {"parse": pct(tuned_t['parse'], tuned_t['n']),
                         "acc": pct(tuned_t['correct'], tuned_t['n']),
                         "neutral": pct(tuned_t['neu_c'], tuned_t['neu_n'])},
               "lgmt": lgmt, "frq": frq}
    Path(args.out).mkdir(parents=True, exist_ok=True)   # eval-only skips the adapter-save mkdir
    (Path(args.out) / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\n[run_sft] metrics -> {Path(args.out)/'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
