#!/usr/bin/env python3
"""Assemble a SINGLE self-contained Colab file from slm_core.py + run_sft.py.

Rather than hand-retype ~750 lines (and risk drift from the tested code), this
stitches the two real modules into one pasteable script: config vars replace the
argparse block, __file__/DATA become a configurable DATA_DIR, and a data-loading
preamble (Colab upload fallback) + pip install are prepended. Re-run after editing
either source to refresh colab_standalone.py.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
core = (HERE / "slm_core.py").read_text(encoding="utf-8")
run = (HERE / "run_sft.py").read_text(encoding="utf-8")

# ---- transform slm_core.py -------------------------------------------------
# drop the __main__ self-test
assert '\nif __name__ == "__main__":' in core
core = core.split('\nif __name__ == "__main__":')[0].rstrip() + "\n"
# __file__ doesn't exist in a pasted cell -> configurable data dir
assert 'HERE = Path(__file__).resolve().parent\nDATA = HERE / "data"' in core
core = core.replace('HERE = Path(__file__).resolve().parent\nDATA = HERE / "data"',
                    'HERE = Path(".")\nDATA = Path(DATA_DIR)')
# _load_dotenv()/_client() reference os.environ -> ensure os is imported (add only if absent)
if "\nimport os\n" not in core:
    core = core.replace("import hashlib\n", "import hashlib\nimport os\n", 1)
# strip the module docstring (keep it lean); keep from `from __future__`
core = core[core.index("from __future__"):]
# a single `from __future__` goes at the very top of the combined file (below)
core = core.replace("from __future__ import annotations\n", "", 1)

# ---- transform run_sft.py --------------------------------------------------
run = run[run.index("from __future__"):]                       # drop shebang + docstring
run = run.replace("from __future__ import annotations\n", "", 1)   # single copy at file top
run = run.replace("HERE = Path(__file__).resolve().parent\n", "")
run = run.replace('os.environ.setdefault("HF_HOME", str(HERE.parent / ".hf_cache"))\n', "")
run = run.replace("import slm_core as S  # noqa: E402\n",
                  "import sys as _sys\nS = _sys.modules[__name__]   # slm_core inlined above\n")

# replace the whole argparse block with a config-driven SimpleNamespace
a = run.index("    ap = argparse.ArgumentParser()")
b = run.index("    args = ap.parse_args()") + len("    args = ap.parse_args()")
ARGS = '''    from types import SimpleNamespace
    args = SimpleNamespace(
        model=MODEL_ID, epochs=EPOCHS, batch_size=BATCH_SIZE, max_tokens=MAX_TOKENS,
        grad_accum=GRAD_ACCUM, lr=LR, lora_r=LORA_R, lora_alpha=LORA_ALPHA,
        max_len=MAX_LEN, max_steps=MAX_STEPS, save_every=SAVE_EVERY,
        train_file=(TRAIN_FILE or ""), eval_per_family=EVAL_PER_FAMILY,
        max_new_tokens=MAX_NEW_TOKENS, eval_batch=EVAL_BATCH, lgmt_eval=LGMT_EVAL,
        neutral_frac=NEUTRAL_FRAC, frq_eval=FRQ_EVAL,
        judge_model=JUDGE_MODEL, out=OUT_DIR, load_4bit=LOAD_4BIT,
        resume=RESUME, skip_updates=SKIP_UPDATES, eval_only=EVAL_ONLY, cpu=CPU)'''
run = run[:a] + ARGS + run[b:]

# swap the CLI entrypoint for a Colab driver
assert 'if __name__ == "__main__":\n    raise SystemExit(main())' in run
run = run.replace('if __name__ == "__main__":\n    raise SystemExit(main())',
                  "# ------------------------------- run -------------------------------\n"
                  "_ensure_data()\nmain()")

# ---- header: config + pip install + data loader ----------------------------
HEADER = '''# ==========================================================================
#  Critical-reasoning SLM — SINGLE-FILE Colab SFT (Qwen3.5)
#  Paste this whole file into ONE Colab cell (Runtime -> GPU) and run it.
#
#  DATA: it needs the pre-built corpus (from build_corpus.py):
#     - sft_train.jsonl   (~13 MB)
#     - eval_items.json   (~2 MB)
#  Put both in DATA_DIR below. Easiest options:
#     (a) mount Drive:  from google.colab import drive; drive.mount('/content/drive')
#         then set DATA_DIR to the folder that holds the two files; OR
#     (b) leave DATA_DIR="." and you'll be prompted to upload the two files.
# ==========================================================================
from __future__ import annotations

# ------------------------------ CONFIG ------------------------------
# Default = Qwen3.5-2B in bf16: fast (tensor cores, no nf4 dequant), good, fits any
# Colab GPU. FLAGSHIP (best accuracy: ~82% on the held-out slice, +23 over base) is
# Qwen3.5-4B via 4-bit QLoRA — uncomment the flagship block. (0.8B was the OLD default.)
MODEL_ID        = "Qwen/Qwen3.5-2B"
LOAD_4BIT       = False     # bf16. Set True for nf4 QLoRA (needed to fit 4B).
BATCH_SIZE      = 8         # max rows/batch (token budget also caps it)
MAX_TOKENS      = 3072      # token budget per batch (rows x longest row)
GRAD_ACCUM      = 4
# --- FLAGSHIP 4B (best results) — uncomment on a 16 GB+ Colab GPU (T4/L4/A100): ---
# MODEL_ID = "Qwen/Qwen3.5-4B"; LOAD_4BIT = True
# BATCH_SIZE = 4; MAX_TOKENS = 1536; GRAD_ACCUM = 2
# (8 GB sm_120 laptop only: BATCH_SIZE=2, MAX_TOKENS=1024, and do NOT set
#  PYTORCH_CUDA_ALLOC_CONF=expandable_segments. Colab T4/A100 have no such limit.)
EPOCHS          = 1
LR              = 2e-4
LORA_R          = 16
LORA_ALPHA      = 32
MAX_LEN         = 1024
MAX_STEPS       = 0         # 0 = full EPOCHS
SAVE_EVERY      = 100       # checkpoint adapter every N updates (0 = only at end)
EVAL_PER_FAMILY = 40
MAX_NEW_TOKENS  = 512
EVAL_BATCH      = 8         # batched eval generation (~3x faster; lower if OOM)
NEUTRAL_FRAC    = 0.25      # over-sample neutral/non-entailment to ~25% of the eval
LGMT_EVAL       = 20        # LGMT consistency probe items/condition (0 = off)
FRQ_EVAL        = 0         # >0 => open-ended satellite probe, N items/family (needs a judge)
JUDGE_MODEL     = ""        # e.g. "claude-haiku-4-5" / "claude-opus-4-8" for parse-fallback + FRQ judge
# --- judge/FRQ grading credentials (blank = no judging) ---
ANTHROPIC_BASE_URL   = ""   # gateway URL (blank = api.anthropic.com)
ANTHROPIC_AUTH_TOKEN = ""   # Bearer token for a gateway  (OR use ANTHROPIC_API_KEY)
ANTHROPIC_API_KEY    = ""   # x-api-key (if not using a gateway)
DATA_DIR        = "."       # folder holding sft_train.jsonl + eval_items.json
OUT_DIR         = "runs/colab"
TRAIN_FILE      = ""        # "" => DATA_DIR/sft_train.jsonl (e.g. sft_train_stepb.jsonl)
RESUME          = ""        # dir of a prior checkpoint adapter to resume from
SKIP_UPDATES    = 0         # with RESUME: updates already done
EVAL_ONLY       = False     # True => skip training, just eval RESUME's adapter
CPU             = False
# --------------------------------------------------------------------
import os as _os
for _k, _v in (("ANTHROPIC_BASE_URL", ANTHROPIC_BASE_URL),
               ("ANTHROPIC_AUTH_TOKEN", ANTHROPIC_AUTH_TOKEN),
               ("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY)):
    if _v:
        _os.environ[_k] = _v   # judge/FRQ grading reads these (slm_core._client)

import subprocess as _sp, sys as _sys0
def _pip():
    pkgs = ["transformers>=4.51", "peft>=0.11", "accelerate>=0.30", "safetensors>=0.4"]
    if LOAD_4BIT:
        pkgs.append("bitsandbytes>=0.43")
    # Qwen3.5 linear-attention fast path (~10x); optional, skip failure quietly
    _sp.run([_sys0.executable, "-m", "pip", "install", "-q", *pkgs], check=False)
    _sp.run([_sys0.executable, "-m", "pip", "install", "-q", "flash-linear-attention"], check=False)
_pip()

from pathlib import Path as _Path
def _ensure_data():
    """Make sure sft_train.jsonl + eval_items.json exist under DATA_DIR (upload if not)."""
    need = ["sft_train.jsonl", "eval_items.json"]
    missing = [f for f in need if not (_Path(DATA_DIR) / f).exists()]
    if not missing:
        return
    try:
        from google.colab import files          # noqa
        print(f"[data] upload these files: {missing}")
        up = files.upload()
        for name in up:
            dest = _Path(DATA_DIR) / _Path(name).name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if _Path(name) != dest:
                _Path(dest).write_bytes(up[name])
    except Exception as e:
        raise SystemExit(
            f"[data] missing {missing} under DATA_DIR={DATA_DIR!r} and no Colab upload "
            f"available ({e}). Put the two corpus files there (build_corpus.py produces them).")
    still = [f for f in need if not (_Path(DATA_DIR) / f).exists()]
    if still:
        raise SystemExit(f"[data] still missing after upload: {still}")

'''

banner_core = "\n# ===================== inlined slm_core.py =====================\n"
banner_run = "\n\n# ============ inlined run_sft.py (train + batched eval) ============\n"
out = HEADER + banner_core + core.rstrip() + "\n" + banner_run + run

dest = HERE / "colab_standalone.py"
dest.write_text(out, encoding="utf-8")
print(f"wrote {dest} ({len(out.splitlines())} lines)")
