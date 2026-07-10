#!/bin/bash
# Self-healing overnight supervisor (sole owner once the initial wrapper is retired).
#
#  * 4B -> FULL epoch. While a run_sft 4B process is alive, just watch. If it dies
#    before the FINAL adapter (runs/gpu_4b/adapter/) exists, relaunch it --resume
#    from the last checkpoint, skipping the updates already done (progress.json, or
#    the "[checkpoint] upd N" line in overnight.log for the pre-existing old-code run).
#    This is why 4B is self-healing: it trains at the ~7900 MiB VRAM edge for ~11 h,
#    where this GPU has flaked, and no one is awake to restart it.
#  * Anti-thrash: if several consecutive relaunches make NO new progress, give up
#    (GPU wedged) rather than spin. Hard wall-clock cap too.
#  * 2B -> best-effort after the 4B epoch completes, checkpointed every 100 updates.
#    Tries bf16 FIRST (tensor cores + full fla fast path, no nf4 dequant -> much
#    faster + higher quality; ~4 GB weights fit 8 GB). Falls back to 4-bit QLoRA
#    only if bf16 leaves no checkpoint at all (a fast crash = OOM). Not auto-resumed,
#    so the user cutting it short leaves the last checkpoint as a usable adapter.
#  * SEPARATE out dirs (gpu_4b / gpu_2b): 2B never touches 4B weights/metrics.
set -u
cd "$(dirname "$0")" || exit 1
VENV=/home/vs/AlphaAI/llm/specialized-llm/02-experiments/critical-reasoning-slm/.venv/bin/python
DEADLINE=$(( $(date +%s) + 14*3600 ))    # absolute safety stop, 14 h from now

log(){ echo "[sup $(date '+%F %T')] $*"; }

alive4b(){ pgrep -f "run_sft.py --model Qwen/Qwen3.5-4B" >/dev/null; }
alive2b(){ pgrep -f "run_sft.py --model Qwen/Qwen3.5-2B" >/dev/null; }
done4b(){ [ -f runs/gpu_4b/adapter/adapter_config.json ]; }
done2b(){ [ -f runs/gpu_2b/adapter/adapter_config.json ]; }

# updates already completed for 4B: prefer progress.json, else parse overnight.log
progress4b(){
  local p
  p=$(grep -o '"updates_done"[^,}]*' runs/gpu_4b/checkpoint/progress.json 2>/dev/null \
      | grep -o '[0-9]\+' | tail -1)
  if [ -z "$p" ]; then
    p=$(grep -o 'checkpoint\] upd [0-9]\+' runs/overnight.log 2>/dev/null \
        | grep -o '[0-9]\+' | tail -1)
  fi
  echo "${p:-0}"
}

launch4b(){
  local skip resume=""
  if [ -f runs/gpu_4b/checkpoint/adapter_config.json ]; then
    skip=$(progress4b)
    resume="--resume runs/gpu_4b/checkpoint --skip-updates ${skip:-0}"
    log "relaunch 4B resuming from checkpoint (skip ${skip:-0} updates)"
  else
    log "launch 4B fresh (no checkpoint yet)"
  fi
  "$VENV" run_sft.py --model Qwen/Qwen3.5-4B --load-4bit --epochs 1 \
    --batch-size 2 --max-tokens 1024 --grad-accum 2 \
    --save-every 100 --eval-per-family 30 --lgmt-eval 20 --out runs/gpu_4b $resume
  log "4B attempt exited rc=$?"
}

log "supervisor online; watching 4B (live run continues writing overnight.log)"

# ---- Phase 4B: drive to a full epoch, self-heal on crash ----
stall=0; last=-1
while ! done4b; do
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then log "DEADLINE reached in phase 4B; stopping"; exit 0; fi
  if alive4b; then sleep 60; continue; fi          # something is training -> just watch
  cur=$(progress4b)
  if [ "${cur:-0}" -le "$last" ]; then stall=$((stall+1)); else stall=0; fi
  last=${cur:-0}
  if [ "$stall" -ge 4 ]; then
    log "4B made no progress past $last updates over $stall attempts -> GPU likely wedged; giving up"
    break
  fi
  sleep 15                                         # let VRAM free after a crash
  launch4b
  sleep 5
done

# ---- Phase 2B: single best-effort attempt, only if 4B truly completed ----
if done4b; then
  log "4B epoch COMPLETE (runs/gpu_4b/adapter present)."
  if done2b; then
    log "2B already complete; nothing to do."
  elif alive2b; then
    log "2B already running; leaving it."
  elif [ "$(date +%s)" -ge "$DEADLINE" ]; then
    log "past deadline; NOT starting 2B."
  else
    # Attempt 1: bf16 (NO --load-4bit). 2B in bf16 is ~4 GB of weights -> fits 8 GB
    # with headroom, and gets the tensor-core matmuls + full fla fast path (no nf4
    # dequant on every matmul) -> much faster per step AND higher quality than 4-bit.
    log "launching 2B in bf16 (batch8, checkpointed; safe to cut short)"
    "$VENV" run_sft.py --model Qwen/Qwen3.5-2B --epochs 1 \
      --batch-size 8 --max-tokens 1024 --grad-accum 2 \
      --save-every 100 --eval-per-family 30 --lgmt-eval 20 --out runs/gpu_2b
    log "2B bf16 attempt exited rc=$?"
    # Fall back to 4-bit ONLY if bf16 produced nothing (no final adapter AND no
    # checkpoint) -> it crashed before update 100, almost certainly an OOM. If bf16
    # left ANY usable adapter/checkpoint, keep it (a partial bf16 result beats
    # restarting in a different precision).
    if [ ! -f runs/gpu_2b/adapter/adapter_config.json ] \
       && [ ! -f runs/gpu_2b/checkpoint/adapter_config.json ]; then
      if [ "$(date +%s)" -ge "$DEADLINE" ]; then
        log "2B bf16 failed and past deadline; NOT retrying in 4-bit."
      else
        log "2B bf16 left no checkpoint (likely OOM) -> retrying in 4-bit QLoRA"
        rm -rf runs/gpu_2b
        "$VENV" run_sft.py --model Qwen/Qwen3.5-2B --load-4bit --epochs 1 \
          --batch-size 8 --max-tokens 1024 --grad-accum 2 \
          --save-every 100 --eval-per-family 30 --lgmt-eval 20 --out runs/gpu_2b
        log "2B 4-bit fallback exited rc=$?"
      fi
    else
      log "2B bf16 produced a usable adapter/checkpoint -> keeping it (no 4-bit fallback)"
    fi
  fi
else
  log "4B did NOT complete (stall/deadline) -> per instructions, NOT starting 2B."
fi
log "supervisor done."
