#!/usr/bin/env bash
# Does MAX_NUM_SEQS explain the user's 40+ tok/s? Test 32 and 48 (playbook values).
#
#   ./seqs-sweep.sh "32 48" [model]
#
# WHY: every benchmark so far used MAX_NUM_SEQS=4 (chosen for the stated 1-4 stream
# workload). The published playbook uses 32, and notes upstream uses 48 but that it
# "crashes with probabilistic" sampling — we use greedy, so 48 is worth a try. A larger
# seq budget changes scheduler behaviour and can raise single-stream throughput.
#
# ⚠ WARMUP MUST MATCH THE MEASUREMENT SHAPE. The previous run wedged because warmup used
# max_tokens=256 while the measurement used 512, so `_prepare_dflash_inputs_kernel`
# compiled DURING the measured run: one rank stalled in the JIT while the other blocked in
# `_ALLGATHER_BASE`, and the 600 s watchdog killed the engine (113 GB free, 0 swap — not
# memory). So warm with the SAME max_tokens, prompt, and concurrency used to measure.
set -uo pipefail

SEQS_LIST="${1:-32 48}"
MODEL="${2:-deepseek-v4-flash-0731-nvfp4}"
K="${K:-7}"
MAXTOK="${MAXTOK:-512}"
D=${REPO:-$HOME/nvfp4-playbook}
S=$D/nvfp4/serve-0731-nvfp4.sh
REMOTE=${REMOTE:-user@node2}
BASE=http://${RANK0_IP:-10.0.0.1}:8891/v1
CNAME=dsv4-0731-nvfp4
PROMPT="Write a thorough technical explanation of how speculative decoding works in large language model serving: the draft model, the verification step, acceptance, and why it speeds up decoding without changing the output distribution."

for SEQS in $SEQS_LIST; do
  # CAPTURE = SEQS*(K+1) grows the cudagraph pool and eats KV; give the bigger runs more util.
  if   [ "$SEQS" -ge 48 ]; then UTIL=0.87
  elif [ "$SEQS" -ge 32 ]; then UTIL=0.86
  else UTIL=0.835; fi

  echo "=============================================================="
  echo " MAX_NUM_SEQS=$SEQS  K=$K  util=$UTIL  (capture=$((SEQS*(K+1))))"
  echo "=============================================================="
  docker rm -f $CNAME >/dev/null 2>&1 || true
  ssh $REMOTE "docker rm -f $CNAME >/dev/null 2>&1 || true"
  sleep 10

  ssh $REMOTE "NODE_RANK=1 SPEC_DECODE=on MTP_K=$K MAX_NUM_SEQS=$SEQS GPU_UTIL=$UTIL bash $S" >/dev/null 2>&1
  NODE_RANK=0 SPEC_DECODE=on MTP_K=$K MAX_NUM_SEQS=$SEQS GPU_UTIL=$UTIL bash "$S" >/dev/null 2>&1

  ok=0
  for _ in $(seq 1 60); do
    curl -sf --max-time 5 "$BASE/models" >/dev/null 2>&1 && { ok=1; break; }
    if docker logs $CNAME 2>&1 | tail -60 | grep -qE 'ValueError|Traceback|CUDA out of memory'; then
      echo "  FAILED TO START: $(docker logs $CNAME 2>&1 | grep -E 'ValueError|Error' | grep -viE 'raise |please check' | tail -1 | cut -c1-170)"
      break
    fi
    docker ps --format '{{.Names}}' | grep -q $CNAME || { echo "  rank0 gone"; break; }
    sleep 30
  done
  [ "$ok" = "1" ] || { echo "  SKIP seqs=$SEQS"; continue; }

  echo "  KV: $(docker logs $CNAME 2>&1 | grep -E 'GPU KV cache size|Maximum concurrency' | tail -2 | sed 's/.*INFO[^]]*] //' | tr '\n' ' ')"

  # warm at the EXACT measurement shape (same max_tokens), 4x
  echo "  warming at max_tokens=$MAXTOK ..."
  for _ in 1 2 3 4; do
    curl -sf --max-time 900 "$BASE/completions" -H 'Content-Type: application/json' \
      -d "$(python3 -c "
import json;print(json.dumps({'model':'$MODEL','prompt':'''$PROMPT''','max_tokens':$MAXTOK,'temperature':0}))")" >/dev/null 2>&1
  done
  n0=$(docker logs --since 3m $CNAME 2>&1 | grep -c 'JIT compilation during inference')
  n1=$(ssh $REMOTE "docker logs --since 3m $CNAME 2>&1 | grep -c 'JIT compilation during inference'" 2>/dev/null || echo '?')
  echo "  JIT during warm: rank0=$n0 rank1=$n1"

  curl -sf --max-time 10 "$BASE/models" >/dev/null 2>&1 || { echo "  WEDGED during warm"; continue; }
  MAXTOK=$MAXTOK "$D/nvfp4/decode-rate.sh" "$MODEL" 3 2>&1 | tail -5
  curl -sf --max-time 20 http://${RANK0_IP:-10.0.0.1}:8891/metrics 2>/dev/null \
    | grep -E '^vllm:spec_decode_num_(drafts|draft_tokens|accepted_tokens)_total' \
    | python3 -c "
import sys,re
m=sys.stdin.read()
def g(n):
    x=re.search(r'vllm:%s_total\S*\s+([0-9.e+]+)'%n,m); return float(x.group(1)) if x else 0
d,dt,a=g('spec_decode_num_drafts'),g('spec_decode_num_draft_tokens'),g('spec_decode_num_accepted_tokens')
print('    acceptance %.3f  accepted/draft %.2f  (%d drafts)' % (a/dt if dt else 0, a/d if d else 0, d))
"
  echo
done
