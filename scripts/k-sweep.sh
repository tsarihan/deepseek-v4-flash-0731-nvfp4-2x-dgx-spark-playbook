#!/usr/bin/env bash
# Find the fastest MTP speculative depth K for the NVFP4 build.
#
#   ./k-sweep.sh "3 5"        # K values; K=7 usually already measured
#
# WHY K MATTERS NOW: after transcoding MTP to NVFP4 acceptance is 0.312 (was 0.121),
# but per-position survival is 76/57/37/23/14/8/4 % at depths 0..6 -- i.e. most drafts
# are dead by depth 3. Drafting 7 tokens when ~3 survive costs draft compute and, more
# importantly, reserves K+1 token slots per sequence: the engine warns
# "max_num_scheduled_tokens is set to 8144 based on the speculative decoding settings",
# so a large K eats the --max-num-batched-tokens budget. Lower K may well be FASTER
# despite drafting less.
#
# Each K is a full relaunch of both ranks (~10 min). Measures decode tok/s at conc 1-4
# AND the resulting acceptance, so the two can be read together.
set -uo pipefail

KS="${1:-3 5}"
SEQS="${SEQS:-4}"
UTIL="${UTIL:-0.835}"
CONC="${CONC:-1,2,4}"
D=${REPO:-$HOME/nvfp4-playbook}
S=$D/nvfp4/serve-0731-nvfp4.sh
REMOTE=${REMOTE:-user@node2}
BASE=http://${RANK0_IP:-10.0.0.1}:8891/v1
MODEL=deepseek-v4-flash-0731-nvfp4

acc() {  # print acceptance from live metrics
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
}

for K in $KS; do
  echo "=============================================================="
  echo " MTP_K=$K   seqs=$SEQS util=$UTIL conc=$CONC"
  echo "=============================================================="
  docker rm -f dsv4-0731-nvfp4 >/dev/null 2>&1 || true
  ssh $REMOTE 'docker rm -f dsv4-0731-nvfp4 >/dev/null 2>&1 || true'
  sleep 10

  ssh $REMOTE "NODE_RANK=1 SPEC_DECODE=on MTP_K=$K MAX_NUM_SEQS=$SEQS GPU_UTIL=$UTIL bash $S" >/dev/null 2>&1
  NODE_RANK=0 SPEC_DECODE=on MTP_K=$K MAX_NUM_SEQS=$SEQS GPU_UTIL=$UTIL bash "$S" >/dev/null 2>&1

  echo "  waiting for startup..."
  ok=0
  for _ in $(seq 1 60); do
    curl -sf --max-time 5 "$BASE/models" >/dev/null 2>&1 && { ok=1; break; }
    if docker logs dsv4-0731-nvfp4 2>&1 | tail -60 | grep -qE 'ValueError|Traceback|CUDA out of memory'; then
      echo "  FAILED: $(docker logs dsv4-0731-nvfp4 2>&1 | grep -E 'ValueError|Error' | tail -1 | cut -c1-150)"
      break
    fi
    docker ps --format '{{.Names}}' | grep -q dsv4-0731-nvfp4 || { echo "  rank0 gone"; break; }
    sleep 30
  done
  [ "$ok" = "1" ] || { echo "  SKIPPING K=$K"; continue; }

  # Warm the Triton JIT before measuring. Two reasons, both observed today:
  #  1) cold kernels compile DURING inference and wreck the first rung
  #     (4.66 tok/s cold vs 5.14 warm; TTFT 14.95s vs 0.45s).
  #  2) WORSE: the two ranks compile at DIFFERENT times, so while one JITs the other sits
  #     in a collective. With --async-scheduling + spec decode that skew exceeded the
  #     timeout and WEDGED the pair -- rank1's last log line was
  #     "Triton kernel JIT compilation during inference: _compute_global_topk_indices..."
  #     at 03:52:49, and rank0's collective timed out at 04:03:40. vLLM's own warning says
  #     "consider extending warmup to cover this shape/config".
  # So warm across VARIED SHAPES (prompt length x batch), not one prompt repeated: each
  # distinct shape can trigger a different kernel specialisation.
  echo "  warming (varied shapes)..."
  # include LONG-CONTEXT shapes: 131K NIAH deadlocked on a kernel that short
  # prompts never compile (_topk_log_softmax_kernel). Memory was fine; JIT skew.
  for words in 8 64 512 2048 16384 65536; do
    P=$(python3 -c "print(' '.join('token%d'%i for i in range($words)))")
    for _ in 1 2; do
      curl -sf --max-time 900 "$BASE/completions" -H 'Content-Type: application/json' \
        -d "$(python3 -c "
import json,sys
print(json.dumps({'model':'$MODEL','prompt':sys.argv[1],'max_tokens':128,'temperature':0}))
" "$P")" >/dev/null 2>&1
    done
  done
  # concurrent warmup too -- batched shapes differ from single-stream ones
  for _ in 1 2 3 4; do
    curl -sf --max-time 900 "$BASE/completions" -H 'Content-Type: application/json' \
      -d "{\"model\":\"$MODEL\",\"prompt\":\"Explain speculative decoding in detail.\",\"max_tokens\":128,\"temperature\":0}" >/dev/null 2>&1 &
  done
  wait
  # confirm both ranks are quiet before measuring
  sleep 5
  n0=$(docker logs --since 2m dsv4-0731-nvfp4 2>&1 | grep -c 'JIT compilation during inference')
  n1=$(ssh $REMOTE "docker logs --since 2m dsv4-0731-nvfp4 2>&1 | grep -c 'JIT compilation during inference'" 2>/dev/null || echo '?')
  echo "  JIT compiles in last 2min: rank0=$n0 rank1=$n1 (want 0/0 before measuring)"

  timeout 5400 python3 "$D/sweep.py" --base-url "$BASE" --model "$MODEL" \
    --concurrency "$CONC" --max-tokens 1024 \
    --tag "k$K" --out "$D/results/k$K-sweep.json" 2>&1 | tail -7
  acc
  echo
done

echo "=============================================================="
echo " K SWEEP SUMMARY (NVFP4, spec-on, MTP transcoded)"
echo "=============================================================="
python3 - "$D" "$KS" <<'PY'
import json, os, sys
d, ks = sys.argv[1], sys.argv[2].split()
# K=7 was measured under the tag nvfp4-specon-mtpfixed
files = {k: os.path.join(d, "results", "k%s-sweep.json" % k) for k in ks}
files.setdefault("7", os.path.join(d, "results", "nvfp4-specon-mtpfixed-sweep.json"))
print("%-5s %6s %10s %14s %14s" % ("K", "conc", "TTFT p50", "per-stream", "aggregate"))
print("-" * 56)
best = None
for k in sorted(files, key=lambda x: int(x)):
    p = files[k]
    if not os.path.exists(p):
        print("%-5s (not run)" % k); continue
    for r in json.load(open(p)).get("rows", []):
        if not r.get("success"):
            continue
        print("%-5s %6d %10.3f %14.2f %14.2f" % (
            k, r["concurrency"], r["ttft_p50_s"],
            r["per_stream_decode_tps"], r["aggregate_tps"]))
        if r["concurrency"] == 1 and (best is None or r["per_stream_decode_tps"] > best[1]):
            best = (k, r["per_stream_decode_tps"])
if best:
    print("\nfastest single-stream: K=%s at %.2f tok/s" % best)
PY
