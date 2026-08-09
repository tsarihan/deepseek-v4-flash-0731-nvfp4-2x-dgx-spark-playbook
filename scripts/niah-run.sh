#!/usr/bin/env bash
# NIAH ladder with PER-RUNG long-context warmup.
#
#   ./niah-run.sh <tag> <model> [contexts]
#
# WHY PER-RUNG WARMUP: the MXFP4 ladder deadlocked entering the 131K rung. rank1 JIT-compiled
# `_topk_log_softmax_kernel` at 09:50:40 while rank0 sat in a collective, which timed out at
# 10:05:11 (`Last enqueued NCCL work: -1` on both). Memory was FINE -- 6.6 GB available, 0 MB
# swap -- so this is JIT skew, not exhaustion. A generic warmup at 8/64/512/2048 tokens never
# compiles the kernels a 131K prompt needs. So before each rung we send a SHORT-OUTPUT probe
# AT THAT CONTEXT LENGTH, which forces both ranks to compile the long-context kernels together,
# and only then run the scored needles.
#
# The probe costs one extra prefill per rung (the dominant cost at long context), which is the
# price of not losing the whole ladder to a wedge.
set -uo pipefail

TAG="${1:?usage: niah-run.sh <tag> <model> [contexts]}"
MODEL="${2:?}"
CONTEXTS="${3:-4096,32768,131072,262144}"
D=${REPO:-$HOME/nvfp4-playbook}
BASE=http://${RANK0_IP:-10.0.0.1}:8891/v1
REMOTE=${REMOTE:-user@node2}

jitcount() {  # $1 = container name
  local n0 n1
  n0=$(docker logs --since 3m "$1" 2>&1 | grep -c 'JIT compilation during inference')
  n1=$(ssh -o ConnectTimeout=10 $REMOTE "docker logs --since 3m $1 2>&1 | grep -c 'JIT compilation during inference'" 2>/dev/null || echo '?')
  echo "$n0/$n1"
}

CNAME=$(docker ps --format '{{.Names}}' | grep -E 'dsv4-0731' | head -1)
[ -n "$CNAME" ] || { echo "no dsv4 container running"; exit 1; }
echo "container: $CNAME   model: $MODEL   contexts: $CONTEXTS"

for ctx in ${CONTEXTS//,/ }; do
  echo "--- warming at ctx=$ctx (forces both ranks to JIT the long-context kernels) ---"
  # ~1 word is well under 1 token for this filler, so overshoot then let the server truncate
  # nothing -- we only need the SHAPE, not an exact count.
  P=$(python3 -c "print(' '.join('token%d'%i for i in range(int($ctx/2.7))))")
  python3 - "$BASE" "$MODEL" "$P" <<'PY' || true
import json,sys,urllib.request
base,model,prompt=sys.argv[1:4]
req={"model":model,"prompt":prompt,"max_tokens":4,"temperature":0}
r=urllib.request.Request(base+"/completions",data=json.dumps(req).encode(),
                         headers={"Content-Type":"application/json"})
try:
    with urllib.request.urlopen(r,timeout=3600) as resp:
        json.load(resp)
    print("    warm probe ok")
except Exception as e:
    print("    warm probe failed: %r" % (e,))
PY
  echo "    JIT compiles (rank0/rank1) after warm: $(jitcount "$CNAME")"
  a1=$(free -m|awk '/^Mem:/{print $7}'); s1=$(free -m|awk '/^Swap:/{print $3}')
  echo "    mem: spark-1 avail=${a1}MB swap=${s1}MB"
  if [ "${s1:-0}" -gt 1024 ] 2>/dev/null; then echo "    *** SWAPPING -- abort ***"; exit 1; fi

  echo "--- scoring needles at ctx=$ctx ---"
  timeout 7200 python3 "$D/needles.py" --base-url "$BASE" --model "$MODEL" \
    --contexts "$ctx" --needles 5 --tag "$TAG-$ctx" \
    --out "$D/results/$TAG-niah-$ctx.json" 2>&1 | grep -E 'prompt=|OK |MISS|needles=' | tail -8
  curl -sf --max-time 5 "$BASE/models" >/dev/null 2>&1 || { echo "    SERVER DIED at ctx=$ctx"; exit 1; }
done

echo "=== $TAG NIAH SUMMARY ==="
python3 - "$D" "$TAG" <<'PY'
import json,glob,os,sys
d,tag=sys.argv[1],sys.argv[2]
rows=[]
for f in sorted(glob.glob(os.path.join(d,'results','%s-niah-*.json'%tag))):
    try: j=json.load(open(f))
    except Exception: continue
    for r in j.get('results',[]):
        if 'error' in r: rows.append((r.get('target'),'ERROR',r['error'][:50])); continue
        got=r.get('found',r.get('needles_found')); tot=r.get('needles',5)
        rows.append((r.get('actual_tokens',r.get('target')), '%s/%s'%(got,tot), ''))
print('%-12s %-8s %s' % ('ctx','needles','note'))
for c,n,note in rows: print('%-12s %-8s %s' % (format(c,',') if isinstance(c,int) else c, n, note))
PY
