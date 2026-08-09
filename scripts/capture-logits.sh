#!/usr/bin/env bash
# Capture TEACHER-FORCED logprobs from a running build, for a NUMERICAL A/B gate.
#
# WHY THIS EXISTS (council C.23, re-confirmed 2026-08-08): "the output looks
# coherent" is NOT a valid acceptance gate for a transcoded checkpoint. A uniform
# scale error (e.g. a wrong weight_scale_2 exponent) rescales the whole logit
# vector but leaves RMSNorm-normalised directions and softmax-argmax ORDER intact
# -> greedy decoding still emits fluent, correct-looking text. Only comparing the
# actual logprob VALUES against a reference build catches it.
#
# TEACHER-FORCED, not free-running (council v11 fixed a real flaw in the first
# version of this script): with free-running greedy, the two builds generate their
# OWN continuations, so as soon as they diverge at one token every later position
# is conditioned on different text -- the deltas then measure DIVERGENCE, not
# quantisation error. `echo=true, max_tokens=0` makes both builds score the
# IDENTICAL token sequence at every position. One prefill, no decode: also the
# cheapest probe in power terms that can actually fail.
#
# Coverage: use a LONG passage, not a handful of short prompts. 5 x 24 generated
# tokens touches only a small slice of a 256-expert bank; a few thousand scored
# positions gives ~30x the expert-activation coverage in a single request.
#
# Only one ~160GB model fits per node, so NVFP4 and W4A16 can never be up at the
# same time. Capture each build's logprobs while it is running, then diff offline:
#   (NVFP4 up)  ./capture-logits.sh nvfp4  http://${RANK0_IP:-10.0.0.1}:8891/v1 deepseek-v4-flash-0731-nvfp4
#   (W4A16 up)  ./capture-logits.sh w4a16  http://${RANK0_IP:-10.0.0.1}:8888/v1 deepseek-v4-flash-0731
#   ./compare-logits.py results/logits-nvfp4.json results/logits-w4a16.json
#
# NOISE-FLOOR MODE: run twice against the SAME build with different chunked-prefill
# boundaries to measure this build's own non-determinism floor F, then judge the
# cross-build diff as a multiple of F rather than against a guessed constant:
#   BATCH_HINT=8192 ./capture-logits.sh nvfp4-a <url> <model>
#   BATCH_HINT=2048 ./capture-logits.sh nvfp4-b <url> <model>
set -euo pipefail

TAG="${1:?usage: capture-logits.sh <tag> <base_url> <model>}"
BASE="${2:?}"
MODEL="${3:?}"
OUT="${OUT:-${REPO:-$HOME/nvfp4-playbook}/results/logits-${TAG}.json}"
PASSAGE="${PASSAGE:-${REPO:-$HOME/nvfp4-playbook}/scripts/probe-passage.txt}"
mkdir -p "$(dirname "$OUT")"

[ -f "$PASSAGE" ] || { echo "missing probe passage: $PASSAGE" >&2; exit 1; }

python3 - "$BASE" "$MODEL" "$OUT" "$TAG" "$PASSAGE" <<'PY'
import json, sys, urllib.request

base, model, out, tag, passage_path = sys.argv[1:6]
text = open(passage_path).read()

# Teacher-forced: score the given tokens, generate nothing.
req = {"model": model, "prompt": text, "max_tokens": 0, "echo": True,
       "logprobs": 20, "temperature": 0, "seed": 0}
r = urllib.request.Request(base + "/completions",
                           data=json.dumps(req).encode(),
                           headers={"Content-Type": "application/json"})
with urllib.request.urlopen(r, timeout=1800) as resp:
    body = json.load(resp)

ch = body["choices"][0]
lp = ch.get("logprobs") or {}
tl = lp.get("token_logprobs") or []
nn = [x for x in tl if x is not None]
if len(nn) < 10:
    print("FAIL: prompt_logprobs came back null/empty -- the echo path is not "
          "returning scores under this config. Do NOT interpret a later "
          "'no comparable logprobs' as a model failure; fix this first.",
          file=sys.stderr)
    sys.exit(1)

# mean NLL over scored positions: one robust scalar that catches degradation
# even when every top-1 still agrees.
mean_nll = -sum(nn) / len(nn)

json.dump({"tag": tag, "model": model, "mode": "teacher_forced",
           "scored_positions": len(nn), "mean_nll": mean_nll,
           "tokens": lp.get("tokens"), "token_logprobs": tl,
           "top_logprobs": lp.get("top_logprobs")},
          open(out, "w"))
print("[capture-logits] %s: %d scored positions, mean NLL %.4f nats -> %s"
      % (tag, len(nn), mean_nll, out), file=sys.stderr)
PY
