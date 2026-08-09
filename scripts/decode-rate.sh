#!/usr/bin/env bash
# Measure PURE decode tok/s on a warm server, separating it from TTFT.
#
#   ./decode-rate.sh <model> [n_repeats]
#
# WHY THIS EXISTS: sweep.py reports tokens/WALL_TIME, which folds TTFT, queueing and
# scheduler overhead into the "per-stream decode tps" figure. For a short prompt that is
# close to decode rate, but it is not the same number, and it under-reports. NIAH reports
# decode separately and gave much higher values (e.g. MXFP4 18.81 vs sweep 11.16 at the
# same config). This measures decode the way a user experiences it: time from FIRST token
# to LAST token, divided by tokens generated in that window.
set -uo pipefail

MODEL="${1:?usage: decode-rate.sh <model> [n_repeats]}"
N="${2:-3}"
BASE="${BASE:-http://${RANK0_IP:-10.0.0.1}:8891/v1}"
MAXTOK="${MAXTOK:-512}"

python3 - "$BASE" "$MODEL" "$N" "$MAXTOK" <<'PY'
import json, sys, time, urllib.request

base, model, n, maxtok = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
PROMPT = ("Write a thorough technical explanation of how speculative decoding works in "
          "large language model serving: the draft model, the verification step, acceptance, "
          "and why it speeds up decoding without changing the output distribution.")

ttfts, decs, tots = [], [], []
for i in range(n):
    req = {"model": model, "prompt": PROMPT, "max_tokens": maxtok,
           "temperature": 0, "stream": True,
           "stream_options": {"include_usage": True}}
    r = urllib.request.Request(base + "/completions",
                               data=json.dumps(req).encode(),
                               headers={"Content-Type": "application/json"})
    # ⚠ COUNT TOKENS, NOT SSE CHUNKS. With speculative decoding ONE streamed chunk can
    # carry SEVERAL accepted draft tokens, so counting chunks undercounts badly — measured
    # 160 "tokens" for a response the server reported as 256 (1.6x low), which made decode
    # look like 5.0 tok/s instead of 14.1. Ask for usage in the stream and use the server's
    # own completion_tokens; fall back to chunk count only if usage is absent.
    t0 = time.time(); tf = None; nchunk = 0; ntok = None
    with urllib.request.urlopen(r, timeout=1800) as resp:
        for line in resp:
            if not line.startswith(b"data: ") or b"[DONE]" in line:
                continue
            ch = json.loads(line[6:])
            u = ch.get("usage")
            if u and u.get("completion_tokens"):
                ntok = u["completion_tokens"]
            txt = ch.get("choices", [{}])[0].get("text") if ch.get("choices") else None
            if txt:
                if tf is None:
                    tf = time.time()
                nchunk += 1
    t1 = time.time()
    if ntok is None:
        ntok = nchunk
    if tf is None or ntok < 2:
        print("  run %d: no tokens" % i); continue
    ttft = tf - t0
    dec_window = t1 - tf                    # first token -> last token
    dec_rate = (ntok - 1) / dec_window if dec_window > 0 else 0   # tokens, not chunks
    ttfts.append(ttft); decs.append(dec_rate); tots.append(ntok / (t1 - t0))
    print("  run %d: %d tok | TTFT %.3fs | DECODE %.2f tok/s | end-to-end %.2f tok/s"
          % (i + 1, ntok, ttft, dec_rate, ntok / (t1 - t0)))

if decs:
    print()
    print("  MEDIAN  TTFT %.3fs | DECODE %.2f tok/s | end-to-end %.2f tok/s"
          % (sorted(ttfts)[len(ttfts)//2], sorted(decs)[len(decs)//2],
             sorted(tots)[len(tots)//2]))
    print("  (sweep.py reports the END-TO-END column, not DECODE)")
PY
