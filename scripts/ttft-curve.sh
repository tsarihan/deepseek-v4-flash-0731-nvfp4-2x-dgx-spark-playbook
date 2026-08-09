#!/usr/bin/env bash
# TTFT vs PROMPT LENGTH — the test that actually decides whether the NVFP4
# transcode was worth doing.
#
# WHY: NVFP4's predicted win is TTFT/PREFILL only. Decode is bandwidth-bound and
# both paths move 4-bit weights, so decode gain was always expected to be ~0 (and
# at batch 1 w4a4 can LOSE: you pay per-layer activation quantisation + MoE
# prepare/finalize overhead that W4A16 doesn't, and FP4 tensor cores buy nothing
# with one row). The 1/4/8/16/32 sweep used 75-token prompts, i.e. it measured
# decode with essentially no prefill in it — the one axis NVFP4 was never going
# to win. This measures the axis it should win.
#
# POWER SAFETY: contexts stop at 131072. Prior needle runs completed 5/5 at
# 4096/32768/131072 on spark-1; BOTH power trips were at 262144. So this curve
# stays inside demonstrated-safe territory and stops short of the trip point.
# Do NOT extend this to 262144+ until the power path is physically checked.
#
#   ./ttft-curve.sh <tag> <base_url> <model>
set -uo pipefail

TAG="${1:?usage: ttft-curve.sh <tag> <base_url> <model>}"
BASE="${2:?}"
MODEL="${3:?}"
CONTEXTS="${CONTEXTS:-4096,32768,131072}"
OUT="${OUT:-${REPO:-$HOME/nvfp4-playbook}/results/ttft-${TAG}.json}"
mkdir -p "$(dirname "$OUT")"

python3 - "$BASE" "$MODEL" "$CONTEXTS" "$OUT" "$TAG" <<'PY'
import json, sys, time, urllib.request

base, model, contexts, out, tag = sys.argv[1:6]
rows = []

def count_tokens(text):
    """Ask the server. Do NOT estimate.

    The first version of this script assumed 1 word == 1 token and built the
    prompt as "token0 token1 ...". That tokenizes to ~2.76 tokens per word, so a
    nominal 4096-token prompt was really 11,288 tokens and every prefill rate it
    printed was understated by the same 2.76x. Verify, like needles.py does.
    """
    # /tokenize lives at the SERVER ROOT, not under /v1 (needles.py:48 does the
    # same removesuffix). Posting to /v1/tokenize returns 404.
    r = urllib.request.Request(base.removesuffix("/v1") + "/tokenize",
                               data=json.dumps({"model": model, "prompt": text}).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=600) as resp:
        d = json.load(resp)
    return d.get("count") or len(d.get("tokens", []))


for ctx in [int(c) for c in contexts.split(",")]:
    # Build to a VERIFIED token count: grow the filler until /tokenize agrees,
    # rather than trusting a words-to-tokens guess.
    words = ["token%d" % i for i in range(max(64, ctx // 3))]
    prompt = " ".join(words)
    actual = count_tokens(prompt)
    # converge by scaling the word count on the measured ratio
    for _ in range(6):
        if abs(actual - ctx) <= max(64, ctx * 0.01):
            break
        ratio = ctx / max(actual, 1)
        n = max(16, int(len(words) * ratio))
        words = ["token%d" % i for i in range(n)]
        prompt = " ".join(words)
        actual = count_tokens(prompt)

    req = {
        "model": model, "prompt": prompt,
        "max_tokens": 8,          # short gen: isolate PREFILL, not decode
        "temperature": 0, "stream": True,
    }
    body = json.dumps(req).encode()
    r = urllib.request.Request(base + "/completions", data=body,
                               headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    try:
        with urllib.request.urlopen(r, timeout=1800) as resp:
            for line in resp:
                if line.startswith(b"data: ") and b"[DONE]" not in line:
                    chunk = json.loads(line[6:])
                    if chunk.get("choices", [{}])[0].get("text"):
                        ttft = time.time() - t0
                        break
        total = time.time() - t0
        row = {"context_tokens": ctx, "ttft_s": round(ttft, 3) if ttft else None,
               "total_s": round(total, 3), "ok": ttft is not None}
        if ttft:
            row["prefill_tok_per_s"] = round(ctx / ttft, 1)
    except Exception as e:
        row = {"context_tokens": ctx, "ok": False, "error": repr(e)}
    rows.append(row)
    print("  ctx=%-8d %s" % (ctx, row), flush=True)
    if not row["ok"]:
        print("  stopping: a longer context will not pass where this failed", flush=True)
        break

json.dump({"tag": tag, "model": model, "rows": rows}, open(out, "w"), indent=2)
print("\n  ctx      | TTFT s  | prefill tok/s")
print("  ---------+---------+--------------")
for r in rows:
    if r.get("ok"):
        print("  %-8d | %7.2f | %12.1f" % (r["context_tokens"], r["ttft_s"], r["prefill_tok_per_s"]))
print("\nwrote " + out)
PY
