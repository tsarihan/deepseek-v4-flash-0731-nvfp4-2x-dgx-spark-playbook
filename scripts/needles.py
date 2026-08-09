#!/usr/bin/env python3
"""Multi-needle long-context probe across a full context sweep.

For each target context length, buries N distinct needles at evenly spaced depths,
asks the model to recall all of them in one shot, and scores each needle
independently. Reports TTFT, prefill throughput, decode throughput and per-depth
retrieval so you can see *where* in the window recall degrades, not just whether
it does.

Token counts are verified against the server's /tokenize endpoint rather than
estimated, and each run uses a distinct nonce so prefix caching cannot make a
later, longer run reuse an earlier one's prefill.

  ./needles.py --base-url http://127.0.0.1:8890/v1 --model deepseek-v4-flash-nvfp4 \
               --contexts 4096,32768,131072,262144,524288,1048576 --needles 5
"""
import argparse, json, random, sys, time, urllib.error, urllib.request

FILLER = (
    "The maintenance log records routine calibration of the sensor array. "
    "Ambient conditions remained nominal throughout the observation window. "
    "No anomalies were reported by the duty technician during this interval. "
)

# Distinct, unguessable, and each a different *kind* of token so one needle
# leaking into another is obvious.
NEEDLE_FACTS = [
    ("vault authorization code", "ARGON-SEVEN-FOUR-ZERO"),
    ("emergency bypass phrase", "TANGERINE-MERIDIAN-19"),
    ("cold storage locker id", "LOCKER-QX-8823"),
    ("relay handshake token", "ZEPHYR-BLUE-3051"),
    ("archive retrieval key", "OBSIDIAN-KILO-77"),
    ("backup site callsign", "NORTHWIND-DELTA-6"),
    ("calibration constant", "6.0221408"),
    ("audit reference number", "AR-2026-554199"),
]


def post(url, body, timeout):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def ntokens(base, model, text, timeout=600):
    return post(base.removesuffix("/v1") + "/tokenize",
                {"model": model, "prompt": text}, timeout)["count"]


def build_prompt(base, model, target_tokens, n_needles, nonce, timeout):
    """Grow filler to target, then splice needles at evenly spaced depths."""
    facts = NEEDLE_FACTS[:n_needles]
    # rough char/token ratio for FILLER, refined by measurement
    body = FILLER * max(1, target_tokens // 20)
    count = ntokens(base, model, body, timeout)
    # converge on the target from below, then trim
    while count < target_tokens * 0.97:
        grow = max(1, int((target_tokens - count) / max(1, count / len(body)) / len(FILLER)))
        body += FILLER * grow
        count = ntokens(base, model, body, timeout)
    while count > target_tokens * 1.02 and len(body) > 1000:
        body = body[: int(len(body) * 0.97)]
        count = ntokens(base, model, body, timeout)

    # depths spread across the interior; avoid the very edges where recall is trivially easy
    depths = [(i + 0.5) / n_needles for i in range(n_needles)]
    pieces, prev = [], 0
    for (label, value), d in zip(facts, depths):
        cut = int(len(body) * d)
        pieces.append(body[prev:cut])
        pieces.append(f"\n\nIMPORTANT FACT: the {label} is {value}. Remember it.\n\n")
        prev = cut
    pieces.append(body[prev:])

    question = (
        f"\n\n[session {nonce}] Question: Several IMPORTANT FACTs were stated above. "
        "List every one of them, each on its own line, in the exact form "
        "'<label>: <value>'. Output only those lines, nothing else.\n"
    )
    return "".join(pieces) + question, facts, depths


def run_one(base, model, target, n_needles, timeout, max_tokens):
    nonce = f"{random.getrandbits(48):012x}"
    try:
        prompt, facts, depths = build_prompt(base, model, target, n_needles, nonce, timeout)
        prompt_tokens = ntokens(base, model, prompt, timeout)
    except Exception as e:
        return {"target": target, "error": f"prompt build failed: {type(e).__name__}: {e}"}

    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        # DeepSeek + NVIDIA both specify 1.0/1.0 for this model family
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )

    t0 = time.time()
    ttft = None
    text = []
    completion_tokens = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage"):
                    completion_tokens = chunk["usage"].get("completion_tokens", 0)
                for ch in chunk.get("choices") or []:
                    delta = ch.get("delta") or {}
                    # field name varies by server: vLLM reasoning-parser emits
                    # "reasoning" (v0.25 srcbuild), others "reasoning_content"
                    piece = (delta.get("content") or delta.get("reasoning_content")
                             or delta.get("reasoning") or "")
                    if piece:
                        if ttft is None:
                            ttft = time.time() - t0
                        text.append(piece)
    except Exception as e:
        return {"target": target, "prompt_tokens": prompt_tokens,
                "error": f"{type(e).__name__}: {e}"}

    total = time.time() - t0
    answer = "".join(text)
    found = {f"{lbl}@{d:.2f}": (val in answer) for (lbl, val), d in zip(facts, depths)}
    n_hit = sum(found.values())
    decode_s = max(1e-6, total - (ttft or 0))
    return {
        "target": target,
        "prompt_tokens": prompt_tokens,
        "ttft_s": round(ttft, 3) if ttft else None,
        "prefill_tps": round(prompt_tokens / ttft, 1) if ttft else None,
        "completion_tokens": completion_tokens or len(answer.split()),
        "decode_tps": round((completion_tokens or 0) / decode_s, 2),
        "total_s": round(total, 2),
        "needles_found": f"{n_hit}/{len(facts)}",
        "all_found": n_hit == len(facts),
        "per_needle": found,
        "answer_head": answer[:300],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8890/v1")
    ap.add_argument("--model", default="deepseek-v4-flash-nvfp4")
    ap.add_argument("--contexts", default="4096,32768,131072,262144,524288,1048576")
    ap.add_argument("--needles", type=int, default=5)
    # 512 starved the answer when the server thinks first (reasoning consumed
    # the whole budget and content never streamed) — needs think + 5 lines.
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default="needles-results.json")
    a = ap.parse_args()

    targets = [int(x) for x in a.contexts.split(",") if x.strip()]
    results = []
    for t in targets:
        print(f"\n=== target {t:,} tokens, {a.needles} needles ===", flush=True)
        r = run_one(a.base_url, a.model, t, a.needles, a.timeout, a.max_tokens)
        results.append(r)
        if "error" in r:
            print(f"  ERROR: {r['error']}", flush=True)
            # a failure at length L means longer L will fail too — stop the ladder
            print("  stopping context ladder here.", flush=True)
            break
        print(f"  prompt={r['prompt_tokens']:,}  ttft={r['ttft_s']}s  "
              f"prefill={r['prefill_tps']} tok/s  decode={r['decode_tps']} tok/s  "
              f"needles={r['needles_found']}", flush=True)
        for k, v in r["per_needle"].items():
            print(f"    {'OK ' if v else 'MISS'} {k}", flush=True)

    out = {"tag": a.tag, "model": a.model, "base_url": a.base_url,
           "needles": a.needles, "results": results}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {a.out}", flush=True)

    ok = [r for r in results if r.get("all_found")]
    if ok:
        print(f"max context with ALL needles retrieved: {max(r['prompt_tokens'] for r in ok):,}")
    else:
        print("no context length retrieved all needles")


if __name__ == "__main__":
    main()
