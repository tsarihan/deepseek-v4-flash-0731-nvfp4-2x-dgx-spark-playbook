#!/usr/bin/env python3
"""Concurrency sweep for DeepSeek-V4-Flash-0731 on 2x DGX Spark.

Long-answer prompt, one distinct nonce per request so --enable-prefix-caching
cannot make later streams reuse earlier prefill work. Fixed max_tokens so every
stream does the same amount of decode work at every concurrency level.
"""
import argparse, asyncio, json, statistics, time, urllib.error, urllib.request

LONG_ANSWER = (
    "Write a detailed technical explanation of how modern mixture-of-experts "
    "language models are served efficiently across multiple GPUs. Cover expert "
    "routing, KV-cache management, speculative decoding, tensor parallelism, and "
    "pipeline parallelism. Be thorough and use numbered sections."
)


TEMPERATURE = 1.0   # DeepSeek 0731 card + NVIDIA NVFP4 evals both specify 1.0
TOP_P = 1.0         # 0.95 for agentic scenarios, 1.0 otherwise


def stream_one(base_url, model, nonce, max_tokens, timeout):
    """One streaming request. Returns timing + token counts."""
    prompt = f"[request id {nonce}] {LONG_ANSWER}"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first = None
    usage = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                ev = json.loads(line[6:])
                choices = ev.get("choices") or []
                delta = choices[0].get("delta", {}) if choices else {}
                if first is None and (
                    delta.get("content")
                    or delta.get("reasoning")
                    or delta.get("reasoning_content")
                ):
                    first = time.perf_counter()
                if ev.get("usage"):
                    usage = ev["usage"]
    except Exception as e:  # keep the sweep alive; record the failure
        return {"ok": False, "error": f"{type(e).__name__}: {e}"[:200]}
    done = time.perf_counter()
    if usage is None or first is None:
        return {"ok": False, "error": "no usage/first-token in stream"}
    out_tok = usage.get("completion_tokens", 0)
    in_tok = usage.get("prompt_tokens", 0)
    decode_s = max(done - first, 1e-9)
    return {
        "ok": True,
        "ttft_s": first - started,
        "decode_s": decode_s,
        "total_s": done - started,
        "prompt_tokens": in_tok,
        "completion_tokens": out_tok,
        # per-stream decode rate: tokens after the first, over decode window
        "decode_tps": (out_tok - 1) / decode_s if out_tok > 1 else 0.0,
        "t_start": started,
        "t_done": done,
    }


async def run_case(base_url, model, conc, max_tokens, timeout, tag):
    nonces = [f"{tag}-c{conc}-s{i}-{int(time.time()*1000)}" for i in range(conc)]
    wall0 = time.perf_counter()
    results = await asyncio.gather(
        *[
            asyncio.to_thread(stream_one, base_url, model, n, max_tokens, timeout)
            for n in nonces
        ]
    )
    wall = time.perf_counter() - wall0
    ok = [r for r in results if r.get("ok")]
    bad = [r for r in results if not r.get("ok")]
    if not ok:
        return {"concurrency": conc, "success": 0, "failed": len(bad),
                "errors": [b["error"] for b in bad[:3]]}
    total_out = sum(r["completion_tokens"] for r in ok)
    return {
        "concurrency": conc,
        "success": len(ok),
        "failed": len(bad),
        "errors": [b["error"] for b in bad[:3]],
        "wall_s": round(wall, 2),
        "ttft_mean_s": round(statistics.fmean(r["ttft_s"] for r in ok), 3),
        "ttft_p50_s": round(statistics.median(r["ttft_s"] for r in ok), 3),
        "ttft_max_s": round(max(r["ttft_s"] for r in ok), 3),
        "out_tokens_total": total_out,
        "out_tokens_mean": round(total_out / len(ok), 1),
        "prompt_tokens_mean": round(
            statistics.fmean(r["prompt_tokens"] for r in ok), 1),
        # per-stream decode throughput (what one user feels)
        "per_stream_decode_tps": round(
            statistics.fmean(r["decode_tps"] for r in ok), 2),
        # aggregate server decode throughput over the whole batch wall time
        "aggregate_tps": round(total_out / wall, 2),
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8888/v1")
    ap.add_argument("--model", default="deepseek-v4-flash-0731")
    ap.add_argument("--concurrency", default="1,4,8,16,32,48")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default="sweep-results.json")
    a = ap.parse_args()

    levels = [int(x) for x in a.concurrency.split(",")]

    print(f"warmup (2 requests, model={a.model}) ...", flush=True)
    w = await run_case(a.base_url, a.model, 2, min(128, a.max_tokens), a.timeout,
                       a.tag + "-warm")
    print(f"  warmup: {w.get('success')}/2 ok, "
          f"ttft={w.get('ttft_mean_s')}s\n", flush=True)

    rows = []
    for c in levels:
        print(f"=== concurrency {c} ===", flush=True)
        r = await run_case(a.base_url, a.model, c, a.max_tokens, a.timeout, a.tag)
        rows.append(r)
        print(json.dumps(r, indent=2), flush=True)
        with open(a.out, "w") as f:
            json.dump({"tag": a.tag, "model": a.model,
                       "max_tokens": a.max_tokens, "rows": rows}, f, indent=2)
        await asyncio.sleep(5)  # let the server drain between cases

    print("\n  conc | ok | TTFT p50 | per-stream tok/s | aggregate tok/s")
    print("  -----+----+----------+------------------+----------------")
    for r in rows:
        if r.get("success"):
            print(f"  {r['concurrency']:>4} | {r['success']:>2} | "
                  f"{r['ttft_p50_s']:>8.2f} | {r['per_stream_decode_tps']:>16.2f} | "
                  f"{r['aggregate_tps']:>14.2f}")
        else:
            print(f"  {r['concurrency']:>4} | 0 FAILED: {r.get('errors')}")


if __name__ == "__main__":
    asyncio.run(main())
