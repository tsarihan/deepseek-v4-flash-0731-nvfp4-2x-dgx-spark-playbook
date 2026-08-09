#!/usr/bin/env python3
"""Deterministic quality probe — run identically with speculation on and off.

Speculative decoding with rejection sampling is meant to be distribution-
preserving, so greedy (temperature 0) outputs should be identical or nearly so.
This records exact outputs plus correctness on checkable questions, so the two
runs can be diffed rather than argued about.
"""
import argparse, json, hashlib, urllib.request

# Questions with objectively checkable answers, plus reasoning-heavy ones where
# a degraded drafter would plausibly show up first.
PROBES = [
    ("arith", "Compute 4871 * 3629. Give only the final number.", "17676859"),
    ("multistep", "A train leaves at 14:35 and travels 3 hours 48 minutes. "
                  "What time does it arrive? Answer HH:MM only.", "18:23"),
    ("logic", "All bloops are razzies. All razzies are lazzies. Are all bloops "
              "lazzies? Answer yes or no only.", "yes"),
    ("recall", "What is the chemical symbol for tungsten? Answer with the "
               "symbol only.", "W"),
    ("code", "Write a Python one-liner that returns the sum of squares of "
             "1..n. Just the code.", None),
    ("longform", "Explain in exactly 3 numbered sentences why MoE models use "
                 "a router.", None),
]


def ask(base, model, prompt, max_tokens=2048):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0, "top_p": 1.0, "max_tokens": max_tokens, "seed": 42}
    req = urllib.request.Request(f"{base}/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    m = d["choices"][0]["message"]
    return (m.get("content") or "").strip(), d.get("usage", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8888/v1")
    ap.add_argument("--model", default="deepseek-v4-flash-0731")
    ap.add_argument("--label", required=True, help="e.g. spec-on / spec-off")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = []
    for name, q, expect in PROBES:
        try:
            text, usage = ask(a.base_url, a.model, q)
            ok = None if expect is None else (
                expect.lower() in text.lower().replace(",", ""))
            rows.append({"probe": name, "answer": text,
                         "sha": hashlib.sha256(text.encode()).hexdigest()[:16],
                         "expected": expect, "correct": ok,
                         "completion_tokens": usage.get("completion_tokens")})
            print(f"[{name}] correct={ok} tokens={usage.get('completion_tokens')} "
                  f":: {text[:90]!r}", flush=True)
        except Exception as e:
            rows.append({"probe": name, "error": f"{type(e).__name__}: {e}"[:200]})
            print(f"[{name}] ERROR {e}", flush=True)

    scored = [r for r in rows if r.get("correct") is not None]
    n_ok = sum(1 for r in scored if r["correct"])
    print(f"\n{a.label}: {n_ok}/{len(scored)} checkable answers correct")
    with open(a.out, "w") as f:
        json.dump({"label": a.label, "rows": rows,
                   "checkable_correct": n_ok, "checkable_total": len(scored)},
                  f, indent=2)


if __name__ == "__main__":
    main()
