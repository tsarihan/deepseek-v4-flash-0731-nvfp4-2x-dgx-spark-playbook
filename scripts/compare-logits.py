#!/usr/bin/env python3
"""Numerical A/B gate for the 0731 NVFP4 transcode vs the MXFP4/W4A16 reference.

Usage:
  # measure this build's own non-determinism floor F (same build, two runs)
  compare-logits.py results/logits-nvfp4.json results/logits-nvfp4-run2.json

  # then judge the cross-build diff as a multiple of F
  compare-logits.py results/logits-nvfp4.json results/logits-w4a16.json --floor F

Why not just eyeball the text: a uniform scale error rescales the logit vector but
preserves argmax order, so greedy text stays fluent and correct while the
distribution is wrong. This compares the VALUES, teacher-forced, so both builds
score the identical token sequence and the deltas cannot be contaminated by
generation divergence.

Signals, in increasing order of how much they should worry you:

  1. top-1 agreement  - weakest; survives large uniform errors.
  2. |delta| spread   - magnitude of per-position disagreement. Judge against the
                        MEASURED floor F, not a constant: MXFP4 is W4A16 while this
                        path is W4A4, so activation quantisation noise is expected
                        and a fixed threshold written for weight-only noise
                        (the old median < 0.15) would false-FAIL.
  3. mean NLL delta   - one robust scalar. Real degradation shows up here even at
                        100% top-1 agreement.
  4. mean SIGNED delta - THE SCALE-BUG DETECTOR, and the one gate immune to the
                        W4A16->W4A4 regime change. Quantisation noise of any
                        flavour is zero-centred; a systematic offset is a scale
                        error regardless of spread. DO NOT loosen this gate to
                        make a FAIL go away -- recalibrate the magnitude gates
                        instead, which is what --floor is for.
"""
import json
import math
import sys


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    # Parse positionally, consuming "--floor V" as a PAIR. (Filtering only tokens
    # starting with "--" left the VALUE behind in args, so `--floor 0.1442` gave
    # len(args)==3 and printed the usage text instead of running.)
    argv = sys.argv[1:]
    args, floor, i = [], None, 0
    while i < len(argv):
        if argv[i] == "--floor" and i + 1 < len(argv):
            floor = float(argv[i + 1]); i += 2; continue
        args.append(argv[i]); i += 1
    if len(args) != 2:
        print(__doc__)
        return 2

    A, B = load(args[0]), load(args[1])
    a_name, b_name = args[0].split("/")[-1], args[1].split("/")[-1]

    for d, n in ((A, a_name), (B, n2 := b_name)):
        if d.get("mode") != "teacher_forced":
            print(f"FAIL: {n} is not teacher-forced. Free-running captures measure "
                  "generation divergence, not quantisation error -- recapture with "
                  "the current capture-logits.sh.")
            return 1

    ta, tb = A.get("tokens") or [], B.get("tokens") or []
    la, lb = A.get("token_logprobs") or [], B.get("token_logprobs") or []
    if ta != tb:
        n = sum(1 for x, y in zip(ta, tb) if x != y)
        print(f"FAIL: the two captures scored different token sequences "
              f"({n} positions differ) -- same passage must be used for both.")
        return 1

    deltas = [x - y for x, y in zip(la, lb) if x is not None and y is not None]
    if not deltas:
        print("FAIL: no comparable logprobs")
        return 1

    # top-1 agreement from the per-position top_logprobs dicts
    top1_match = top1_total = 0
    for da, db in zip(A.get("top_logprobs") or [], B.get("top_logprobs") or []):
        if not da or not db:
            continue
        top1_total += 1
        if max(da, key=da.get) == max(db, key=db.get):
            top1_match += 1

    n = len(deltas)
    absd = sorted(abs(d) for d in deltas)
    median_abs, p95_abs = absd[n // 2], absd[int(n * 0.95)]
    mean = sum(deltas) / n
    std = math.sqrt(sum((d - mean) ** 2 for d in deltas) / n)
    sem = std / math.sqrt(n) if n else float("inf")
    z = mean / sem if sem else 0.0
    dnll = A.get("mean_nll", float("nan")) - B.get("mean_nll", float("nan"))

    print(f"A = {a_name}  (mean NLL {A.get('mean_nll'):.4f})")
    print(f"B = {b_name}  (mean NLL {B.get('mean_nll'):.4f})")
    print()
    if top1_total:
        print(f"  top-1 agreement : {top1_match}/{top1_total} "
              f"({100.0 * top1_match / top1_total:.2f}%)")
    print(f"  scored positions: {n}")
    print(f"  median |delta|  : {median_abs:.4f} nats")
    print(f"  p95 |delta|     : {p95_abs:.4f} nats")
    print(f"  mean delta      : {mean:+.4f} nats  (std {std:.4f}, z={z:+.1f})")
    print(f"  delta mean NLL  : {dnll:+.4f} nats/token")
    if floor:
        print(f"  vs floor F      : median is {median_abs / floor:.1f}x F  (F={floor:.4f})")
    print()

    verdict, notes = "PASS", []
    if top1_total and top1_match / top1_total < 0.95:
        verdict = "FAIL"
        notes.append("top-1 agreement below 95%")
    # magnitude gate: relative to the measured floor when we have one
    if floor and median_abs > 8 * floor:
        verdict = "FAIL"
        notes.append(f"median |delta| is {median_abs / floor:.1f}x the build's own "
                     "non-determinism floor")
    elif not floor and median_abs > 0.4:
        verdict = "FAIL"
        notes.append(f"median |delta| {median_abs:.3f} > 0.4 nats (no floor supplied; "
                     "measure F with two same-build runs for a calibrated gate)")
    if dnll == dnll and dnll > 0.1:
        verdict = "FAIL"
        notes.append(f"mean NLL is {dnll:+.3f} nats/token worse -- real degradation")
    # the scale-bug gate -- unchanged across the W4A16->W4A4 regime change
    if abs(mean) > 0.05 and abs(z) > 10:
        verdict = "FAIL"
        notes.append(f"mean delta {mean:+.3f} nats is systematically non-zero "
                     f"(z={z:+.1f}) -- UNIFORM SCALE ERROR (check weight_scale_2), "
                     "not random quant noise")

    print(f"VERDICT: {verdict}")
    for x in notes:
        print(f"  - {x}")
    if verdict == "PASS":
        print("  - deltas are small and zero-centred: consistent with quantisation")
        print("    noise alone, no evidence of a systematic scale error")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
