#!/usr/bin/env python3
"""Pre-flight for transcoding the MTP experts: does every MTP expert fit the E4M3 window?

The transcode is only lossless if, for each expert-weight group, every non-zero E8M0
exponent lands in E4M3's representable range after the global shift:

    G      = 8 - max(e_old)          (so the largest block maps to e_new = 8, E4M3 max normal)
    e_new  = e_old + G               must satisfy  -6 <= e_new <= 8

max is 8 by construction, so the binding constraint is the MINIMUM: an expert whose block
exponents span more than 14 powers of two cannot be represented and the transcoder aborts.
This checks that BEFORE spending hours rewriting 164 GB.

w1 and w3 SHARE one G (union of their exponents) because the loader couples [:,0]/[:,1];
w2 is independent.
"""
import json, os, struct, sys
import numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else "/src"

idx = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))["weight_map"]
_hdr_cache = {}


def read_u8(name):
    shard = idx[name]
    path = os.path.join(SRC, shard)
    if shard not in _hdr_cache:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n).decode())
        _hdr_cache[shard] = (hdr, 8 + n)
    hdr, base = _hdr_cache[shard]
    s, e = hdr[name]["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + s)
        return np.frombuffer(f.read(e - s), dtype=np.uint8)


blocks = sorted({k.split(".")[1] for k in idx if k.startswith("mtp.") and "experts" in k})
experts = sorted({int(k.split(".experts.")[1].split(".")[0])
                  for k in idx if k.startswith("mtp.0.ffn.experts.")})
print("MTP blocks: %s   experts per block: %d" % (blocks, len(experts)))

worst_span = -1
worst_id = None
n_bad = 0
n_checked = 0
has_ff = 0

for B in blocks:
    for E in experts:
        # w1 UNION w3 share one G; w2 independent
        for group, ws in (("w13", ["w1", "w3"]), ("w2", ["w2"])):
            lo, hi = None, None
            for wX in ws:
                nm = "mtp.%s.ffn.experts.%d.%s.scale" % (B, E, wX)
                if nm not in idx:
                    continue
                a = read_u8(nm)
                if np.any(a == 0xFF):
                    has_ff += 1
                nz = a[a != 0]
                if not len(nz):
                    continue
                mn, mx = int(nz.min()) - 127, int(nz.max()) - 127
                lo = mn if lo is None else min(lo, mn)
                hi = mx if hi is None else max(hi, mx)
            if lo is None:
                continue
            n_checked += 1
            G = 8 - hi
            e_new_lo = lo + G           # e_new_hi is 8 by construction
            span = hi - lo
            if span > worst_span:
                worst_span, worst_id = span, "mtp.%s E%d %s" % (B, E, group)
            if e_new_lo < -6:
                n_bad += 1
                if n_bad <= 5:
                    print("  OUT OF RANGE: mtp.%s E%d %s  G=%d  e_new_lo=%d (span %d > 14)"
                          % (B, E, group, G, e_new_lo, span))

print()
print("groups checked      : %d" % n_checked)
print("widest span         : %d powers of two  (%s)" % (worst_span, worst_id))
print("E4M3 window         : 14 (e_new in [-6, 8])")
print("0xFF bytes found    : %d  (any is a hard abort)" % has_ff)
print()
print("VERDICT: %s" % ("FAIL - %d group(s) exceed the E4M3 window" % n_bad if n_bad
                       else "PASS - every MTP expert fits; transcode is lossless"))
sys.exit(1 if n_bad or has_ff else 0)
