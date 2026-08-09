#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transcode DeepSeek-V4-Flash-0731 (MXFP4 experts) -> NVFP4 experts, on disk.

Runs INSIDE the vllm-dsv4:src-sm121 container (torch/safetensors/numpy available,
NO GPU needed -- pure CPU/disk). Mount /data/models read+write.

Why raw-byte I/O (not safetensors.torch.save_file):
  torch 2.11 has float8_e4m3fn but NOT float8_e8m0fn, so safetensors.torch cannot
  load/save E8M0 tensors (attn scales, MTP scales). We parse the safetensors
  header JSON directly and read/write raw tensor bytes, controlling dtypes as
  strings. The E8M0->E4M3 conversion is pure uint8 byte math.

Convention (memory: deepseekv4-nvfp4-modelopt-convention, council-validated):
  Source 0731 expert tensor per weight:  .weight (I8 [R,Kp])  .scale (F8_E8M0 [R,K/32])
  Target NVFP4 expert tensor per weight: .weight (U8 byte-copy)  .weight_scale (F8_E4M3 [R,K/16])
                                          .weight_scale_2 (F32 scalar = 2^-G)  .input_scale (F32 scalar = 1.0)
  G13(L,E) = 8 - max(E8M0 exps over w1 UNION w3)   (w1 & w3 SHARE one G; CT loader couples [:,0]/[:,1])
  G2(L,E)  = 8 - max(E8M0 exps over w2)            (independent)
  E8M0 byte b -> e_old = b-127 ; e_new = e_old+G ; E4M3 byte = (e_new+7)<<3 ; valid e_new in [-6,8]
  b==0x00 (all-zero block) -> emit E4M3 0x01 (min subnormal) ; b==0xFF -> abort.
  G computed EXCLUDING 0x00 bytes.
  E8M0 [R,C] (block-32, 32 packed nibbles/scale) -> E4M3 [R,2C] (block-16): each byte -> two identical bytes.

Everything else (attn FP8, shared_experts FP8, norms BF16, embed, MTP experts MXFP4, hc_*) -> VERBATIM copy.
MTP stays MXFP4 (NOT transcoded): mtp.0.ffn.experts.E.w{1,2,3}.{weight,scale} copied as-is.

Usage:
  python3 transcode_0731_to_nvfp4.py transcode SRC_DIR OUT_DIR [--shards lo:hi] [--resume]
  python3 transcode_0731_to_nvfp4.py config    SRC_DIR REF_DIR OUT_DIR   # write merged config.json
  python3 transcode_0731_to_nvfp4.py verify    SRC_DIR OUT_DIR           # bit-exactness check
  python3 transcode_0731_to_nvfp4.py index     SRC_DIR OUT_DIR           # (re)write model.safetensors.index.json
"""
import os, sys, json, struct, math, re, shutil, glob

# ----------------------------------------------------------------------------- safetensors raw I/O
def read_header(path):
    """Return (header_dict, header_len) where header_len includes any space padding."""
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        hdr = f.read(n)
    return json.loads(hdr.decode('utf-8')), n

def tensor_data(path, header, name):
    """Read raw bytes of one tensor (absolute offset computed from header data_offsets)."""
    meta = header[name]
    s, e = meta['data_offsets']
    with open(path, 'rb') as f:
        f.seek(8 + len(json.dumps(header).encode('utf-8')) + (0))  # not used; use stored offsets
    # NOTE: data_offsets are relative to the END of the header (i.e. 8 + header_padded_len).
    # We need the actual header padded length, which we have as the 8-byte prefix length.
    pass

def _data_base(path):
    """Absolute byte offset where tensor data region begins = 8 + header_len."""
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
    return 8 + n

def read_tensor_bytes(path, data_base, rel_start, rel_end):
    with open(path, 'rb') as f:
        f.seek(data_base + rel_start)
        return f.read(rel_end - rel_start)

def write_safetensors(path, tensors):
    """tensors: list of (name, dtype_str, shape_list, raw_bytes). Writes a valid safetensors file."""
    # compute data_offsets
    header = {}
    off = 0
    blobs = []
    for name, dtype, shape, data in tensors:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(data)]}
        blobs.append(data)
        off += len(data)
    hdr_json = json.dumps(header, separators=(',', ':')).encode('utf-8')
    # pad header with spaces to 8-byte align the data start (reference impl behavior)
    pad = (8 - ((8 + len(hdr_json)) % 8)) % 8
    hdr_padded = hdr_json + b' ' * pad
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(hdr_padded)))
        f.write(hdr_padded)
        for b in blobs:
            f.write(b)

# ----------------------------------------------------------------------------- E8M0 -> E4M3
def e8m0_to_e4m3(src_u8, G):
    """src_u8: np.uint8 array of E8M0 bytes. Returns np.uint8 array SAME SHAPE of E4M3 bytes."""
    import numpy as np
    out = np.empty_like(src_u8)
    nonzero = src_u8 != 0
    # 0x00 -> 0x01 (min subnormal, all-zero block)
    out[~nonzero] = 0x01
    if np.any(src_u8 == 0xFF):
        bad = int(np.sum(src_u8 == 0xFF))
        raise ValueError(f"E8M0 byte 0xFF encountered ({bad} times) -- abort")
    e_old = src_u8.astype(np.int32) - 127          # biased exponent
    e_new = e_old + G
    # guard range for non-zero
    lo = e_new[nonzero].min(initial=8)
    hi = e_new[nonzero].max(initial=-6)
    if hi > 8:
        raise ValueError(f"e_new max {hi} > 8 -- G too large (G={G})")
    if lo < -6:
        # report how many would be subnormal-out-of-range
        nbad = int(np.sum(nonzero & (e_new < -6)))
        raise ValueError(f"e_new min {lo} < -6 ({nbad} blocks out of E4M3 range; span too wide) -- abort")
    e4m3_field = e_new + 7                          # E4M3 biased exponent field
    byte = (e4m3_field.astype(np.int32) << 3) & 0xFF
    out[nonzero] = byte[nonzero].astype(np.uint8)
    # flag if we emitted any subnormals (e_new < 1) -- info only
    n_sub = int(np.sum(nonzero & (e_new < 1)))
    if n_sub:
        print(f"      [info] {n_sub} scale blocks emitted as E4M3 subnormals (e_new<1, G={G})", flush=True)
    return out

# ----------------------------------------------------------------------------- tensor classification
# Main-layer experts: layers.L.ffn.experts.E.wX.{weight,scale}
# MTP experts:        mtp.B.ffn.experts.E.wX.{weight,scale}   (B in 0..2)
# IDENTICAL structure and geometry -- both are I8 [2048,2048] + F8_E8M0 [2048,128] -- so the
# SAME transcode applies. MTP is keyed as layer (1000+B) to keep the (L,E) G-dict flat while
# staying disjoint from the 0..42 main layers.
EXPERT_RE = re.compile(r'^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$')
MTP_RE    = re.compile(r'^mtp\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$')
MTP_LAYER_BASE = 1000

def parse_expert(name):
    """-> (L, E, wX, kind) or None. MTP block B maps to L = MTP_LAYER_BASE + B.

    WHY MTP IS NOW TRANSCODED (changed 2026-08-09): leaving MTP as MXFP4 while the main
    stack became NVFP4 made the speculative DRAFT and the TARGET disagree -- draft
    acceptance collapsed to 0.121 vs the 0.55-0.72 MXFP4 baseline, with the draft losing
    the very first token half the time. The scale hierarchies differ in DEPTH (MXFP4: one
    E8M0 exponent per 32 elements, no global; NVFP4: per-tensor global weight_scale_2 x
    E4M3 per 16 elements x the E2M1 element), so a mixed model asks one runtime to hold
    two different scale conventions. Transcoding MTP too makes the whole model one
    convention. The conversion is algebraically lossless: E4M3_block = 2^(e_old+G) and
    weight_scale_2 = 2^-G, so their product is exactly the original 2^(e_old), and the
    32-element block splits into two 16-element blocks carrying the same exponent.
    """
    m = EXPERT_RE.match(name)
    if m:
        return int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)  # L, E, wX, kind
    m = MTP_RE.match(name)
    if m:
        return MTP_LAYER_BASE + int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)
    return None

# ----------------------------------------------------------------------------- PASS 1: compute G dict
def compute_g(src_dir):
    """Stream all shards; for each expert scale tensor compute max non-zero E8M0 exponent.
    Return g13 = {(L,E): G}, g2 = {(L,E): G}."""
    import numpy as np
    idx_path = os.path.join(src_dir, 'model.safetensors.index.json')
    with open(idx_path) as f:
        idx = json.load(f)
    weight_map = idx['weight_map']
    # group expert scale tensors by shard
    shard_scales = {}  # shard_file -> list of (name, L, E, wX)
    for name, shard in weight_map.items():
        p = parse_expert(name)
        if p and p[3] == 'scale':
            shard_scales.setdefault(shard, []).append((name, p[0], p[1], p[2]))
    # max e_old per (L,E,wX)
    max_eold = {}  # (L,E,wX) -> max e_old (excluding 0x00)
    shards_sorted = sorted(shard_scales.keys())
    total = sum(len(v) for v in shard_scales.values())
    done = 0
    for shard in shards_sorted:
        path = os.path.join(src_dir, shard)
        header, hlen = read_header(path)
        base = 8 + hlen
        for name, L, E, wX in shard_scales[shard]:
            meta = header[name]
            s, e = meta['data_offsets']
            raw = read_tensor_bytes(path, base, s, e)
            arr = np.frombuffer(raw, dtype=np.uint8)
            nz = arr != 0
            if not nz.any():
                me = -128  # all zero
            else:
                me = int(arr[nz].astype(np.int32).max()) - 127
            max_eold[(L, E, wX)] = me
            done += 1
        print(f"  [pass1] {shard}: {len(shard_scales[shard])} scales ({done}/{total})", flush=True)
    # union G for w1/w3, independent for w2
    g13 = {}
    g2 = {}
    keys = set((L, E) for (L, E, _) in max_eold)
    n_zero_tensors = 0
    for (L, E) in sorted(keys):
        m1 = max_eold.get((L, E, 'w1'), -128)
        m3 = max_eold.get((L, E, 'w3'), -128)
        m2 = max_eold.get((L, E, 'w2'), -128)
        m13 = max(m1, m3)
        if m13 == -128:
            g13[(L, E)] = 8  # both zero -> neutral fallback
            n_zero_tensors += 1
        else:
            g13[(L, E)] = 8 - m13
        if m2 == -128:
            g2[(L, E)] = 8
            n_zero_tensors += 1
        else:
            g2[(L, E)] = 8 - m2
    if n_zero_tensors:
        print(f"  [pass1] WARNING: {n_zero_tensors} all-zero expert weight tensors (fallback G=8)", flush=True)
    # report G range
    g13_vals = list(g13.values())
    g2_vals = list(g2.values())
    print(f"  [pass1] G13 range [{min(g13_vals)},{max(g13_vals)}] (w1/w3 union) over {len(g13)} experts", flush=True)
    print(f"  [pass1] G2  range [{min(g2_vals)},{max(g2_vals)}] (w2)        over {len(g2)} experts", flush=True)
    return g13, g2

# ----------------------------------------------------------------------------- PASS 2: transcode shards
def transcode_shard(src_path, out_path, g13, g2):
    """Transform one input shard -> one output shard."""
    import numpy as np
    header, hlen = read_header(src_path)
    base = 8 + hlen
    out_tensors = []
    n_exp = 0
    n_verbatim = 0
    for name, meta in header.items():
        if name == '__metadata__':
            # carry metadata verbatim into output header
            # (handled specially below)
            continue
        dtype = meta['dtype']
        shape = meta['shape']
        s, e = meta['data_offsets']
        raw = read_tensor_bytes(src_path, base, s, e)
        p = parse_expert(name)
        if p:
            L, E, wX, kind = p
            if kind == 'weight':
                # byte-copy I8 -> U8, same bytes, dtype U8
                out_tensors.append((name, 'U8', shape, raw))
                n_exp += 1
            elif kind == 'scale':
                # E8M0 [R,C] -> E4M3 [R,2C]
                R, C = shape
                arr = np.frombuffer(raw, dtype=np.uint8).reshape(R, C)
                G = g13[(L, E)] if wX in ('w1', 'w3') else g2[(L, E)]
                e4m3 = e8m0_to_e4m3(arr, G)
                doubled = np.repeat(e4m3, 2, axis=1)  # [R, 2C]
                new_name = name.replace('.scale', '.weight_scale')
                out_tensors.append((new_name, 'F8_E4M3', [R, 2 * C], doubled.tobytes()))
                # add weight_scale_2 = 2^-G (F32 scalar) and input_scale = 1.0 (F32 scalar)
                ws2 = struct.pack('<f', float(2.0 ** (-G)))
                isc = struct.pack('<f', 1.0)
                base_name = name[:-len('.scale')]  # layers.L.ffn.experts.E.wX
                out_tensors.append((f'{base_name}.weight_scale_2', 'F32', [], ws2))
                out_tensors.append((f'{base_name}.input_scale', 'F32', [], isc))
                n_exp += 1
        else:
            # verbatim copy (attn, shared, norms, embed, MTP experts, hc_*, etc.)
            out_tensors.append((name, dtype, shape, raw))
            n_verbatim += 1
    # carry __metadata__ if present (place first like source)
    if '__metadata__' in header:
        out_tensors.insert(0, ('__metadata__', header['__metadata__'], None, b''))
    write_safetensors(out_path, out_tensors)
    return n_exp, n_verbatim

# handle __metadata__ specially: it has no dtype/shape/data_offsets in the same way.
# Actually safetensors __metadata__ is a dict, not a tensor. We must NOT emit it as a tensor.
# Rewrite write_safetensors to accept a metadata dict separately.

def write_safetensors2(path, tensors, metadata=None):
    """tensors: list of (name, dtype_str, shape_list, raw_bytes). metadata: dict or None."""
    header = {}
    off = 0
    blobs = []
    for name, dtype, shape, data in tensors:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(data)]}
        blobs.append(data)
        off += len(data)
    if metadata:
        header['__metadata__'] = metadata
    hdr_json = json.dumps(header, separators=(',', ':')).encode('utf-8')
    pad = (8 - ((8 + len(hdr_json)) % 8)) % 8
    hdr_padded = hdr_json + b' ' * pad
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(hdr_padded)))
        f.write(hdr_padded)
        for b in blobs:
            f.write(b)

def transcode_shard2(src_path, out_path, g13, g2):
    import numpy as np
    header, hlen = read_header(src_path)
    base = 8 + hlen
    out_tensors = []
    n_exp = 0
    n_verbatim = 0
    metadata = header.get('__metadata__', None)
    for name, meta in header.items():
        if name == '__metadata__':
            continue
        dtype = meta['dtype']
        shape = meta['shape']
        s, e = meta['data_offsets']
        raw = read_tensor_bytes(src_path, base, s, e)
        p = parse_expert(name)
        if p:
            L, E, wX, kind = p
            if kind == 'weight':
                out_tensors.append((name, 'U8', shape, raw))
                n_exp += 1
            else:  # scale
                R, C = shape
                arr = np.frombuffer(raw, dtype=np.uint8).reshape(R, C)
                G = g13[(L, E)] if wX in ('w1', 'w3') else g2[(L, E)]
                e4m3 = e8m0_to_e4m3(arr, G)
                doubled = np.repeat(e4m3, 2, axis=1)
                base_name = name[:-len('.scale')]
                out_tensors.append((f'{base_name}.weight_scale', 'F8_E4M3', [R, 2 * C], doubled.tobytes()))
                out_tensors.append((f'{base_name}.weight_scale_2', 'F32', [], struct.pack('<f', float(2.0 ** (-G)))))
                out_tensors.append((f'{base_name}.input_scale', 'F32', [], struct.pack('<f', 1.0)))
                n_exp += 1
        else:
            out_tensors.append((name, dtype, shape, raw))
            n_verbatim += 1
    write_safetensors2(out_path, out_tensors, metadata)
    return n_exp, n_verbatim

# ----------------------------------------------------------------------------- index.json
def write_index(out_dir, shard_files):
    """Rebuild model.safetensors.index.json from the written shards."""
    weight_map = {}
    total = 0
    for shard in sorted(shard_files):
        path = os.path.join(out_dir, shard)
        header, _ = read_header(path)
        total += os.path.getsize(path)
        for name, meta in header.items():
            if name == '__metadata__':
                continue
            weight_map[name] = shard
    idx = {
        "metadata": {"total_size": total},
        "weight_map": weight_map,
    }
    with open(os.path.join(out_dir, 'model.safetensors.index.json'), 'w') as f:
        json.dump(idx, f, indent=2)
    return len(weight_map), total

# ----------------------------------------------------------------------------- config.json merge
def write_config(src_dir, ref_dir, out_dir):
    with open(os.path.join(src_dir, 'config.json')) as f:
        src = json.load(f)
    with open(os.path.join(ref_dir, 'config.json')) as f:
        ref = json.load(f)
    # start from 0731 (keeps arch + DSpark fields), replace ONLY quantization_config
    out = dict(src)
    out['quantization_config'] = ref['quantization_config']
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(out, f, indent=2)
    return out

# ----------------------------------------------------------------------------- verify (bit-exactness)
def verify(src_dir, out_dir, n_samples=64):
    import numpy as np
    with open(os.path.join(out_dir, 'model.safetensors.index.json')) as f:
        out_idx = json.load(f)['weight_map']
    with open(os.path.join(src_dir, 'model.safetensors.index.json')) as f:
        src_idx = json.load(f)['weight_map']
    # gather expert weight+scale names
    exp_keys = []  # (L,E,wX)
    for name in src_idx:
        p = parse_expert(name)
        if p and p[3] == 'scale':
            exp_keys.append((p[0], p[1], p[2]))
    if not exp_keys:
        print("verify: no expert scales found", flush=True)
        return False
    rng = np.random.default_rng(12345)
    picks = [exp_keys[i] for i in rng.integers(0, len(exp_keys), size=min(n_samples, len(exp_keys)))]
    # cache shard header/base for source and output
    src_cache = {}
    out_cache = {}
    def get(cache, idx_map, dir_, name):
        shard = idx_map[name]
        if shard not in cache:
            path = os.path.join(dir_, shard)
            h, hl = read_header(path)
            cache[shard] = (path, h, 8 + hl)
        return cache[shard]
    ok = 0
    bad = 0
    for (L, E, wX) in picks:
        wname = f'layers.{L}.ffn.experts.{E}.{wX}.weight'
        sname = f'layers.{L}.ffn.experts.{E}.{wX}.scale'
        owname = wname  # weight name unchanged
        osname = f'layers.{L}.ffn.experts.{E}.{wX}.weight_scale'
        ows2 = f'layers.{L}.ffn.experts.{E}.{wX}.weight_scale_2'
        # source weight + scale
        sp, sh, sb = get(src_cache, src_idx, src_dir, wname)
        sm = sh[wname]; sw = read_tensor_bytes(sp, sb, *sm['data_offsets'])
        sp2, sh2, sb2 = get(src_cache, src_idx, src_dir, sname)
        sm2 = sh2[sname]; ss = read_tensor_bytes(sp2, sb2, *sm2['data_offsets'])
        # output weight + weight_scale + weight_scale_2
        op, oh, ob = get(out_cache, out_idx, out_dir, owname)
        om = oh[owname]; ow = read_tensor_bytes(op, ob, *om['data_offsets'])
        op2, oh2, ob2 = get(out_cache, out_idx, out_dir, osname)
        om2 = oh2[osname]; os_ = read_tensor_bytes(op2, ob2, *om2['data_offsets'])
        op3, oh3, ob3 = get(out_cache, out_idx, out_dir, ows2)
        om3 = oh3[ows2]; ows2b = read_tensor_bytes(op3, ob3, *om3['data_offsets'])
        # 1) weight bytes EXACT match (byte copy)
        if sw != ow:
            print(f"  [verify] WEIGHT MISMATCH L{L} E{E} {wX} ({len(sw)} vs {len(ow)} bytes)", flush=True)
            bad += 1; continue
        # 2) scale relationship: source effective = 2^(b-127); target effective = 2^-G * 2^((b4>>3)-7)
        #    => (b4>>3)-7 == (b-127)+G ; need G. Recover G from weight_scale_2 = 2^-G.
        Gf = struct.unpack('<f', ows2b)[0]
        G = round(-math.log2(Gf)) if Gf > 0 else None
        src_arr = np.frombuffer(ss, dtype=np.uint8)
        out_arr = np.frombuffer(os_, dtype=np.uint8)
        # out_arr is [R, 2C] flattened row-major; src is [R, C]. Compare block-by-block (sample a few rows).
        R, C = sh2[sname]['shape']
        src2d = src_arr.reshape(R, C)
        out2d = out_arr.reshape(R, 2 * C)
        # check a sample of rows
        rows = rng.integers(0, R, size=min(8, R))
        good = True
        for r in rows:
            for c in range(min(4, C)):
                b = int(src2d[r, c])
                b4_a = int(out2d[r, 2 * c])
                b4_b = int(out2d[r, 2 * c + 1])
                # both halves identical
                if b4_a != b4_b:
                    print(f"  [verify] HALF MISMATCH L{L} E{E} {wX} r{r} c{c}: {b4_a}!={b4_b}", flush=True)
                    good = False; break
                if b == 0:
                    if b4_a != 0x01:
                        print(f"  [verify] ZERO MISMATCH L{L} E{E} {wX} r{r} c{c}: src0 -> {b4_a} (want 0x01)", flush=True)
                        good = False; break
                else:
                    e_new_expected = (b - 127) + G
                    e_new_actual = (b4_a >> 3) - 7
                    if e_new_expected != e_new_actual:
                        print(f"  [verify] SCALE MISMATCH L{L} E{E} {wX} r{r} c{c}: exp e_new {e_new_expected} got {e_new_actual} (src b={b}, G={G})", flush=True)
                        good = False; break
            if not good:
                break
        if good:
            ok += 1
        else:
            bad += 1
    print(f"  [verify] {ok} OK / {bad} BAD out of {ok+bad} sampled expert weights", flush=True)
    return bad == 0

# ----------------------------------------------------------------------------- main
def main():
    cmd = sys.argv[1]
    if cmd == 'transcode':
        src_dir = sys.argv[2]; out_dir = sys.argv[3]
        shards_arg = None
        resume = False
        for a in sys.argv[4:]:
            if a.startswith('--shards='):
                lo, hi = a.split('=', 1)[1].split(':')
                shards_arg = (int(lo), int(hi))
            elif a == '--resume':
                resume = True
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(src_dir, 'model.safetensors.index.json')) as f:
            idx = json.load(f)
        all_shards = sorted(set(idx['weight_map'].values()))
        if shards_arg:
            sel = all_shards[shards_arg[0]:shards_arg[1]]
        else:
            sel = all_shards
        print(f"=== TRANSCODE {len(sel)} shards: {src_dir} -> {out_dir} ===", flush=True)
        print(f"=== pass 1: compute G per expert (streaming {len(all_shards)} shards) ===", flush=True)
        g13, g2 = compute_g(src_dir)
        # save G dict for resume/debug
        with open(os.path.join(out_dir, '_g_dict.json'), 'w') as f:
            json.dump({'g13': {f'{k[0]},{k[1]}': v for k, v in g13.items()},
                       'g2':  {f'{k[0]},{k[1]}': v for k, v in g2.items()}}, f)
        print(f"=== pass 2: transcode {len(sel)} shards ===", flush=True)
        done_shards = []
        for i, shard in enumerate(sel):
            out_path = os.path.join(out_dir, shard)
            if resume and os.path.exists(out_path):
                print(f"  [pass2] ({i+1}/{len(sel)}) {shard}: EXISTS (resume skip)", flush=True)
                done_shards.append(shard)
                continue
            n_exp, n_verb = transcode_shard2(os.path.join(src_dir, shard), out_path, g13, g2)
            done_shards.append(shard)
            print(f"  [pass2] ({i+1}/{len(sel)}) {shard}: {n_exp} expert tensors, {n_verb} verbatim, {os.path.getsize(out_path)//(1024*1024)} MiB", flush=True)
        print(f"=== writing index ===", flush=True)
        nt, total = write_index(out_dir, done_shards)
        print(f"=== DONE: {nt} tensors, {total//(1024*1024*1024)} GiB total, {len(done_shards)} shards ===", flush=True)
    elif cmd == 'config':
        src_dir, ref_dir, out_dir = sys.argv[2], sys.argv[3], sys.argv[4]
        write_config(src_dir, ref_dir, out_dir)
        print(f"=== config.json written to {out_dir} ===", flush=True)
    elif cmd == 'verify':
        src_dir, out_dir = sys.argv[2], sys.argv[3]
        ok = verify(src_dir, out_dir)
        sys.exit(0 if ok else 1)
    elif cmd == 'index':
        out_dir = sys.argv[2]
        with open(os.path.join(out_dir, 'model.safetensors.index.json')) as f:
            idx = json.load(f)
        shards = sorted(set(idx['weight_map'].values()))
        nt, total = write_index(out_dir, shards)
        print(f"=== index rewritten: {nt} tensors, {total//(1024*1024*1024)} GiB ===", flush=True)
    else:
        print(__doc__)
        sys.exit(1)

if __name__ == '__main__':
    main()