# DeepSeek-V4-Flash-0731 **NVFP4** on 2× DGX Spark (GB10)

A working transcode of DeepSeek-V4-Flash-0731 from **MXFP4** to **NVFP4** experts, served
TP=2 across two DGX Spark (GB10, `sm_121a`) nodes at 1M context, with a like-for-like
benchmark against the MXFP4 original on the same hardware and the same vLLM image.

**Model (164 GiB, 48 shards):** https://huggingface.co/tomsarihan/DeepSeek-V4-Flash-0731-NVFP4
**Model card:** [docs/MODEL-CARD.md](docs/MODEL-CARD.md)
**Companion repo (the MXFP4 baseline this is measured against):**
https://github.com/tsarihan/deepseek-v4-flash-0731-2x-dgx-spark-playbook

© 2026 Tom Sarihan, Desnet AI LLC. Apache-2.0 (see `LICENSE`, `NOTICE`).

---

## The short version

NVFP4 is **faster than MXFP4 at every context length** and **numerically indistinguishable
from it**. The cost is a small amount of KV capacity.

| axis | NVFP4 vs MXFP4 | how it was measured |
|---|---|---|
| Prefill / TTFT | **1.14–1.32× faster** | NIAH rungs 4K→265K, token counts verified via `/tokenize` |
| Single-stream decode | **1.27–1.83× faster** | same NIAH runs |
| Retrieval accuracy | **identical** — 5/5 needles, all depths, to 265,781 tok | 5 needles at depths 0.1–0.9 |
| Logit distribution | **below the noise floor** (see below) | teacher-forced, 4,559 scored positions |
| Draft acceptance | **identical** — 0.273 vs 0.274 | vLLM spec-decode counters |
| KV capacity | MXFP4 **+6%** | at matched `gpu-memory-utilization` |

The strongest single result: **NVFP4 differs from MXFP4 by less than one build differs from
itself between runs.** Cross-build p95 |Δlogprob| is 0.1217; the same-build run-to-run floor
is 0.1442. Top-1 agreement is 99.28% cross-build vs 99.14% same-build.

Peak single-stream decode observed: **40.28 tok/s** (NVFP4, spec-on K=7, `max-num-seqs 48`).

---

## Reproducibility — everything is pinned

| component | pin |
|---|---|
| vLLM image | `vllm-dsv4:src-sm121` — vLLM `0.25.2.dev0+g752a3a504` |
| torch | `2.11.0+cu130` |
| flashinfer | `0.6.15` |
| GPU arch | `sm_121a` (GB10), `TORCH_CUDA_ARCH_LIST=12.1a` |
| MoE backend | `FLASHINFER_CUTLASS` (NVFP4), `B12X_MXFP4` (MXFP4 baseline) |
| KV cache | `fp8_ds_mla` (both sides; see note) |
| Interconnect | CX7 200GbE RoCE, `rocep1s0f0`, measured 109 Gb/s |

**Note on `nvfp4_ds_mla`:** it exists only in `src-sm121`; the upstream `dev403` image
rejects it (`invalid choice`). For DeepSeek-V4 both dtypes use the **same 584-byte page**
(`kv_cache_interface.py`: 448B NoPE + 128B RoPE + 8B scale), so the choice costs zero KV
bytes and measured within ~1.4% either way.

---

## What NVFP4 actually changes

The transcode converts **only the routed expert weights**. Attention, shared experts, the
sparse-MLA indexer, norms and embeddings stay exactly as they were.

Inspecting the compiled MoE kernel's SASS shows how small that slice is:

```
HMMA .F32.BF16                          36528   35.1%
HMMA .16816.F32                         36528   35.1%
QMMA .E4M3.E4M3            (FP8)        24480   23.5%
OMMA .SF.16864.F32.E2M1.E2M1.UE4M3.4X    4736    4.6%   <-- native NVFP4
QMMA .SF...E4M3.E2M1.E8                   896    0.9%
QMMA .SF...E2M1.E4M3.E8                   896    0.9%
                              native FP4: 6.3%   BF16: 70.2%
```

**Native FP4 math is confirmed running** — `OMMA.SF...E2M1.E2M1.UE4M3` is the block-scaled
FP4×FP4 tensor-core instruction, every cubin targets `sm_121a`, and the emulation path's
"dequantize weights on the fly" warning never fires. But it is only **6.3% of MMA work**.

That single number explains the whole result: a 1.2–1.3× speedup from quantizing 6% of the
math is a large *relative* gain on that slice, and it is also why **the advantage narrows as
context grows** (1.31× at 4K → 1.14× at 265K) — the BF16 attention and indexer path scales
with context while the expert GEMMs do not.

---

## The scale-hierarchy mapping (why the transcode is lossless)

MXFP4 and NVFP4 differ in how many levels of scale sit between the stored 4-bit element and
the real value:

```
MXFP4  (2 levels):  value = E2M1_elem × 2^(e_old)              block = 32 elems, E8M0 scale
NVFP4  (3 levels):  value = E2M1_elem × E4M3_block × weight_scale_2
                            (individual)   (block=16)    (per-tensor GLOBAL)
```

The transcoder emits `E4M3_block = 2^(e_old + G)` and `weight_scale_2 = 2^-G`, so their
product is **exactly** the original `2^(e_old)`. One 32-element block splits into two
16-element blocks carrying the same exponent byte, so every element keeps its effective
scale. `G = 8 − max(e_old)` puts the largest block at E4M3's maximum normal exponent.

The 4-bit weight bytes are a **byte-for-byte copy** — nibble order is unchanged. Only the
scale metadata is rewritten, which is why the checkpoint grows (163.5 GiB vs 156 GiB): two
extra tensors per expert weight (`weight_scale_2`, `input_scale`) and twice as many block
scales.

Verified before and after: `check-mtp-range.py` confirms every expert group fits the E4M3
window (widest span 6 of a 14-wide window, zero `0xFF` bytes), and
`transcode_0731_to_nvfp4.py verify` reports **64 OK / 0 BAD** on sampled expert weights.

---

## Findings

### 1. The MTP draft must be transcoded too — this was the single biggest win

Leaving the speculative-decode draft (MTP) as MXFP4 while the main stack became NVFP4 puts
**two scale conventions in one model**. The draft and target then disagree constantly:

| | acceptance | accepted/draft | pos-0 survival |
|---|---|---|---|
| MTP left as MXFP4 | 0.121 | 0.85 | 50.2% |
| **MTP transcoded to NVFP4** | **0.31–0.38** | **2.2–2.3** | **76.4%** |

It also deadlocked the engine. After transcoding MTP, spec decode is worth **~2×** over
spec-off, in line with what the MXFP4 baseline gets.

Measured against a like-for-like MXFP4 run on the same cluster, NVFP4 acceptance is
**0.273 vs 0.274 — identical.** (An earlier "0.55–0.72 baseline" came from a different
build and config and was never a valid target.)

### 2. `ignore` in `quantization_config` does nothing — vLLM reads `ignored_layers`

The first transcode tried to exempt MTP with
`"ignore": ["*.attn.*", "*.ffn.shared_experts.*", "head", "mtp.*"]`. That exemption was
**completely dead**, for two independent reasons:

- `Fp8Config.from_config` reads `ignored_layers` or `modules_to_not_convert`. **`ignore` is
  never read.**
- `is_layer_skipped` defaults to `prefix_full_match` — **exact string equality, not glob**.
  `"mtp.*"` only matches a layer literally named `mtp.*`.

Verified directly: `is_layer_skipped('mtp.0.ffn.experts', [...,'mtp.*'])` → `False`.

If you need an exemption, use exact prefixes from the tensor index:
`"ignored_layers": ["mtp.0.ffn.experts", "mtp.1.ffn.experts", "mtp.2.ffn.experts"]`.

### 3. FULL CUDA graphs are unusable on GB10 — the draft step is permanently un-graphed

The DSpark speculator's own docstring says *"CUDA graphs (FULL, mirroring DFlash) cover the
whole draft step: the parallel backbone forward AND the sequential Markov sampling"*, and it
does emit `Capturing dspark CUDA graphs (FULL)` at startup. Under `PIECEWISE` that capture
never happens and the draft runs eager on every drafted token.

But every FULL-family mode wedges on GB10:

```
FULL_AND_PIECEWISE  -> graphs captured, first request 443s, RuntimeError: cancelled
FULL_DECODE_ONLY    -> graphs captured, first request 673s, RuntimeError: cancelled
```

Both with zero JIT activity and >100 GB free memory. This reproduces vLLM #40969.
**`PIECEWISE` is the only usable mode**, so there is a permanent, unavoidable spec-decode tax
on this hardware. Do not "fix" it back to FULL.

### 4. `max-num-seqs` costs ~2.5× on decode — even for a single stream

| `max-num-seqs` | `gpu-mem-util` | KV tokens | @1M | single-stream decode |
|---|---|---|---|---|
| 4 | 0.835 | 1,654,544 | 1.58× | ~16 tok/s |
| 32 | 0.86 | 2,075,773 | 1.98× | 18.2 median, 23.4 best |
| 48 | 0.87 | 2,174,851 | 2.07× | **40.3 best** (but see stability note) |

A larger sequence budget changes how the scheduler batches drafted tokens and lifts even
single-stream throughput. If you benchmark at `max-num-seqs 4` because your workload is
1–4 streams, **you will understate your own hardware by ~2.5×**.

### 5. The images that matter, and one that does not

`ghcr.io/anemll/dspark-vllm-gx10:0.1.1` and our `vllm-dsv4:src-sm121` are the **same vLLM** —
identical base commit, identical torch and flashinfer, and **byte-identical md5** on every
hot-path file including `v1/worker/gpu/spec_decode/dspark/speculator.py`,
`kv_cache_interface.py`, `mla/sparse_swa.py`, `mla/flashmla_sparse.py`, `b12x_mxfp4_moe.py`
and both MoE oracles. Benchmarked head to head, anemll gives no speedup.

What *does* differ is the upstream `vllm/vllm-openai:pinned-dev403` image, which **cannot run
MXFP4 0731 on GB10 at all** — four distinct failures, each ruling out one workaround:

1. DeepGEMM on → `layout.hpp:60 Unknown SF transformation` (no `arch_major=12` branch)
2. `VLLM_USE_DEEP_GEMM=0` → `hyperconnection.hpp:56 Unsupported architecture` (MHC needs
   DeepGEMM; it also silently drops the MoE backend to MARLIN)
3. `--moe-backend flashinfer_b12x` → rejected, that is the NVFP4 backend list
4. `--moe-backend flashinfer_cutlass` → `Unsupported mxfp4_backend ...
   FLASHINFER_CUTLASS_MXFP4_MXFP8`

Also note `--linear-backend cutlass` does **not** stop DeepGEMM touching the MoE scale
packing — that needs `VLLM_USE_DEEP_GEMM=0`, a separate switch.

### 6. Long context holds perfectly on both quantizations

5 needles at depths 0.1/0.3/0.5/0.7/0.9, nonce-salted so prefix caching cannot serve a
later run from an earlier prefill:

| context | needles (MXFP4 / NVFP4) | prefill × | TTFT × | decode × |
|---|---|---|---|---|
| 4,318 | 5/5 · 5/5 | 1.31× | 1.31× | 1.83× |
| 33,375 | 5/5 · 5/5 | 1.32× | 1.32× | — |
| 132,973 | 5/5 · 5/5 | 1.23× | 1.23× | 1.27× |
| 265,781 | 5/5 · 5/5 | 1.14× | 1.14× | 1.57× |

NVFP4 saves **55 seconds of TTFT** at 265K and 37s at 133K. Zero swap on either node at any
rung; the tightest moment was ~4–5 GB available during the 262K rungs.

### 7. Memory tuning is asymmetric between the two builds — control for it

NVFP4 weights are ~4 GiB larger (extra scale metadata), so at equal `gpu-memory-utilization`
it has less room for KV. But utilization absorbs almost all of it: at 0.87 NVFP4 reaches
2,174,851 tokens against MXFP4's 2,307,250 — a **6% gap**. An earlier draft of this work
reported 45%, which was an artifact of running NVFP4 at 0.835 (raised only enough to clear
the spec-on 1M boot floor) against MXFP4 at its own default. **Always report the utilization
each side ran at.**

---

## Gotchas that invalidate benchmarks

These cost real time here, and every one produces a plausible-looking wrong number.

**Never estimate token counts.** A filler of `token0 token1 token2 …` tokenizes to ~2.76
tokens per word for this tokenizer. A prompt labelled "4096 tokens" was really 11,288, and
every prefill rate derived from it was understated 2.76×. Ask the server:
`POST /tokenize` — note it lives at the **server root**, not under `/v1` (that 404s).

**Count tokens, not stream chunks.** With speculative decoding one SSE chunk can carry
several accepted draft tokens. Counting chunks undercounted 1.6× (160 counted vs 256
reported) and turned 14.1 tok/s into 5.0. Request `stream_options: {include_usage: true}`
and use the server's `completion_tokens`.

**JIT skew wedges TP=2, and it looks exactly like memory exhaustion.** The two ranks compile
Triton kernels independently, at whatever moment each first sees a shape. While one compiles,
the other blocks in a collective:

```
[rank0] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=3466,
        OpType=_ALLGATHER_BASE, NumelIn=517120, NumelOut=1034240,
        Timeout(ms)=600000) ran for 600071 ms before timing out
```

At that moment there was **113 GB free and zero swap**. Every wedge observed in this work had
ample memory. The fix is to warm across the shapes you intend to measure — including prompt
length, output length and concurrency — and then assert both ranks are quiet:

```bash
docker logs --since 3m <container> | grep -c 'JIT compilation during inference'   # want 0
```

When warmup is done right, both ranks compile at the *same timestamp* and nothing hangs.

**`shm_broadcast` "No available shared memory broadcast block" means a peer is missing.**
Not local compilation, whatever the message suggests. Check for both `Worker_TP0` and
`Worker_TP1`.

**Warm before measuring, at the shape you measure.** A cold first rung measures kernel
compilation: 4.66 tok/s with a 14.95s TTFT cold, versus 5.14 at 0.45s warm — on the same
config, minutes apart.

**Spec-decode throughput is noisy.** ~45% run-to-run spread at acceptance ~0.32. Three runs
is not a median; use ten or more.

---

## Setup

See [QUICKSTART.md](QUICKSTART.md) for the shortest path from the MXFP4 checkpoint to a
serving NVFP4 endpoint.

The full narrative — what broke, what the wrong turns were, and how each was diagnosed — is
in [docs/JOURNEY.md](docs/JOURNEY.md).

## Reproducing the measurements

```bash
scripts/ttft-curve.sh    <tag> <base-url> <model>   # prefill/TTFT vs context
scripts/niah-run.sh      <tag> <model> [contexts]   # needle retrieval, per-rung warmup
scripts/decode-rate.sh   <model> [n]                # warm decode, token-accurate
scripts/capture-logits.sh <tag> <base-url> <model>  # teacher-forced logprobs
scripts/compare-logits.py A.json B.json --floor F   # numerical A/B
scripts/k-sweep.sh       "3 5 7"                    # MTP depth
scripts/seqs-sweep.sh    "32 48"                    # sequence budget
```

## Credits

Built on DeepSeek-V4-Flash-0731 and the vLLM DeepSeek-V4 stack. The MXFP4 baseline, the
DSpark port and the `sm_121a` kernels come from the work referenced in the
[companion playbook](https://github.com/tsarihan/deepseek-v4-flash-0731-2x-dgx-spark-playbook)
and from [Anemll](https://github.com/Anemll/dspark-vllm-gx10).
