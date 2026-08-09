# RESULTS

All measurements on 2× DGX Spark (GB10, `sm_121a`), TP=2, vLLM `0.25.2.dev0+g752a3a504`
(`vllm-dsv4:src-sm121`) for **both** quantizations, torch `2.11.0+cu130`, flashinfer
`0.6.15`, `CUDAGRAPH_MODE=PIECEWISE`, `max-model-len 1048576`.

Unless stated otherwise: spec-on, `MTP_K=7`, `draft_sample_method=greedy`, servers warmed at
the measured shape, token counts taken from the server (`/tokenize` or `usage`).

---

## 1. Needle-in-a-haystack + prefill/TTFT/decode

5 needles per rung at depths 0.10 / 0.30 / 0.50 / 0.70 / 0.90, nonce-salted so prefix caching
cannot serve a later run from an earlier prefill. `MAX_NUM_SEQS=4`, `GPU_UTIL=0.835`.

| ctx (actual) | MXFP4 needles | NVFP4 needles | MX prefill | NV prefill | prefill × | MX TTFT | NV TTFT | MX decode | NV decode |
|---|---|---|---|---|---|---|---|---|---|
| 4,318 | 5/5 | 5/5 | 767.3 | 1005.8 | **1.31×** | 5.63s | 4.29s | 16.71 | 30.64 |
| 33,375 | 5/5 | 5/5 | 521.6 | 688.1 | **1.32×** | 63.99s | 48.51s | — | 20.07 |
| 132,973 | 5/5 | 5/5 | 681.3 | 840.3 | **1.23×** | 195.18s | 158.25s | 18.81 | 23.94 |
| 265,781 | 5/5 | 5/5 | 594.5 | 678.7 | **1.14×** | 447.04s | 391.59s | 18.74 | 29.49 |

Rates in tok/s. NVFP4 saves **55 s** of TTFT at 265K and 37 s at 133K.

**Retrieval is identical** — every needle at every depth on both builds, out to 265,781
tokens. Zero swap on either node at any rung; tightest was ~4–5 GB available during 262K.

The prefill advantage narrows with context (1.31× → 1.14×) because only 6.3% of MMA work is
FP4; the BF16 attention/indexer path scales with context while the expert GEMMs do not.

---

## 2. Numerical A/B — teacher-forced logprobs

`echo=true, max_tokens=0, logprobs=20, temperature=0` over an identical ~5.1k-token
prose+code passage (`scripts/probe-passage.txt`), 4,559 scored positions.

Teacher-forced matters: with free-running generation, once two builds diverge at one token
every later position is conditioned on different text, so the deltas measure divergence
rather than quantization error.

| comparison | top-1 agreement | median \|Δ\| | p95 \|Δ\| | mean Δ (z) | mean NLL |
|---|---|---|---|---|---|
| NVFP4 vs **MXFP4** | 99.28% | 0.0000 | **0.1217** | −0.0030 (z=−1.8) | 0.2088 / 0.2058 |
| NVFP4 vs **itself** (floor) | 99.14% | 0.0000 | **0.1442** | −0.0024 (z=−1.4) | — |

**The two builds differ from each other by less than one build differs from itself between
runs.** Cross-build p95 is *smaller* than the same-build run-to-run floor, and cross-build
top-1 agreement is *higher*. ΔNLL is +0.0030 nats/token against a 0.1 concern threshold.

The mean-signed-delta z-test is the gate that detects uniform scale errors and is unaffected
by the W4A16→W4A4 regime change; it is clear at z = −1.8. Magnitude gates must be calibrated
against the measured floor, not a constant — a threshold written for weight-only noise will
false-fail here.

---

## 3. Draft acceptance

Same conditions, spec-on K=7, from vLLM's `spec_decode` counters.

| build | acceptance | accepted/draft |
|---|---|---|
| MXFP4 | 0.274 | 1.92 |
| **NVFP4** | **0.273** | **1.91** |

Identical. Acceptance *falls* with draft depth (0.376 at K=5 → 0.273 at K=7) while
throughput *rises* — what drives speed is accepted tokens per step, not hit rate.

### Before the MTP fix

With MTP left as MXFP4 while the main stack was NVFP4:

| | acceptance | accepted/draft | pos-0 | pos-1 | pos-2 | pos-3 |
|---|---|---|---|---|---|---|
| mixed conventions | 0.121 | 0.85 | 50.2% | 23.2% | 8.2% | 2.1% |
| **all NVFP4** | **0.31–0.38** | **2.2–2.3** | **76.4%** | **56.7%** | **36.6%** | **22.6%** |

---

## 4. MTP draft depth (K)

NVFP4, spec-on, `MAX_NUM_SEQS=4`, `GPU_UTIL=0.835`, warmed.

| K | TTFT@1 | conc-1 | conc-2 | conc-4 | agg@4 | acceptance |
|---|---|---|---|---|---|---|
| 3 | 0.419 | 14.19 | *died* | *died* | — | — |
| 5 | 0.555 | 14.81 | 11.27 | **8.53** | **32.59** | 0.376 |
| **7** | **0.410** | **16.41** | 11.44 | 8.01 | 31.53 | 0.273 |

**K=7 is fastest single-stream and has the lowest TTFT; K=5 wins at 4 concurrent streams**
(the extra K+1 draft slots compete for the batched-token budget under contention).

The K=3 failures were a benchmarking artifact, not a limit: that run measured with 1 JIT
compile still pending per rank, while K=5 and K=7 measured at 0/0 and completed every rung.

---

## 5. Sequence budget — the largest single config effect

NVFP4, spec-on K=7, warmed, 512-token generations.

| `max-num-seqs` | `gpu-mem-util` | KV tokens | @1M | single-stream decode |
|---|---|---|---|---|
| 4 | 0.835 | 1,654,544 | 1.58× | ~16 tok/s |
| 32 | 0.86 | 2,075,773 | 1.98× | 18.18 median, 23.41 best |
| 48 | 0.87 | 2,174,851 | 2.07× | **40.28 best** |

A larger sequence budget lifts even single-stream throughput. Benchmarking at
`max-num-seqs 4` because the workload is 1–4 streams **understates the hardware by ~2.5×**.

⚠ `max-num-seqs 48` was observed to wedge one run in three; 32 completed every run. Treat 32
as the safe recommendation and validate 48 over 10+ generations before relying on it.

⚠ Decode has ~45% run-to-run spread at acceptance ~0.32 (16.17–23.41 tok/s over three runs).
Three runs is not a median.

---

## 6. Memory and KV capacity

| config | model weights | KV avail | KV tokens | @1M |
|---|---|---|---|---|
| MXFP4, util 0.80 | 74.07 GiB | 15.37 GiB | 2,307,250 | 2.20× |
| NVFP4, util 0.835 | 78.11 GiB | 10.63 GiB | 1,654,544 | 1.58× |
| NVFP4, util 0.87 | 78.11 GiB | — | 2,174,851 | 2.07× |

NVFP4 weights are ~4 GiB larger — the 4-bit weight bytes are byte-identical, but the
transcode adds `weight_scale_2` + `input_scale` per expert weight and doubles the block-scale
count (E8M0 `[R,K/32]` → E4M3 `[R,K/16]`). On disk: 163.5 GiB vs 156 GiB.

At **matched utilization the KV gap is 6%**, not the 45% implied by comparing NVFP4 at 0.835
against MXFP4 at its default. Always report the utilization each side ran at.

KV cost is ~7,230 bytes/token. For DeepSeek-V4 both `fp8_ds_mla` and `nvfp4_ds_mla` use the
same 584-byte page (448B NoPE + 128B RoPE + 8B scale), so the KV dtype choice costs nothing
here and measured within ~1.4% either way.

---

## 7. Kernel-level evidence

SASS of the compiled `fused_moe_120.so` (all cubins `sm_121a`):

| instruction | count | share |
|---|---|---|
| `HMMA .F32.BF16` | 36,528 | 35.1% |
| `HMMA .16816.F32` | 36,528 | 35.1% |
| `QMMA .E4M3.E4M3` (FP8) | 24,480 | 23.5% |
| **`OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`** | **4,736** | **4.6%** |
| `QMMA.SF...E4M3.E2M1.E8` | 896 | 0.9% |
| `QMMA.SF...E2M1.E4M3.E8` | 896 | 0.9% |

**native FP4: 6.3% · BF16: 70.2% · FP8: 23.5%**

`OMMA.SF...E2M1.E2M1.UE4M3` is the block-scaled FP4×FP4 tensor-core instruction — native
NVFP4 math is running, not emulated (the emulation path's "dequantize weights on the fly"
warning never fires). But it covers only the routed expert GEMMs.

---

## 8. Transcode verification

| check | result |
|---|---|
| E4M3 window pre-flight | **PASS** — 1,536 MTP groups, widest exponent span 6 of 14, zero `0xFF` |
| Bit-exactness (`verify`) | **64 OK / 0 BAD** |
| Tensor count | 142,973 (138,365 original + 4,608 new MTP scale tensors) |
| Total size | 164.0 GiB |
| G ranges | G13 [10,13] (w1/w3 union), G2 [9,13] (w2 independent) |

---

## Environment notes

- Interconnect verified at **109 Gb/s** (`ib_write_bw -d rocep1s0f0 -F --report_gbits`).
  A mismatched HCA makes NCCL fall back silently to the management network.
- `vm.swappiness = 0` on both nodes. Zero swap events across every benchmark in this document,
  including both full NIAH ladders.
- Every engine wedge observed had ample free memory (113 GB at one, 6.6 GB at another) and
  zero swap — see `docs/JOURNEY.md` §5 for why these are JIT-skew deadlocks, not exhaustion.
