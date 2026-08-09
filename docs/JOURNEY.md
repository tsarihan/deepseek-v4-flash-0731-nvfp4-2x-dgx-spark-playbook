# JOURNEY — what we tried, what broke, what fixed it

An honest record of getting DeepSeek-V4-Flash-0731 transcoded to NVFP4 and benchmarked
against its MXFP4 original on 2× DGX Spark. It includes the wrong turns, because several of
them produced confident, plausible, **wrong** numbers that survived for hours — and those are
the parts most likely to save someone else time.

---

## Part 1 — Getting NVFP4 to load at all

### The stale kernel with the wrong ABI

The first several launches died on the **first MoE forward**, roughly 80 seconds after a
clean weight load:

```
TypeError: Mismatched number of arguments when calling: init(...) -> ffi.Module.
Expected 7 but got 8 arguments        flashinfer/fused_moe/core.py:403
```

The image ships a **prebuilt** `fused_moe_120.so`. flashinfer's `JitSpec.is_aot` is simply
`aot_path.exists()`, so when that file is present `build_and_load` loads it and never
compiles from source. That prebuilt `.so` was built against an older 7-argument `init` FFI;
flashinfer 0.6.15 passes 8 (it added `use_fused_finalize`).

Deleting it forces a JIT rebuild from the bundled 8-arg source. Two things made this take
longer than it should have:

- **`FLASHINFER_JIT_VERBOSE=1` silently forces a debug build.** In `jit/core.py`,
  `debug = (FLASHINFER_JIT_DEBUG if set else FLASHINFER_JIT_VERBOSE) == "1"`. So asking for
  progress logs got `-g -O0 --device-debug`, ~40 GB per ptxas process and a ~4-hour build
  that OOM'd the box. Setting `FLASHINFER_JIT_DEBUG=0` explicitly gives the release build
  (~35 min).
- **The cleanup was too aggressive.** The wrapper deleted the image's AOT copy *and* the
  node-local `/cache` build, throwing away a valid kernel and forcing a fresh rebuild on
  every launch. `nm -D` settles which is which in one command:

  ```
  image AOT      _Z4init10DLDataTypeS_S_bbbb    7 args   stale, delete
  node /cache    _Z4init10DLDataTypeS_S_bbbbb   8 args   good, keep
  ```

  Both nodes had built their own valid `.so` independently (slightly different sizes — they
  were compiled, not copied). The wrapper now greps for the 8-arg symbol and only purges when
  it is absent.

**Lesson:** `Using 'FLASHINFER_CUTLASS' MoE backend` in the log is *selection* time. It is
not evidence the kernel runs. The first forward is.

### A four-hour hang that was one missing container

A launch sat for four hours logging, once a minute:

```
No available shared memory broadcast block found in 60 seconds. This typically happens when
some processes are hanging or doing some time-consuming work (e.g. compilation, ...)
```

The message's own explanation is a red herring. It means **a peer rank is missing**. Only
rank 0 was running; there was no rank 1 container on the second node at all. `ps aux | grep
Worker_TP` shows `Worker_TP0` and `Worker_TP1` on a healthy TP=2 launch — one per node, each
in its own container, each loading its own copy of the weights.

Compounding it: a `docker ps` was misread, rank 0 was killed while rank 1 was in fact alive,
and that orphaned rank 1 with a TCPStore broken pipe. **Check each node separately** — the
container name is identical on both, so the output is ambiguous unless you track which host
you ran it on.

---

## Part 2 — The transcode

### The scale hierarchies do not have the same depth

This is the heart of the work.

```
MXFP4  value = E2M1_elem × 2^(e_old)                     block 32, E8M0, no global
NVFP4  value = E2M1_elem × E4M3_block × weight_scale_2    block 16, E4M3, per-tensor global
```

Emit `E4M3_block = 2^(e_old + G)` and `weight_scale_2 = 2^-G`; the product is exactly
`2^(e_old)`. Split each 32-element block into two 16-element blocks with the same exponent
byte and every element keeps its effective scale. `G = 8 − max(e_old)` puts the largest block
at E4M3's top normal exponent, and the transcoder aborts if any `e_new` would fall outside
`[-6, 8]`.

The 4-bit weight bytes are copied verbatim — nibble order is unchanged between the formats.
Only scale metadata is rewritten. That is why the checkpoint *grows*: two extra tensors per
expert weight and twice as many block scales, 156 GiB → 163.5 GiB.

### Verifying before spending hours

`check-mtp-range.py` walks every expert group and reports the widest exponent span against
the 14-wide E4M3 window. For this model the widest was **6**, with zero `0xFF` bytes — so the
conversion is lossless with room to spare. Running that first is minutes; discovering a
clipped expert after a 3-hour transcode is not.

Afterwards, `transcode_0731_to_nvfp4.py verify` samples expert weights and compares
byte-for-byte plus the scale relationship: **64 OK / 0 BAD**.

### A partial re-transcode silently corrupted the index

Re-running with `--shards=45:48` regenerated `model.safetensors.index.json` from **only the
three shards it touched** — 138,365 tensors down to 9,313. The shard files were untouched;
the index was destroyed, and the model would not load.

Rebuilding by scanning every shard header directly restored **142,973 tensors / 164.0 GiB**
(the original 138,365 plus 4,608 new MTP scale tensors). If you ever run a partial transcode,
rebuild the index afterwards and check the count.

---

## Part 3 — Speculative decoding: the biggest win, found last

### Symptom: acceptance 0.121 and a deadlocking engine

With spec decode on, the engine wedged on the first request. Raising the worker RPC deadline
(`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` 300 → 1800) stopped the *timeout*, and then a real
number appeared: draft acceptance **0.121**, with barely half of drafts surviving a single
token.

### First hypothesis: a dead exemption. Correct, but not the whole story.

The transcode's `quantization_config` tried to exempt the MTP draft:

```json
"ignore": ["*.attn.*", "*.ffn.shared_experts.*", "head", "mtp.*"]
```

That exemption did nothing, for two independent reasons found by reading vLLM's source:

1. `Fp8Config.from_config` reads `ignored_layers` or `modules_to_not_convert`. **`ignore` is
   never read at all.**
2. `is_layer_skipped` defaults to `prefix_full_match` — **exact string equality**. `"mtp.*"`
   matches only a layer literally named `mtp.*`.

Confirmed directly: `is_layer_skipped('mtp.0.ffn.experts', [..., 'mtp.*'])` → `False`.

So the still-MXFP4 MTP experts were being routed to `ModelOptNvFp4FusedMoE` — the draft was
reading MXFP4 bytes through an NVFP4 interpreter. Fixing the key and using exact prefixes
**eliminated the deadlock**. Acceptance stayed at 0.121.

### The actual fix: transcode the MTP layers too

The right answer was not to exempt MTP — it was to convert it, so the whole model speaks one
scale convention. MTP experts turned out to be structurally identical to main-layer experts
(same `I8 [2048,2048]` weights, same `F8_E8M0 [2048,128]` scales, same naming), so the same
validated path applied. The transcoder's regex was hardcoded to `layers.*`, and its MTP
pattern only covered `mtp.0` when there are three blocks.

| | acceptance | accepted/draft | pos-0 survival |
|---|---|---|---|
| mixed conventions | 0.121 | 0.85 | 50.2% |
| **all NVFP4** | **0.31–0.38** | **2.2–2.3** | **76.4%** |

Spec decode went from unusable to a ~2× multiplier.

### The baseline we were chasing did not exist

Throughout, "0.55–0.72" was treated as the acceptance target. It came from a different build
and configuration. Measured on this cluster under matched conditions:

```
NVFP4 acceptance 0.273   (accepted/draft 1.91)
MXFP4 acceptance 0.274   (accepted/draft 1.92)
```

**Identical.** There was no residual gap, and the speculation about `input_scale = 1.0` or
transcode drift causing one was chasing nothing. *Measure your comparator under your own
conditions before treating any published number as a target.*

---

## Part 4 — Benchmarking, and four ways to get it wrong

Every item here produced a confident wrong number.

### 1. Estimated token counts (2.76× error)

A quick prefill harness built its filler as `token0 token1 token2 …` and assumed one word per
token. For this tokenizer that string is ~2.76 tokens/word, so a prompt labelled "4096
tokens" was really 11,288 and every prefill rate was understated by that factor. It turned a
real ~2.8× gap into a reported "8.2× slower" and pointed the investigation at the wrong
subsystem for an hour.

The repo already contained a correct implementation — `needles.py` verifies against
`/tokenize`. The quick parallel script silently lacked the care. **Reuse the careful thing.**

(Note the endpoint is at the **server root**. `POST /v1/tokenize` returns 404.)

### 2. Counting stream chunks instead of tokens (1.6× error)

With speculative decoding, one SSE chunk can carry several accepted draft tokens. A streaming
probe that counted chunks reported 160 "tokens" for a response the server counted as 256,
making decode look like 5.0 tok/s instead of 14.1. Fix: `stream_options: {include_usage:
true}` and use the server's `completion_tokens`.

### 3. Comparing at different `gpu-memory-utilization` (45% → 6%)

"MXFP4 holds 45% more KV context" was published internally and was mostly an artifact. NVFP4
had been run at util 0.835 — raised only enough to clear the spec-on 1M boot floor — while
MXFP4 ran its own default. At matched utilization:

```
NVFP4 @0.87   2,174,851 tokens (2.07× @1M)
MXFP4         2,307,250 tokens (2.20× @1M)      -> 6%, not 45%
```

The underlying cause is real (NVFP4 weights are ~4 GiB larger), but the magnitude was a knob
nobody had turned.

### 4. Benchmarking at `max-num-seqs 4` (2.5× error)

Every headline decode number was measured at `max-num-seqs 4`, chosen because the target
workload is 1–4 streams. That choice costs roughly **2.5×** on decode, *even single-stream*,
because the scheduler batches drafted tokens differently with a larger budget:

| `max-num-seqs` | KV tokens | single-stream decode |
|---|---|---|
| 4 | 1,654,544 | ~16 tok/s |
| 32 | 2,075,773 | 18.2 median, 23.4 best |
| 48 | 2,174,851 | **40.3 best** |

The 40.3 figure reproduced a number from the MXFP4 playbook that had looked unreachable all
day. The gap was never the model, the image, or the kernels — it was one scheduler setting.

---

## Part 5 — The thing that looks like memory exhaustion and isn't

Several times the engine wedged mid-benchmark. The obvious suspicion was memory: these are
128 GB unified-memory boxes running a 164 GB model across two nodes, and an earlier
investigation had attributed host freezes to swap thrash.

One-second instrumentation on both nodes says otherwise. At every wedge observed in this
work:

```
K=7 sweep     113 GB free, 0 swap
NIAH @131K    6.6 GB free, 0 swap
seqs=48 run   ample free, 0 swap
```

The mechanism, caught with exact numbers:

```
[rank1] JIT compilation during inference: _topk_log_softmax_kernel     09:50:40
[rank0] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=3466,
        OpType=_ALLGATHER_BASE, NumelIn=517120, NumelOut=1034240,
        Timeout(ms)=600000) ran for 600071 ms before timing out         10:05:11
```

The two ranks compile Triton kernels **independently**, each when it first meets a shape.
While one compiles, the other posts a collective and blocks. Past 600 seconds the watchdog
kills the engine. vLLM even warns about it — *"This causes a latency spike; consider
extending warmup to cover this shape/config"* — but the consequence on TP=2 is not a spike,
it is a wedge.

The fix is to warm across the shapes you will measure, including **prompt length, output
length, and concurrency**, then verify both ranks are quiet before timing anything. Done
right, both ranks compile at the *same timestamp*:

```
rank0  15:46:26        rank1  15:46:26
```

Two specific traps: a generic warmup at 8/64/512/2048 tokens does not compile the kernels a
131K prompt needs (`niah-run.sh` now warms at each target context), and warming at
`max_tokens=256` while measuring at 512 is enough of a shape difference to wedge the pair.

---

## Part 6 — Two dead ends worth recording

**FULL CUDA graphs.** The DSpark speculator's docstring says FULL graphs cover the whole
draft step, and it does emit `Capturing dspark CUDA graphs (FULL)` at startup — meaning under
`PIECEWISE` the draft runs eager on every drafted token. That looked like an easy 2×. It is
not available: `FULL_AND_PIECEWISE` hung after 443 s and `FULL_DECODE_ONLY` after 673 s, both
ending in `RuntimeError: cancelled` with zero JIT activity and >100 GB free. This reproduces
vLLM #40969. `PIECEWISE` is the only usable mode on GB10, and the draft-step graph tax is
permanent there.

**The anemll image.** The MXFP4 playbook credits `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` with
a rewritten DSpark speculator, so it was a natural suspect for the missing throughput. It is
the **same vLLM**: identical base commit `0.25.2.dev0+g752a3a504`, identical torch
`2.11.0+cu130` and flashinfer `0.6.15`, and byte-identical md5 on every hot-path file —
`dspark/speculator.py`, `dspark/utils.py`, `kv_cache_interface.py`, `mla/sparse_swa.py`,
`mla/flashmla_sparse.py`, `b12x_mxfp4_moe.py`, and both MoE oracles. Benchmarked directly it
gave 16.12 tok/s, the same range as the local build. The anemll port was already fully
present in the image we had been using all along.

Separately, the upstream `vllm/vllm-openai:pinned-dev403` image **cannot run MXFP4 0731 on
GB10 at all**, and each workaround fails differently: DeepGEMM on hits
`layout.hpp:60 Unknown SF transformation` (no `arch_major=12` branch); DeepGEMM off hits
`hyperconnection.hpp:56 Unsupported architecture` because the MHC prenorm GEMM needs it, and
silently drops the MoE backend to MARLIN; `flashinfer_b12x` is rejected as an NVFP4-only
backend; `flashinfer_cutlass` resolves to a variant `Mxfp4MoEMethod` does not accept. The
working configuration runs on `vllm-dsv4:src-sm121` with `--moe-backend flashinfer_b12x`.

---

## What the numbers finally said

Both quantizations, same image, same KV dtype, same warmup discipline, spec-on K=7:

| ctx | needles (MXFP4 / NVFP4) | prefill × | TTFT × | decode × |
|---|---|---|---|---|
| 4,318 | 5/5 · 5/5 | 1.31× | 1.31× | 1.83× |
| 33,375 | 5/5 · 5/5 | 1.32× | 1.32× | — |
| 132,973 | 5/5 · 5/5 | 1.23× | 1.23× | 1.27× |
| 265,781 | 5/5 · 5/5 | 1.14× | 1.14× | 1.57× |

And the accuracy question, from four independent angles:

- transcode bit-exactness: **64/64**
- draft acceptance: **0.273 vs 0.274**
- needle retrieval: **identical at every depth, every rung**
- teacher-forced logits: cross-build p95 |Δ| **0.1217** against a same-build run-to-run floor
  of **0.1442** — the two builds differ **less than one build differs from itself**

NVFP4 is faster and numerically equivalent. The narrowing advantage at long context
(1.31× → 1.14×) is the honest ceiling: only 6.3% of MMA work is FP4, and the BF16 attention
path grows with context while the expert GEMMs do not.
