---
license: other
license_name: deepseek
license_link: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/LICENSE
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
tags:
  - nvfp4
  - fp4
  - quantized
  - modelopt
  - deepseek_v4
  - dgx-spark
  - gb10
  - vllm
pipeline_tag: text-generation
library_name: vllm
---

# DeepSeek-V4-Flash-0731 — NVFP4 experts

A format transcode of [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
from **MXFP4** to **NVFP4** routed experts, validated serving TP=2 across two DGX Spark
(GB10, `sm_121a`) nodes at 1M context.

**Playbook, transcoder, and full benchmarks:**
https://github.com/tsarihan/deepseek-v4-flash-0731-nvfp4-2x-dgx-spark-playbook

© 2026 Tom Sarihan, Desnet AI LLC.

---

## Why this exists

On consumer/edge Blackwell (`sm_121a`), the native FP4 MoE path is FlashInfer's CUTLASS
NVFP4 kernel. The stock 0731 checkpoint ships MXFP4 experts, which do not take that path.
This checkpoint converts them.

Measured against the MXFP4 original on the same hardware and the same vLLM image:

| axis | NVFP4 vs MXFP4 |
|---|---|
| Prefill / TTFT | **1.14–1.32× faster** (4K → 265K context) |
| Single-stream decode | **1.27–1.83× faster** |
| Needle retrieval | **identical** — 5/5 at all depths to 265,781 tokens |
| Draft acceptance | **identical** — 0.273 vs 0.274 |
| Logit distribution | differs **less than one build differs from itself** between runs |
| KV capacity | ~6% less at matched `gpu-memory-utilization` |

Peak single-stream decode observed: **40.28 tok/s** (spec-on, `MTP_K=7`, `max-num-seqs 48`).

## What was converted

**Routed experts only** — all 43 main layers plus all 3 MTP (speculative draft) blocks,
2,304 MTP expert tensors included. Attention, shared experts, the sparse-MLA indexer, norms
and embeddings are unchanged.

Converting the MTP blocks is **required**, not optional. Leaving them MXFP4 while the main
stack is NVFP4 puts two scale conventions in one model, and draft acceptance collapses from
~0.31 to 0.121 (with the engine deadlocking as well).

## The conversion is algebraically lossless

```
MXFP4  value = E2M1_elem × 2^(e_old)                     block 32, E8M0 scale
NVFP4  value = E2M1_elem × E4M3_block × weight_scale_2    block 16, plus a per-tensor global
```

With `E4M3_block = 2^(e_old + G)` and `weight_scale_2 = 2^-G`, the product is exactly the
original `2^(e_old)`. Each 32-element block splits into two 16-element blocks carrying the
same exponent, so every element keeps its effective scale. `G = 8 − max(e_old)` per expert
weight (w1/w3 share a G over their union; w2 independent).

**The 4-bit weight bytes are copied verbatim.** Only scale metadata is rewritten — which is
why the checkpoint is larger than the source (163.5 GiB vs 156 GiB): two extra tensors per
expert weight and twice as many block scales.

Verified: pre-flight confirms every expert group fits the E4M3 window (widest exponent span
6 of 14, zero `0xFF` bytes), and bit-exactness sampling reports **64 OK / 0 BAD**.

## Tensor layout

Per expert weight `{w1,w2,w3}`:

| tensor | dtype | note |
|---|---|---|
| `.weight` | uint8 | E2M1 packed, byte-identical to source |
| `.weight_scale` | float8_e4m3fn | `[R, K/16]` — doubled from the source's `[R, K/32]` |
| `.weight_scale_2` | float32 scalar | `2^-G` |
| `.input_scale` | float32 scalar | `1.0` (activations quantize dynamically) |

`quantization_config` keeps `quant_method: "fp8"` with `moe_quant_algo: "NVFP4"`, which routes
experts to `ModelOptNvFp4FusedMoE` → `FLASHINFER_CUTLASS`. Attention and shared experts stay
FP8 and untouched.

## Serving

Requires a vLLM build with the DeepSeek-V4 stack and `sm_121a` NVFP4 MoE kernels
(validated on `0.25.2.dev0+g752a3a504`, torch `2.11.0+cu130`, flashinfer `0.6.15`).

```bash
vllm serve /path/to/DeepSeek-V4-Flash-0731-NVFP4 \
  --trust-remote-code --tokenizer-mode deepseek_v4 \
  --tensor-parallel-size 2 --nnodes 2 --node-rank <0|1> \
  --master-addr <rank0-ip> --master-port 25200 \
  --kv-cache-dtype fp8_ds_mla --block-size 256 \
  --max-model-len 1048576 --max-num-seqs 32 \
  --gpu-memory-utilization 0.86 \
  --linear-backend deep_gemm \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}' \
  --enable-prefix-caching --enable-chunked-prefill --async-scheduling
```

Confirm the NVFP4 path engaged — all three lines should appear at load:

```
DeepSeek V4 expert_dtype resolved to 'fp4'
Detected ModelOpt NVFP4 checkpoint (quant_algo=NVFP4)
Using 'FLASHINFER_CUTLASS' NvFp4 MoE backend
```

### Settings that matter on GB10

| setting | value | why |
|---|---|---|
| `--max-num-seqs` | 32 | 4 costs ~2.5× on decode, even single-stream |
| `--gpu-memory-utilization` | 0.86 | 0.80 leaves KV under the 1M floor once MTP weights load |
| `cudagraph_mode` | `PIECEWISE` | every FULL-family mode hangs on GB10 (vLLM #40969) |
| `--linear-backend` | `deep_gemm` | CUTLASS `scaled_mm` dispatch fails on `sm_121a` |
| `--moe-backend` | *(omit)* | auto-selects `FLASHINFER_CUTLASS`. **Never `marlin` here.** |
| `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` | 1800 | 300 is too short for the spec-decode cold start |

## Hardware validated on

2× DGX Spark (GB10, `sm_121a`, 128 GB unified memory each), TP=2 over CX7 200GbE RoCE
(measured 109 Gb/s). Weights load ~78 GiB per node; KV reaches 2.07M tokens at
`gpu-memory-utilization 0.87`.

## Limitations

- Only the routed expert GEMMs are FP4 — **6.3% of MMA instructions** in the compiled MoE
  kernel; the rest stay BF16/FP8. That bounds the achievable speedup and is why the prefill
  advantage narrows at long context (1.31× at 4K → 1.14× at 265K).
- Larger on disk than the MXFP4 source, and costs ~6% KV capacity at matched utilization.
- Needs a vLLM build carrying `sm_121a` NVFP4 MoE kernels; stock upstream images do not run
  this model on GB10.

## License

The weights are a format transcode of DeepSeek-V4-Flash-0731 and remain subject to that
model's original license. The transcoder and tooling are Apache-2.0.
