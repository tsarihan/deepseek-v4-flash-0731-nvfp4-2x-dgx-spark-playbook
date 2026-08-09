# QUICKSTART

From an MXFP4 DeepSeek-V4-Flash-0731 checkpoint to a serving NVFP4 endpoint on 2× DGX Spark.

If you only want to serve the model, **skip step 1** and pull the finished weights:
https://huggingface.co/tomsarihan/DeepSeek-V4-Flash-0731-NVFP4

---

## 0. Prerequisites

- 2× DGX Spark (GB10, `sm_121a`), 128 GB unified memory each
- Both nodes on the CX7 200GbE fabric. **Verify it before trusting any number** — if the
  HCA does not match the interface carrying the inter-node subnet, NCCL silently falls back
  to the management network:
  ```bash
  ib_write_bw -d rocep1s0f0 -F --report_gbits <peer-roce-ip>
  # expect ~100+ Gb/s. Single digits = wrong interface.
  ```
- `vllm-dsv4:src-sm121` on **both** nodes (vLLM `0.25.2.dev0+g752a3a504`, torch
  `2.11.0+cu130`, flashinfer `0.6.15`)
- `vm.swappiness = 0` on both nodes
- ~340 GB free disk if you are transcoding (source 156 GB + output 164 GB)

---

## 1. Transcode MXFP4 → NVFP4 (skip if pulling from HF)

Runs on CPU inside the container. No GPU needed. ~2–3 hours.

```bash
# Pre-flight: does every expert fit the E4M3 window? (minutes, and it saves hours)
docker run --rm --entrypoint /bin/bash \
  -v /data/models/deepseek-v4-flash-0731:/src:ro \
  -v $PWD/scripts:/scripts:ro \
  vllm-dsv4:src-sm121 -c "cd /scripts && python3 check-mtp-range.py /src"
# want: "PASS - every expert fits; transcode is lossless"

# Transcode all 48 shards
docker run --rm --entrypoint /bin/bash \
  -v /data/models/deepseek-v4-flash-0731:/src:ro \
  -v /data/models/deepseek-v4-flash-0731-nvfp4:/out \
  -v $PWD/scripts:/scripts:ro \
  vllm-dsv4:src-sm121 -c "cd /scripts && python3 transcode_0731_to_nvfp4.py transcode /src /out"

# Verify bit-exactness (want: 64 OK / 0 BAD)
docker run --rm --entrypoint /bin/bash \
  -v /data/models/deepseek-v4-flash-0731:/src:ro \
  -v /data/models/deepseek-v4-flash-0731-nvfp4:/out:ro \
  -v $PWD/scripts:/scripts:ro \
  vllm-dsv4:src-sm121 -c "cd /scripts && python3 transcode_0731_to_nvfp4.py verify /src /out"
```

> **⚠ Never run `transcode` with `--shards=N:M` unless you rebuild the index afterwards.**
> A partial run regenerates `model.safetensors.index.json` from only the shards it touched,
> truncating it (138,365 tensors → 9,313 here) and leaving the model unloadable. The shard
> files are fine; rebuild the index by scanning every shard header. Correct totals for this
> model: **142,973 tensors / 164.0 GiB**.

The MTP layers **are** transcoded. Leaving them MXFP4 while the main stack is NVFP4 puts two
scale conventions in one model and collapses draft acceptance from ~0.31 to 0.121.

---

## 2. Copy to both nodes

Each node loads its own copy — there is no shared filesystem.

```bash
rsync -a --info=progress2 /data/models/deepseek-v4-flash-0731-nvfp4/ \
      user@node2:/data/models/deepseek-v4-flash-0731-nvfp4/
# then confirm md5 parity on config.json + the index on both nodes
```

---

## 3. Serve

`scripts/serve-0731-nvfp4.sh` takes `NODE_RANK` and env overrides.
**Start rank 1 first** — rank 0 owns the TCPStore, and a rank 1 that outlives its rank 0
must be relaunched.

```bash
# node 2 (rank 1, headless)
NODE_RANK=1 SPEC_DECODE=on MTP_K=7 MAX_NUM_SEQS=32 GPU_UTIL=0.86 ./serve-0731-nvfp4.sh

# node 1 (rank 0, serves the API on :8891)
NODE_RANK=0 SPEC_DECODE=on MTP_K=7 MAX_NUM_SEQS=32 GPU_UTIL=0.86 ./serve-0731-nvfp4.sh
```

Expect ~8–10 minutes to `Application startup complete` (weight load dominates).

**Confirm the NVFP4 path actually engaged** — three lines, all of which should appear:

```
DeepSeek V4 expert_dtype resolved to 'fp4'
Detected ModelOpt NVFP4 checkpoint (quant_algo=NVFP4)
Using 'FLASHINFER_CUTLASS' NvFp4 MoE backend
```

Backend *selection* is logged at load time and is not proof the kernel runs — the first MoE
forward is where an ABI or format problem surfaces.

### Configuration that matters

| setting | value | why |
|---|---|---|
| `MAX_NUM_SEQS` | **32** | 4 costs ~2.5× on decode, even single-stream. 48 is faster still but less stable. |
| `GPU_UTIL` | **0.86** | 0.80 leaves KV below the 1M floor once MTP draft weights load |
| `MTP_K` | **7** | fastest single-stream; 5 wins at 4 concurrent streams |
| `CUDAGRAPH_MODE` | **PIECEWISE** | every FULL-family mode wedges on GB10 (vLLM #40969) |
| `LINEAR_BACKEND` | `deep_gemm` | CUTLASS `scaled_mm` dispatch fails on `sm_121a` |
| `MOE_BACKEND` | *(empty)* | auto → `FLASHINFER_CUTLASS`. **Never `marlin` on a Spark.** |
| `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` | 1800 | 300 is too short for the spec-decode cold start |

---

## 4. Smoke test

```bash
curl -s http://<rank0>:8891/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash-0731-nvfp4","prompt":"The capital of France is",
       "max_tokens":32,"temperature":0}' | jq -r '.choices[0].text'
```

Coherent output means it loads and generates. It does **not** prove the quantization is
correct — a uniform scale error preserves argmax order, so greedy text stays fluent while the
distribution is wrong. For that, use the numerical gate:

```bash
scripts/capture-logits.sh nvfp4 http://<rank0>:8891/v1 deepseek-v4-flash-0731-nvfp4
# (later, against the MXFP4 build)
scripts/capture-logits.sh mxfp4 http://<rank0>:8891/v1 deepseek-v4-flash-0731
scripts/compare-logits.py results/logits-nvfp4.json results/logits-mxfp4.json --floor 0.1442
```

---

## 5. Benchmark

**Warm first, at the shape you measure.** A cold run measures kernel compilation, and worse,
it can wedge the pair — the two ranks JIT independently, and while one compiles the other
blocks in a collective until the 600 s watchdog fires.

```bash
scripts/decode-rate.sh deepseek-v4-flash-0731-nvfp4 10   # 10 runs; 3 is not a median
scripts/ttft-curve.sh  nvfp4 http://<rank0>:8891/v1 deepseek-v4-flash-0731-nvfp4
scripts/niah-run.sh    nvfp4 deepseek-v4-flash-0731-nvfp4 4096,32768,131072,262144
```

`niah-run.sh` warms at each target context before scoring, which is what keeps the long
rungs from wedging.

---

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `No available shared memory broadcast block` repeating | the peer rank is missing | check for both `Worker_TP0` and `Worker_TP1` |
| `RuntimeError: cancelled` / NCCL collective timeout, memory fine | JIT skew between ranks | warm across your measurement shapes; assert 0 compiles on both |
| `RPC call to sample_tokens timed out` on request #1 | 300 s worker deadline vs spec cold start | `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800` |
| `7.63 GiB KV needed > 7.39 available` at startup | MTP draft weights came out of the KV budget | `GPU_UTIL=0.86` |
| Draft acceptance ~0.12 | MTP still MXFP4 while the model is NVFP4 | transcode the MTP layers |
| `TypeError: Expected 7 but got 8 arguments` on first MoE forward | stale prebuilt `fused_moe_120.so` | delete the image's AOT copy; keep the node's 8-arg `/cache` build |
| Wildly low tok/s | counting SSE chunks, or a cold first run | use `include_usage`; warm first |
