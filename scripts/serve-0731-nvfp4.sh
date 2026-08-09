#!/usr/bin/env bash
# DeepSeek-V4-Flash-0731 (NVFP4 experts, transcoded) on 2x DGX Spark, TP=2.
#
# Same TP=2 1M-ctx topology as serve-0731-official.sh (the MXFP4 prod) BUT:
#   - IMAGE = vllm-dsv4:src-sm121  (the precanary-validated image: builds+runs the
#     flashinfer CUTLASS NVFP4 MoE kernel on sm_121a with FLASHINFER_JIT_DEBUG=0).
#     src-sm121 has the full DeepseekV4 stack: vllm/models/deepseek_v4/nvidia/
#     {model,mtp,dspark}.py + the modelopt NVFP4 quant path (quant_config.py:
#     moe_quant_algo=="NVFP4" -> ModelOptNvFp4FusedMoE -> select_nvfp4_moe_backend
#     -> FLASHINFER_CUTLASS -> FlashInferExperts, SAME kernel the qwen36 precanary
#     validated).
#   - MODEL_DIR = the transcoded 0731-NVFP4 checkpoint (experts 4-tensor modelopt
#     format: weight U8 + weight_scale E4M3 + weight_scale_2 f32 2^-G + input_scale
#     1.0; attn/shared/MTP verbatim).
#   - JIT-fix wrapper entrypoint (serve-wrap-jitfix.sh): rm the stale prebuilt AOT
#     fused_moe_120.so (arity-skew bug) so flashinfer JIT-rebuilds from the 8-arg
#     source. FLASHINFER_JIT_DEBUG=0 = RELEASE build (~35min if rebuild needed),
#     NOT the 4h debug build. NVCC_THREADS=1 + MAX_JOBS=4 = predictable ptxas mem.
#   - /cache = vllm-cache-nvfp4-qwen (REUSES the precanary's already-built
#     fused_moe_120.so at /cache/flashinfer/.cache/.../fused_moe_120.so -> if the
#     JitSpec matches, 0731 loads it instantly; else rebuilds alongside it).
#
#   spark-1:  NODE_RANK=0 ./serve-0731-nvfp4.sh
#   spark-2:  NODE_RANK=1 ./serve-0731-nvfp4.sh
#
# NVFP4 is a TTFT/prefill win (both paths 4-bit; decode is bandwidth-bound, ~0 gain).
# First boot uses SPEC_DECODE=off to ISOLATE the NVFP4 expert path (one fewer
# variable); set SPEC_DECODE=on + MTP_K=7 to match the MXFP4 prod for A/B.
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/data/models/deepseek-v4-flash-0731-nvfp4}"
IMAGE="${IMAGE:-vllm-dsv4:src-sm121}"
NAME="${NAME:-dsv4-0731-nvfp4}"
PORT="${PORT:-8891}"
NODE_RANK="${NODE_RANK:?set NODE_RANK=0 on spark-1, 1 on spark-2}"
MASTER_ADDR="${MASTER_ADDR:-${RANK0_IP:-10.0.0.1}}"
MASTER_PORT="${MASTER_PORT:-25200}"
WRAP="${WRAP:-${REPO:-$HOME/nvfp4-playbook}/scripts/serve-wrap-jitfix.sh}"

# --- tunables ---
KV_DTYPE="${KV_DTYPE:-nvfp4_ds_mla}"   # MATCHES the published MXFP4 playbook (experts MXFP4, KV NVFP4).
# The old comment here said nvfp4_ds_mla was "absent upstream" -- that is STALE:
# kv_cache_interface.py:70,381,611 + sparse_swa.py:87 all handle it in src-sm121.
# Running fp8_ds_mla CONFOUNDED the NVFP4-vs-MXFP4 comparison: it changed the expert
# format AND downgraded the KV cache at once. Same 584B page size, so it costs 0 KV bytes.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"  # 1M per-stream context (the goal)
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"           # sweep 1/4/8/16/32 needs >=32; v1 c16 hung on seqs=8
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-8192}"
GPU_UTIL="${GPU_UTIL:-0.80}"
MTP_K="${MTP_K:-7}"
DRAFT_SAMPLE="${DRAFT_SAMPLE:-greedy}"
SPEC_DECODE="${SPEC_DECODE:-off}"           # OFF for first boot (isolate NVFP4); on for prod-parity A/B
BLOCK_SIZE="${BLOCK_SIZE:-256}"
# MoE backend: leave EMPTY = auto -> modelopt path selects FLASHINFER_CUTLASS for NVFP4.
# NEVER --moe-backend marlin on a Spark; avoid deep_gemm_mega_moe (blocked MXFP4/mega-moe path).
MOE_BACKEND="${MOE_BACKEND:-}"
# FP8 linear layers (attn/shared) need DeepGEMM's kernel, NOT CUTLASS —
# cutlass_scaled_mm dispatch fails on sm_121a (scaled_mm_helper.hpp:17).
# VLLM_USE_DEEP_GEMM=1 makes Fp8LinearMethod select DeepGEMM instead of
# CutlassFp8BlockScaledMMKernel. The vendored vllm.third_party.deep_gemm in
# src-sm121 (vllm 0.25.2.dev) DOES support sm_121a (the "no arch-12" bug was
# 0.26.x standalone, not this vendored 0.25.x build). E8M0 scales match the
# 0731 quant_config (scale_fmt=ue8m0).
LINEAR_BACKEND="${LINEAR_BACKEND:-deep_gemm}"
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-PIECEWISE}"  # FULL hangs on GB10 after ~6 reqs (#40969)
# Worker RPC deadline. vLLM defaults VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS to 300 and
# VLLM_ENGINE_ITERATION_TIMEOUT_S to 60. With SPEC_DECODE=on the FIRST request must
# warm/compile the MTP sampling path across the 200Gb link, which blew past 300s ->
# `TimeoutError: RPC call to sample_tokens timed out` killed EngineCore on request #1
# (2026-08-09 00:38, exit 0 = clean shutdown, NOT an OOM or a power event).
# Distinct from the FULL_AND_PIECEWISE cudagraph hang in serve-0731-official.sh:44,
# which produces the same message from a different cause -- we already run PIECEWISE.
# Also distinct from the client-side API_TIMEOUT_MS in RUNBOOK.md:203.

MOE_ARGS=()
[ -n "$MOE_BACKEND" ] && MOE_ARGS+=(--moe-backend "$MOE_BACKEND")
[ -n "$CUDAGRAPH_MODE" ] && MOE_ARGS+=(--compilation-config "{\"cudagraph_mode\":\"${CUDAGRAPH_MODE}\"}")
[ -n "$LINEAR_BACKEND" ] && MOE_ARGS+=(--linear-backend "$LINEAR_BACKEND")
TOKENIZER_ARGS=(--tokenizer-mode deepseek_v4)

# Parity / tool-call args (match MXFP4 prod so A/B is clean)
PARITY_ARGS=(
  --tool-call-parser deepseek_v4 --enable-auto-tool-choice
  --reasoning-parser deepseek_v4
  --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"<think>","reasoning_end_str":"</think>"}'
  --default-chat-template-kwargs "{\"thinking\":true,\"reasoning_effort\":\"${THINKING:-low}\"}"
  --generation-config vllm
)

if [ "$SPEC_DECODE" = "off" ]; then
  SPEC_ARGS=(); CAPTURE=$MAX_NUM_SEQS
else
  SPEC_ARGS=(--speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":${MTP_K},\"draft_sample_method\":\"${DRAFT_SAMPLE}\"}")
  CAPTURE=$(( MAX_NUM_SEQS * (MTP_K + 1) ))
fi

[ "$NODE_RANK" = "0" ] && HEADLESS=() || HEADLESS=(--headless)

# Pin engine traffic to this node's CX7 200Gb interface (not the 1GbE LAN).
if [ -z "${VLLM_HOST_IP:-}" ]; then
  [ "$NODE_RANK" = "0" ] && VLLM_HOST_IP=${RANK0_IP:-10.0.0.1} || VLLM_HOST_IP=${RANK1_IP:-10.0.0.2}
fi

JIT_VOL="${JIT_VOL:-/data/models/vllm-cache-nvfp4-qwen/flashinfer-jit-cache}"
mkdir -p "$JIT_VOL"

DOCKER="docker"; command -v docker >/dev/null && docker ps >/dev/null 2>&1 || DOCKER="sudo docker"
$DOCKER rm -f "$NAME" >/dev/null 2>&1 || true

$DOCKER run -d --name "$NAME" --gpus all --network host --ipc host --shm-size 64g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --device /dev/infiniband:/dev/infiniband \
  --entrypoint /bin/bash \
  -v "$MODEL_DIR":/model:ro \
  -v /data/models/vllm-cache-nvfp4-qwen:/cache \
  -v "$JIT_VOL":/root/.cache/flashinfer \
  -v "$WRAP":/serve-wrap.sh:ro \
  -e VLLM_CACHE_ROOT=/cache \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e CUTE_DSL_ARCH=sm_121a -e FLASHINFER_WORKSPACE_BASE=/cache/flashinfer \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
  -e VLLM_USE_DEEP_GEMM="${USE_DEEP_GEMM:-1}" \
  -e VLLM_USE_DEEP_GEMM_E8M0="${USE_DEEP_GEMM_E8M0:-1}" \
  -e VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES="${USE_DG_TMA:-1}" \
  -e VLLM_CUDART_SO_PATH="/usr/local/cuda/lib64/libcudart.so.13" \
  -e FLASHINFER_JIT_VERBOSE=1 \
  -e FLASHINFER_JIT_DEBUG=0 \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${EXEC_TIMEOUT:-1800}" \
  -e VLLM_ENGINE_ITERATION_TIMEOUT_S="${ITER_TIMEOUT:-600}" \
  -e MAX_JOBS="${MAX_JOBS:-2}" \
  -e FLASHINFER_NVCC_THREADS="${FLASHINFER_NVCC_THREADS:-1}" \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA="${NCCL_IB_HCA:-rocep1s0f0}" \
  -e NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp1s0f0np0}" \
  -e GLOO_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp1s0f0np0}" \
  -e TP_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp1s0f0np0}" \
  -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_IB_ROCE_VERSION_NUM=2 \
  -e NCCL_CROSS_NIC=1 -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_DEBUG=WARN \
  "$IMAGE" /serve-wrap.sh \
    /model \
    --served-model-name deepseek-v4-flash-0731-nvfp4 \
    --host 0.0.0.0 --port "$PORT" \
    --trust-remote-code \
    "${TOKENIZER_ARGS[@]}" \
    "${MOE_ARGS[@]}" \
    --tensor-parallel-size 2 \
    --kv-cache-dtype "$KV_DTYPE" \
    --block-size "$BLOCK_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
    --max-cudagraph-capture-size "$CAPTURE" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --enable-prefix-caching --enable-chunked-prefill --async-scheduling \
    --enable-prompt-tokens-details \
    "${PARITY_ARGS[@]}" \
    "${SPEC_ARGS[@]}" \
    --distributed-executor-backend mp \
    --nnodes 2 --node-rank "$NODE_RANK" \
    --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
    "${HEADLESS[@]}"

echo "[$NAME] rank=$NODE_RANK kv=$KV_DTYPE spec=$SPEC_DECODE seqs=$MAX_NUM_SEQS TP=2 launching on src-sm121 (NVFP4 modelopt + JIT-fix)."
echo "Watch: $DOCKER logs -f $NAME"
[ "$NODE_RANK" = "0" ] && echo "API: http://0.0.0.0:${PORT}/v1  (first boot: reuses precanary .so if JitSpec matches, else ~35min rebuild)"