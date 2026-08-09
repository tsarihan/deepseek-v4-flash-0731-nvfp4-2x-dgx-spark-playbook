#!/usr/bin/env bash
# Wrapper entrypoint: remove the STALE prebuilt flashinfer CUTLASS MoE .so so
# build_and_load uses the good 8-arg build (from /cache, or JIT-compiled fresh
# from the bundled 8-arg source), then exec the real vllm serve.
# Root cause: the image's flashinfer_jit_cache/.../fused_moe_120.so was built
# against an OLDER 7-arg `init` FFI; flashinfer 0.6.15 MoERunner passes 8 args
# (added use_fused_finalize) -> "Expected 7 but got 8" crash on first MoE forward.
#
# Verified arities (2026-08-08, nm -D on each .so):
#   image AOT    : _Z4init10DLDataTypeS_S_bbbb   = 7 args  -> STALE, delete
#   /cache build : _Z4init10DLDataTypeS_S_bbbbb  = 8 args  -> GOOD, keep
# So we delete ONLY the image's AOT copy. Deleting the /cache build too (the
# previous behaviour) threw away a valid kernel and forced a needless ~35 min
# nvcc rebuild per node on every launch.
set -e

STALE=/usr/local/lib/python3.12/dist-packages/flashinfer_jit_cache/jit_cache/fused_moe_120/fused_moe_120.so
CACHED=/cache/flashinfer/.cache/flashinfer/0.6.15/121a/cached_ops/fused_moe_120/fused_moe_120.so

rm -f "$STALE" 2>/dev/null || true

# Keep the cached build only if it really is the 8-arg ABI; otherwise drop the
# whole cached_ops dir so flashinfer JIT-rebuilds it from the bundled source.
if [ -f "$CACHED" ]; then
  if nm -D --defined-only "$CACHED" 2>/dev/null | grep -q '_Z4init10DLDataTypeS_S_bbbbb'; then
    echo "[jitfix] cached fused_moe_120.so is 8-arg ABI -> reusing (no rebuild)"
  else
    echo "[jitfix] cached fused_moe_120.so is NOT 8-arg ABI -> purging for JIT rebuild"
    rm -rf "$(dirname "$CACHED")" 2>/dev/null || true
  fi
else
  echo "[jitfix] no cached fused_moe_120.so -> flashinfer will JIT-build it"
fi

exec vllm serve "$@"
