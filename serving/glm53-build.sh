#!/usr/bin/env bash
# Reproducible build of the GLM-5.3/NextN-capable llama.cpp for the DGX Spark.
#
# WHY THIS FORK: GLM-5.3-Flash is arch 'glm5next' with an in-GGUF NextN/MTP head
# (nextn_predict_layers=1). Upstream ggml-org/llama.cpp did not carry that
# implementation when this was built; unslothai's glm5next/upstream branch does,
# and it is where `--spec-type draft-mtp` comes from. MTP was verified genuinely
# active here, not stubbed — the server logs a real per-request
# "draft acceptance = ... mean len = ..." line, and decode measured ~30 t/s with
# it against ~18 t/s without.
#
#   REPO   https://github.com/unslothai/llama.cpp
#   BRANCH glm5next/upstream
#   COMMIT d07e71e   ("Add MTP support")
#
# Pinned to the commit on purpose. Re-point COMMIT to move, do not float on the
# branch head: the MTP graph is young code and a silent change to it shows up as
# an acceptance-rate regression, not a build error.
set -euo pipefail

REPO="${REPO:-https://github.com/unslothai/llama.cpp}"
BRANCH="${BRANCH:-glm5next/upstream}"
COMMIT="${COMMIT:-d07e71e}"
SRC="${SRC:-$HOME/llama.cpp-glm5}"
JOBS="${JOBS:-$(nproc)}"

# GB10 (Grace-Blackwell) is compute capability 12.1. Building only this arch keeps
# the compile short; add more if the binary must be portable off this box.
CUDA_ARCH="${CUDA_ARCH:-121}"

if [ ! -d "$SRC/.git" ]; then
  git clone "$REPO" "$SRC"
fi
cd "$SRC"

# Never discard local state silently — this tree has been hand-modified before.
if [ -n "$(git status --porcelain)" ]; then
  echo "build: working tree is dirty; stashing to a named ref rather than discarding" >&2
  git stash push -u -m "glm53-build $(date -u +%Y%m%dT%H%M%SZ)"
  echo "build: recover with 'git stash list' / 'git stash pop'" >&2
fi

git fetch origin "$BRANCH"
git checkout -B "$BRANCH" "$COMMIT"

echo "build: $(git rev-parse --short HEAD) on $BRANCH"

cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=ON \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DGGML_CUDA_FA_ALL_QUANTS="${FA_ALL_QUANTS:-OFF}"
# FA_ALL_QUANTS=OFF compiles exactly four flash-attention KV variants —
# f16/f16, bf16/bf16, q8_0/q8_0, q4_0/q4_0 — and requires K and V to be the SAME
# type. q8_0/q8_0, what glm53.env uses, is in that set.
#
# An unsupported pairing does not fail loudly: llama.cpp returns
# BEST_FATTN_KERNEL_NONE and silently falls back to non-flash attention, so a
# mismatched -ctk/-ctv yields a working but quietly slower server.
#
# =ON unlocks mixed K/V types plus q4_1/q5_0/q5_1 at roughly double the compile
# time. Not worth it for this model: KV is only ~1.7 GiB at 128K, so there is
# almost nothing left to save.

cmake --build build --config Release -j "$JOBS" --target llama-server llama-cli

echo
echo "built: $SRC/build/bin/llama-server"
"$SRC/build/bin/llama-server" --version 2>&1 | head -2
