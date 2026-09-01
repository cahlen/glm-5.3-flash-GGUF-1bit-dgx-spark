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
# FA_ALL_QUANTS=OFF builds only the common flash-attention KV combinations
# (f16/f16, q8_0/q8_0, q4_0/q4_0). q8_0 KV — what glm53.env uses — is in that
# set. Set FA_ALL_QUANTS=ON only if you want to run an exotic K/V pairing;
# it roughly doubles compile time.

cmake --build build --config Release -j "$JOBS" --target llama-server llama-cli

echo
echo "built: $SRC/build/bin/llama-server"
"$SRC/build/bin/llama-server" --version 2>&1 | head -2
