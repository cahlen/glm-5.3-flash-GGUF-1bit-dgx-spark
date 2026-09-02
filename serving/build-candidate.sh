#!/usr/bin/env bash
# Build a CANDIDATE llama.cpp into its own tree, leaving the running one alone.
#
# The serving build lives at ~/llama.cpp-glm5 and is what glm53.service execs.
# Nothing here touches it: a git worktree checks the candidate commit out into a
# separate directory sharing the same object store, so the running binary, its
# build directory and its working tree are all untouched. Rolling back is
# pointing GLM53_BIN at the old path — no rebuild, no checkout.
#
#   ./build-candidate.sh <commit> [dir]
set -euo pipefail

COMMIT="${1:?usage: build-candidate.sh <commit> [dir]}"
SRC="${SRC:-$HOME/llama.cpp-glm5}"
DEST="${2:-$HOME/llama.cpp-glm5-$COMMIT}"
BRANCH="${BRANCH:-glm5next/upstream}"
JOBS="${JOBS:-$(nproc)}"
CUDA_ARCH="${CUDA_ARCH:-121}"

echo "=== candidate build: $COMMIT -> $DEST ==="
cd "$SRC"

# Never move the serving tree. Record where it is so a mistake is visible.
echo "serving tree stays at: $(git rev-parse --short HEAD) ($(git branch --show-current))"

git fetch origin "$BRANCH"

if [ -d "$DEST" ]; then
  echo "worktree already exists at $DEST — reusing"
else
  git worktree add --detach "$DEST" "$COMMIT"
fi

cd "$DEST"
echo "candidate tree at: $(git rev-parse --short HEAD)"

cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=ON \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DGGML_CUDA_FA_ALL_QUANTS="${FA_ALL_QUANTS:-OFF}" \
  -DBUILD_SHARED_LIBS=OFF

cmake --build build --config Release -j "$JOBS" --target llama-server llama-cli

echo
echo "=== built ==="
"$DEST/build/bin/llama-server" --version 2>&1 | head -2
echo "serving tree still at: $(cd "$SRC" && git rev-parse --short HEAD)"
echo
echo "To test:  GLM53_BIN=$DEST/build/bin/llama-server  (via a systemd drop-in)"
