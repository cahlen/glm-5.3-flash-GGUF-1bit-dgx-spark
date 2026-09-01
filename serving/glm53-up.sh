#!/usr/bin/env bash
# Launch GLM-5.3-Flash on llama.cpp from serving/glm53.env.
#
# This is what glm53.service execs, and it is also the supported way to start the
# server by hand. Every tunable lives in glm53.env so the unit file never has to
# be edited to change context, quant, MTP depth or sampling.
#
#   ./glm53-up.sh                  start with glm53.env as-is
#   DRY_RUN=1 ./glm53-up.sh        print the command, start nothing
#   GLM53_CTX=262144 ./glm53-up.sh override a single value for one run
#   ENV_FILE=other.env ./glm53-up.sh
#
# Env passed on the command line WINS over glm53.env, so the bench scripts can
# sweep one variable without rewriting the file.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-$DIR/glm53.env}"

[ -r "$ENV_FILE" ] || { echo "glm53-up: cannot read $ENV_FILE" >&2; exit 1; }

# Source the file without letting it clobber variables already set in the
# environment: snapshot what was pre-set, source, then restore. This is what
# makes `GLM53_CTX=262144 ./glm53-up.sh` work.
_preset="$(env | grep -o '^GLM53_[A-Z0-9_]*' || true)"
declare -A _keep
for k in $_preset; do _keep[$k]="${!k}"; done
# ENV_FILE is configurable by design, so its path is not a constant the linter
# can follow. The suppressing directive must sit immediately before the `.`
# itself: on a compound `set -a; . file; set +a` line a directive attaches to
# `set -a` and silently does nothing.
#
# (Careful when editing the prose above: a comment line that STARTS with the
# linter's name is parsed as a directive and fails the build.)
set -a
# shellcheck source=/dev/null
. "$ENV_FILE"
set +a
for k in "${!_keep[@]}"; do export "$k=${_keep[$k]}"; done

req() { [ -n "${!1:-}" ] || { echo "glm53-up: $1 is not set in $ENV_FILE" >&2; exit 1; }; }
req GLM53_BIN; req GLM53_MODEL; req GLM53_PORT; req GLM53_CTX

if [ "${DRY_RUN:-0}" != "1" ]; then
  [ -x "$GLM53_BIN" ] || { echo "glm53-up: llama-server not built: $GLM53_BIN" >&2; exit 1; }
  [ -r "$GLM53_MODEL" ] || { echo "glm53-up: model not readable: $GLM53_MODEL" >&2; exit 1; }
fi

CMD=(
  "$GLM53_BIN"
  --model "$GLM53_MODEL"
  --alias "${GLM53_ALIAS:-glm-5.3-flash}"
  --host "${GLM53_HOST:-0.0.0.0}" --port "$GLM53_PORT"
  --n-gpu-layers "${GLM53_NGL:-999}"
  --ctx-size "$GLM53_CTX"
  --parallel "${GLM53_PARALLEL:-1}"
  --cache-type-k "${GLM53_CACHE_TYPE_K:-q8_0}"
  --cache-type-v "${GLM53_CACHE_TYPE_V:-q8_0}"
  --flash-attn "${GLM53_FLASH_ATTN:-on}"
  --batch-size "${GLM53_BATCH:-2048}"
  --ubatch-size "${GLM53_UBATCH:-512}"
  --jinja
  --temp "${GLM53_TEMP:-1.0}"
  --top-p "${GLM53_TOP_P:-1.0}"
)

# MTP. Deliberately omitted entirely when 'none' so the nextn heads are not even
# loaded — that is what makes the A/B in mtp_sweep.sh a real comparison.
if [ "${GLM53_SPEC_TYPE:-none}" != "none" ] && [ -n "${GLM53_SPEC_TYPE:-}" ]; then
  CMD+=( --spec-type "$GLM53_SPEC_TYPE" )
  [ -n "${GLM53_SPEC_DRAFT_N_MAX:-}" ] && CMD+=( --spec-draft-n-max "$GLM53_SPEC_DRAFT_N_MAX" )
  [ -n "${GLM53_SPEC_DRAFT_N_MIN:-}" ] && CMD+=( --spec-draft-n-min "$GLM53_SPEC_DRAFT_N_MIN" )
fi

# Reasoning. Empty effort means "let the template decide", which for GLM-5.3 is
# Max — the right default for agentic work. See glm53.env.
[ -n "${GLM53_REASONING_EFFORT:-}" ] && CMD+=( --reasoning-effort "$GLM53_REASONING_EFFORT" )
[ -n "${GLM53_REASONING_FORMAT:-}" ] && CMD+=( --reasoning-format "$GLM53_REASONING_FORMAT" )
case "${GLM53_REASONING_PRESERVE:-on}" in
  on|1|true)   CMD+=( --reasoning-preserve ) ;;
  off|0|false) CMD+=( --no-reasoning-preserve ) ;;
  *) echo "glm53-up: GLM53_REASONING_PRESERVE must be on|off, got '${GLM53_REASONING_PRESERVE}'" >&2; exit 1 ;;
esac

# shellcheck disable=SC2206
[ -n "${GLM53_EXTRA:-}" ] && CMD+=( ${GLM53_EXTRA} )

echo "glm53-up: model=$(basename "$GLM53_MODEL")"
echo "glm53-up: ctx=$GLM53_CTX kv=${GLM53_CACHE_TYPE_K}/${GLM53_CACHE_TYPE_V} spec=${GLM53_SPEC_TYPE:-none} n_max=${GLM53_SPEC_DRAFT_N_MAX:-} port=$GLM53_PORT"
echo "glm53-up: effort=${GLM53_REASONING_EFFORT:-<template default: Max>} preserve=${GLM53_REASONING_PRESERVE:-on} temp=${GLM53_TEMP:-1.0} top_p=${GLM53_TOP_P:-1.0}"

if [ "${DRY_RUN:-0}" = "1" ]; then
  printf '%q ' "${CMD[@]}"; echo
  exit 0
fi

if [ "${GLM53_PREFLIGHT:-0}" = "1" ]; then
  "$DIR/preflight.sh"
fi

exec "${CMD[@]}"
