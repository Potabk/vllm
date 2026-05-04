#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Minimal 2-GPU validation for the EAGLE3 + NixlConnector fix.
#
# Reproduces the bug: mixing FLASHINFER (main model) with FLASH_ATTN (EAGLE3
# draft model) at block_size=128 causes select_common_block_size to return 64
# instead of 128, inflating _physical_blocks_per_logical_kv_block and making
# self.num_blocks != draft tensor shape[0].
#
# Usage (2 GPUs required):
#   CUDA_VISIBLE_DEVICES=0,1 \
#   MODEL=meta-llama/Llama-3.1-8B-Instruct \
#   DRAFTER=RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3 \
#   bash validate_eagle3_fix.sh
#
# Expected behaviour WITH the fix:
#   Both prefill and decode servers start successfully.
#   A simple completion request returns a valid response.
#
# Without the fix the prefill server crashes at startup with:
#   AssertionError: All kv cache tensors must have the same number of blocks

set -euo pipefail

MODEL="${MODEL:-LLM-Research/Meta-Llama-3.1-8B-Instruct}"
DRAFTER="${DRAFTER:-RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3}"

# block_size=128 with FLASHINFER triggers the mixed-backend mismatch:
#   FLASHINFER (main) supports [16,32,64] → common=64 with FA draft → phys_ratio=2
#   FLASH_ATTN draft alone supports MultipleOf(16) → 128 → phys_ratio=1  ← mismatch
BLOCK_SIZE=128
ATTENTION_BACKEND="FLASHINFER"

# P instance: 1 speculative token (prefill just verifies & warms KV)
PREFILL_SPEC_CONFIG="{\"method\":\"eagle3\",\"model\":\"${DRAFTER}\",\"num_speculative_tokens\":1}"
# D instance: 3 speculative tokens (normal decode use-case)
DECODE_SPEC_CONFIG="{\"method\":\"eagle3\",\"model\":\"${DRAFTER}\",\"num_speculative_tokens\":3}"

KV_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_both"}'

PREFILL_PORT=8100
DECODE_PORT=8200
SIDE_CHANNEL_P=5559
SIDE_CHANNEL_D=5659

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd -P)"

cleanup() {
  echo "Cleaning up..."
  kill "$(jobs -pr)" 2>/dev/null || true
  sleep 1
  kill -9 "$(jobs -pr)" 2>/dev/null || true
}
trap cleanup EXIT
trap 'echo "Interrupted."; exit 130' INT TERM

wait_for_server() {
  local port=$1 deadline=300 elapsed=0
  echo "Waiting for server on port ${port}..."
  while [ $elapsed -lt $deadline ]; do
    if curl -sf "http://localhost:${port}/health" >/dev/null 2>&1; then
      echo "  → port ${port} ready"
      return 0
    fi
    sleep 3; elapsed=$((elapsed + 3))
  done
  echo "FAIL: port ${port} did not become ready within ${deadline}s"
  exit 1
}

# ── GPU allocation ────────────────────────────────────────────────────────

IFS=',' read -ra ALL_GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
if [[ ${#ALL_GPUS[@]} -lt 2 ]]; then
  echo "FAIL: need 2 GPUs; got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  exit 1
fi
GPU_P="${ALL_GPUS[0]}"
GPU_D="${ALL_GPUS[1]}"

echo "========================================================"
echo "EAGLE3 + NixlConnector 2-GPU fix validation"
echo "  Model:             ${MODEL}"
echo "  Drafter:           ${DRAFTER}"
echo "  block_size:        ${BLOCK_SIZE}  (key: triggers mixed-backend mismatch)"
echo "  Attention backend: ${ATTENTION_BACKEND}  (FLASHINFER main + FA draft)"
echo "  Prefill GPU:       ${GPU_P}  port ${PREFILL_PORT}"
echo "  Decode  GPU:       ${GPU_D}  port ${DECODE_PORT}"
echo "========================================================"

# ── Prefill (P) ───────────────────────────────────────────────────────────

echo ""
echo "[P] Starting prefill instance..."
CUDA_VISIBLE_DEVICES=$GPU_P \
VLLM_KV_CACHE_LAYOUT=HND \
VLLM_NIXL_SIDE_CHANNEL_PORT=$SIDE_CHANNEL_P \
UCX_NET_DEVICES=all \
vllm serve "$MODEL" \
  --port $PREFILL_PORT \
  --enforce-eager \
  --block-size $BLOCK_SIZE \
  --gpu-memory-utilization 0.7 \
  --tensor-parallel-size 1 \
  --kv-transfer-config "$KV_CONFIG" \
  --speculative-config "$PREFILL_SPEC_CONFIG" \
  --attention-backend $ATTENTION_BACKEND \
  --max-model-len 4096 &

# ── Decode (D) ────────────────────────────────────────────────────────────

echo "[D] Starting decode instance..."
CUDA_VISIBLE_DEVICES=$GPU_D \
VLLM_KV_CACHE_LAYOUT=HND \
VLLM_NIXL_SIDE_CHANNEL_PORT=$SIDE_CHANNEL_D \
UCX_NET_DEVICES=all \
vllm serve "$MODEL" \
  --port $DECODE_PORT \
  --enforce-eager \
  --block-size $BLOCK_SIZE \
  --gpu-memory-utilization 0.7 \
  --tensor-parallel-size 1 \
  --kv-transfer-config "$KV_CONFIG" \
  --speculative-config "$DECODE_SPEC_CONFIG" \
  --attention-backend $ATTENTION_BACKEND \
  --max-model-len 4096 &

wait_for_server $PREFILL_PORT
wait_for_server $DECODE_PORT

# ── Proxy ─────────────────────────────────────────────────────────────────

echo ""
echo "Starting toy proxy server..."
python3 "${REPO_ROOT}/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py" \
  --port 8192 \
  --prefiller-hosts localhost \
  --prefiller-ports $PREFILL_PORT \
  --decoder-hosts localhost \
  --decoder-ports $DECODE_PORT &

sleep 5

# ── Smoke test ────────────────────────────────────────────────────────────

echo ""
echo "Sending smoke-test completion request through proxy..."
RESPONSE=$(curl -sf http://localhost:8192/v1/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL}\",
    \"prompt\": \"The capital of France is\",
    \"max_tokens\": 10,
    \"temperature\": 0
  }")

echo "Response: ${RESPONSE}"
# Basic sanity: response must contain non-empty text
echo "$RESPONSE" | python3 -c "
import sys, json
data = json.load(sys.stdin)
text = data['choices'][0]['text']
assert text.strip(), 'Empty completion text'
print(f'Completion text: {text!r}')
print('PASS: completion returned non-empty text')
"

echo ""
echo "========================================================"
echo "PASS: EAGLE3 + NixlConnector started and responded OK"
echo "      (block_size=${BLOCK_SIZE}, backend=${ATTENTION_BACKEND})"
echo "========================================================"
