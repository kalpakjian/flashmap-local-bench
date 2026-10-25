#!/usr/bin/env bash
# 一鍵執行：接受長度/無損性 -> 端到端 tok/s -> 產生報告
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

WORKDIR="${WORKDIR:-$HOME/flashmap-lab}"
VENV="$WORKDIR/.venv"
export HF_HOME="${HF_HOME:-$WORKDIR/hf-cache}"
# 沒有 CUDA Toolkit (nvcc) 時，FlashInfer 無法 JIT 編譯 sampling kernel
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

if [ ! -x "$VENV/bin/python" ]; then
  echo "[fail] 找不到 venv：$VENV，請先執行 bash setup.sh" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

MAX_TOKENS="${MAX_TOKENS:-512}"

echo
echo "############################################################"
echo "# Phase 1 — 接受長度 γ + 無損性驗證（transformers 參考實作）"
echo "############################################################"
# 即使無損性檢查失敗也要繼續跑 Phase 2，故不讓 set -e 中斷
python acceptance.py --prompts prompts.json --max-new-tokens "$MAX_TOKENS" --num-steps 1 \
  || echo "[warn] Phase 1 未全部通過（退出碼非 0），請看上面的 lossless 標記"

echo
echo "############################################################"
echo "# Phase 2 — 端到端 tok/s（vLLM：基準 vs FlashMap）"
echo "############################################################"
python bench.py --prompts prompts.json --max-tokens "$MAX_TOKENS"

echo
echo "[done] 產出："
ls -la results_acceptance_step1.json results_vllm.json report.md 2>/dev/null || true
