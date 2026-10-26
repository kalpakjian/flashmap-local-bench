#!/usr/bin/env bash
# =============================================================================
# FlashMap 本機實驗環境建置腳本 (Arch Linux / Omarchy)
#   目標：Qwen/Qwen3-8B + j-shen/FlashMap-Qwen3-8B，量測本機加速比
#   用法：bash setup.sh            # 只做 Python 環境
#         bash setup.sh --with-cuda # 額外安裝 CUDA Toolkit (需要 nvcc)
#   特性：冪等，重複執行安全
# =============================================================================
set -euo pipefail

WORKDIR="${WORKDIR:-$HOME/flashmap-lab}"
VENV="$WORKDIR/.venv"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-8B}"
DRAFT_MODEL="${DRAFT_MODEL:-j-shen/FlashMap-Qwen3-8B}"
export HF_HOME="${HF_HOME:-$WORKDIR/hf-cache}"
WITH_CUDA=0
[ "${1:-}" = "--with-cuda" ] && WITH_CUDA=1

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }

mkdir -p "$WORKDIR" "$HF_HOME"

# ---------------------------------------------------------------- 0. 環境檢查
log "0/6 環境檢查"
command -v pacman >/dev/null 2>&1 || die "找不到 pacman，這不是 Arch 系。請確認已從開機選單進入 Omarchy。"
echo "  OS      : $(grep -m1 PRETTY_NAME /etc/os-release | cut -d'"' -f2)"
echo "  Kernel  : $(uname -r)"
echo "  工作目錄: $WORKDIR"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  die "找不到 nvidia-smi。Omarchy 需要先裝 NVIDIA 驅動：\n       omarchy install nvidia && sudo mkinitcpio -P && reboot"
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || warn "nvidia-smi 執行失敗，可能需重開機載入模組"
DRV_CUDA=$(nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9.]*' | head -1 || true)
echo "  驅動    : $DRV_CUDA"

echo "  磁碟    : $(df -h "$WORKDIR" | awk 'NR==2{print $4" 可用 / "$2}')"
AVAIL_GB=$(df -BG "$WORKDIR" | awk 'NR==2{print $4}' | tr -dc '0-9')
[ "${AVAIL_GB:-0}" -ge 35 ] || warn "可用空間約 ${AVAIL_GB}GB，低於建議的 35GB（模型 17.5GB + venv 約 12GB + 快取）"

# ------------------------------------------------------- 1. 系統套件（選擇性）
if [ "$WITH_CUDA" = "1" ]; then
  log "1/6 安裝系統套件 + CUDA Toolkit（需要 sudo）"
  sudo pacman -S --needed --noconfirm base-devel curl
  # Arch 的 CUDA Toolkit 在 extra 倉庫；若無此套件則改用 nvidia 官方倉庫
  if ! sudo pacman -S --needed --noconfirm cuda-toolkit 2>/dev/null; then
    warn "extra 倉庫沒有 cuda-toolkit，改用 nvidia 官方倉庫（會加到 pacman.conf）"
    sudo pacman -Sy --needed --noconfirm archlinux-keyring
    KEY=$(curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/pacman.d/nvidia.gpg)
    sudo sh -c "echo '[nvidia]\nServer = https://us.download.nvidia.com/arch/x86_64\nSigLevel = Required DatabaseOptional' > /etc/pacman.d/nvidia.conf"
    sudo pacman -Sy --noconfirm
    sudo pacman -S --needed --noconfirm cuda-toolkit
  fi
  nvcc --version || warn "nvcc 仍不可用；可略過，vLLM 會用 VLLM_USE_FLASHINFER_SAMPLER=0 退回 PyTorch sampler"
else
  log "1/6 略過系統套件（需要時用：bash setup.sh --with-cuda）"
fi

# ---------------------------------------------------------------- 2. 安裝 uv
log "2/6 準備 uv (Python 環境管理)"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

# ------------------------------------------------------- 3. 建立 venv + 安裝套件
log "3/6 建立 Python 3.12 虛擬環境並安裝套件"
# vLLM 0.31.0 與 SGLang 0.5.20 都鎖 torch==2.13.0；flashmap 0.1.0 額外要求 >=3.10
# flashmap[vllm] 的約束是 vllm>=0.31,<0.32 -> 目前只有 0.31.0
uv venv --python 3.12 "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
uv pip install --upgrade pip
uv pip install "vllm==0.31.0" "flashmap[vllm]" "huggingface_hub"

log "   已安裝版本"
python - <<'PY'
import torch, vllm, transformers, flashmap
print(f"   torch        {torch.__version__}")
print(f"   vllm         {vllm.__version__}")
print(f"   transformers {transformers.__version__}")
print(f"   flashmap     {getattr(flashmap, '__version__', 'n/a')}")
print(f"   cuda avail   {torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"   gpu          {p.name}  {p.total_memory/1024**3:.1f} GB  sm_{p.major}{p.minor}")
PY

# ---------------------------------------------------------------- 4. CUDA smoke
log "4/6 CUDA smoke test"
python - <<'PY'
import torch, sys
if not torch.cuda.is_available():
    sys.exit("  [fail] torch 看不到 GPU")
a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
b = a @ a
torch.cuda.synchronize()
print(f"  [ok] bf16 matmul on {torch.cuda.get_device_name(0)} -> {tuple(b.shape)}")
PY

# ---------------------------------------------------------------- 5. 下載模型
log "5/6 下載模型（首次約 17.5 GB）"
python - <<PY
from huggingface_hub import snapshot_download
for repo in ["$TARGET_MODEL", "$DRAFT_MODEL"]:
    print(f"  -> {repo}", flush=True)
    p = snapshot_download(repo)
    print(f"     {p}")
PY
du -sh "$HF_HOME" 2>/dev/null || true

# ---------------------------------------------------------------- 6. 總結
log "6/6 完成"
cat <<EOF

環境已就緒。下一步：
  cd $WORKDIR
  source .venv/bin/activate
  # 執行完整實驗（接受長度 + 無損性 + 端到端 tok/s）
  python $WORKDIR/bench.py
  # 嚴格「無損性」檢查請用 fp32（bf16 的近似並列會造成假性 MISMATCH）
  python $WORKDIR/acceptance.py --dtype float32

或在原始碼目錄跑一鍵腳本：bash run_all.sh

備註：
  * FlashInfer 需要 nvcc 才能 JIT 編譯。若啟動報 nvcc 錯誤，改用 PyTorch sampler：
      export VLLM_USE_FLASHINFER_SAMPLER=0
  * 模型快取位置：$HF_HOME
EOF
