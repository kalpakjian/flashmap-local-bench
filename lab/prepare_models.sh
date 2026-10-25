#!/usr/bin/env bash
# =============================================================================
# 準備本機模型目錄（Linux / Omarchy）
#
#   為什麼需要這支？
#     Windows 上 HF 快取因缺 symlink 權限，權重是直接放在
#     hf-cache/hub/<repo>/snapshots/<sha>/ 的「實體副本」，blobs/ 是空的。
#     所以不能直接當 HF 快取搬到 Linux 給 huggingface_hub 辨識，
#     但那個 snapshot 目錄本身就是完整可用的模型目錄 → 直接複製過去即可。
#
#   用法：
#     bash prepare_models.sh --from-windows   # 從 /mnt/c 的 Windows 快取搬（省 16GB 下載）
#     bash prepare_models.sh --download       # 直接從 Hugging Face 下載
#     bash prepare_models.sh                  # 自動判斷（有 Windows 快取就搬，否則下載）
#
#   產出：
#     $WORKDIR/models/Qwen3-8B/
#     $WORKDIR/models/FlashMap-Qwen3-8B/
#     $WORKDIR/models/env.sh   → source 這個檔案即可設定 TARGET_MODEL / DRAFT_MODEL
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${WORKDIR:-$HOME/flashmap-lab}"
MODELS="$WORKDIR/models"
export HF_HOME="${HF_HOME:-$WORKDIR/hf-cache}"

MODE="auto"
case "${1:-}" in
  --from-windows) MODE="windows" ;;
  --download)     MODE="download" ;;
  "")             MODE="auto" ;;
  *) echo "用法：bash prepare_models.sh [--from-windows|--download]" >&2; exit 2 ;;
esac

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }

# repo 名 -> 快取目錄名 / 本機模型目錄名
cache_dir() { echo "models--${1//\//--}"; }

# ---------------------------------------------------------------- 找 Windows 快取
WIN_HUB=""
for d in /mnt/c/Users/*/flashmap-lab/hf-cache/hub; do
  if [ -d "$d" ]; then WIN_HUB="$d"; break; fi
done
# 也允許用環境變數指定
if [ -z "$WIN_HUB" ] && [ -n "${WIN_HF_HUB:-}" ] && [ -d "${WIN_HF_HUB}" ]; then
  WIN_HUB="$WIN_HF_HUB"
fi

mkdir -p "$MODELS"

copy_model() {
  # $1 = repo id, $2 = 本機目錄名
  local repo="$1" name="$2"
  local src="$WIN_HUB/$(cache_dir "$repo")/snapshots"
  [ -d "$src" ] || { warn "找不到 $src"; return 1; }
  local snap
  snap="$(ls -1 "$src" 2>/dev/null | head -1)"
  [ -n "$snap" ] || { warn "$src 內沒有 snapshot"; return 1; }
  local n
  n="$(ls -1 "$src/$snap" | wc -l)"
  echo "  來源：$src/$snap（$n 個檔案）"
  mkdir -p "$MODELS/$name"
  # 從 NTFS 複製大檔到 ext4 會比較慢，進度可見
  cp -r "$src/$snap/." "$MODELS/$name/"
  [ -f "$MODELS/$name/config.json" ] || { warn "複製後缺少 config.json"; return 1; }
  local sz
  sz="$(du -sh "$MODELS/$name" | cut -f1)"
  echo "  完成：$MODELS/$name ($sz)"
  return 0
}

if [ "$MODE" = "download" ] || { [ "$MODE" = "auto" ] && [ -z "$WIN_HUB" ]; }; then
  log "從 Hugging Face 下載模型（首次約 17.5 GB）"
  python "$HERE/download_models.py" || die "下載失敗"
  # 下載到快取後，複製一份成一般模型目錄，方便直接指定路徑
  for repo in "Qwen/Qwen3-8B" "j-shen/FlashMap-Qwen3-8B"; do
    name="$(basename "$repo")"
    case "$repo" in
      j-shen/*) name="FlashMap-$(basename "$repo")" ;;
    esac
    src="$HF_HOME/hub/$(cache_dir "$repo")/snapshots"
    snap="$(ls -1 "$src" 2>/dev/null | head -1)"
    if [ -n "$snap" ]; then
      mkdir -p "$MODELS/$name"
      cp -r "$src/$snap/." "$MODELS/$name/" 2>/dev/null || true
    fi
  done
else
  [ -n "$WIN_HUB" ] || die "找不到 Windows 分割區的 HF 快取（/mnt/c/Users/*/flashmap-lab/hf-cache/hub）；改用 --download"
  log "從 Windows 快取搬模型（省下 ~16 GB 下載）"
  echo "  快取位置：$WIN_HUB"
  copy_model "Qwen/Qwen3-8B" "Qwen3-8B" || die "Qwen3-8B 搬運失敗"
  copy_model "j-shen/FlashMap-Qwen3-8B" "FlashMap-Qwen3-8B" || die "FlashMap drafter 搬運失敗"
fi

# ---------------------------------------------------------------- 產出 env.sh
cat > "$MODELS/env.sh" <<EOF
# 由 prepare_models.sh 產生：source 這個檔案即可使用本機模型
export TARGET_MODEL="$MODELS/Qwen3-8B"
export DRAFT_MODEL="$MODELS/FlashMap-Qwen3-8B"
export HF_HOME="$HF_HOME"
EOF

log "完成"
cat <<EOF
模型目錄：
  $MODELS/Qwen3-8B        ($(du -sh "$MODELS/Qwen3-8B" 2>/dev/null | cut -f1))
  $MODELS/FlashMap-Qwen3-8B ($(du -sh "$MODELS/FlashMap-Qwen3-8B" 2>/dev/null | cut -f1))

使用方式：
  source $MODELS/env.sh
  python acceptance.py --dataset gsm8k --num-prompts 80 --max-new-tokens 2048

注意：
  * 本機目錄不含 symlink 結構，但模型檔本身完整，可直接用 AutoModel 載入。
  * drafter 目錄含 configuration_flashmap.py / modeling_flashmap.py，
    加上已安裝 flashmap 套件（會註冊 flashmap model type），trust_remote_code 兩種情況都能載入。
EOF
