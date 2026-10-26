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

# ---------------------------------------------------------------- 選 Python
# 這支腳本在 README 的順序裡排在 setup.sh 之前，那時 $WORKDIR/.venv 還不存在，
# 所以不能假設 `python` 就是裝了 huggingface_hub 的那一個（系統 Python 通常沒有）。
# 依序找：WORKDIR 的 venv → python3 → python，並實際 import 一次確認。
pick_python() {
  local cand
  for cand in "$WORKDIR/.venv/bin/python" "$(command -v python3 || true)" \
              "$(command -v python || true)"; do
    [ -n "$cand" ] && [ -x "$cand" ] || continue
    if "$cand" -c 'import huggingface_hub' >/dev/null 2>&1; then
      echo "$cand"
      return 0
    fi
  done
  return 1
}

# ---------------------------------------------------------------- 複製後檢查
# Linux 的 HF 快取是 symlink 農場：snapshots/<sha>/config.json -> ../../blobs/<hash>。
# 用 `cp -r` 會把這些連結原樣搬過去，於是目標目錄全部指向不存在的 blobs/
# （看起來只有幾十 KB、載入模型時才炸）。所以複製一律加 -L 解參照，
# 搬完再用 -xtype l 確認沒有殘留的斷鏈。
# Windows 端因為沒有 symlink 權限、快取本來就是實體檔案，加 -L 也無害。
verify_model_dir() {
  local d="$1" bad
  [ -f "$d/config.json" ] || { warn "缺少 config.json：$d"; return 1; }
  bad="$(find "$d" -maxdepth 1 -xtype l 2>/dev/null | wc -l)"
  if [ "$bad" -gt 0 ]; then
    warn "$d 有 $bad 個斷掉的 symlink（複製時沒有解參照？）"
    return 1
  fi
  return 0
}

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
  # 先清掉舊目錄：若殘留上一次複製的斷鏈，cp -L 會拒絕寫入
  rm -rf "$MODELS/$name"
  mkdir -p "$MODELS/$name"
  # 從 NTFS 複製大檔到 ext4 會比較慢，進度可見
  cp -rL "$src/$snap/." "$MODELS/$name/"
  verify_model_dir "$MODELS/$name" || return 1
  local sz
  sz="$(du -sh "$MODELS/$name" | cut -f1)"
  echo "  完成：$MODELS/$name ($sz)"
  return 0
}

if [ "$MODE" = "download" ] || { [ "$MODE" = "auto" ] && [ -z "$WIN_HUB" ]; }; then
  log "從 Hugging Face 下載模型（首次約 17.5 GB）"
  PY="$(pick_python)" || die "找不到裝了 huggingface_hub 的 Python（系統 Python 通常沒有）。
  這支腳本排在 setup.sh 之前，那時 venv 還沒建好。可以改成：
     bash setup.sh && bash prepare_models.sh --download   # 先建環境（setup.sh 也會下載模型）
  或一次性環境：
     uv run --with huggingface_hub python $HERE/download_models.py"
  echo "  python：$PY"
  "$PY" "$HERE/download_models.py" || die "下載失敗"
  # 下載到快取後，複製一份成一般模型目錄，方便直接指定路徑
  for repo in "Qwen/Qwen3-8B" "j-shen/FlashMap-Qwen3-8B"; do
    name="$(basename "$repo")"
    case "$repo" in
      j-shen/*) name="FlashMap-${name#FlashMap-}" ;;   # 已經有前綴就不要重複加
    esac
    src="$HF_HOME/hub/$(cache_dir "$repo")/snapshots"
    snap="$(ls -1 "$src" 2>/dev/null | head -1)"
    if [ -n "$snap" ]; then
      rm -rf "$MODELS/$name"          # 同上：清掉可能殘留的斷鏈
      mkdir -p "$MODELS/$name"
      cp -rL "$src/$snap/." "$MODELS/$name/" || die "複製 $name 失敗"
      verify_model_dir "$MODELS/$name" || die "$name 複製後目錄不完整（見上方訊息）"
    fi
  done
else
  [ -n "$WIN_HUB" ] || die "找不到 Windows 分割區的 HF 快取（/mnt/c/Users/*/flashmap-lab/hf-cache/hub）。
  要下載請改用 --download（需先 bash setup.sh 建好 venv），或用 WIN_HF_HUB=/path/to/hub 指定位置。"
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
  * 本機目錄是「解參照後的真實檔案」（cp -rL），約 17 GB，與 hf-cache 各一份；
    這樣不依賴 HF 快取的 symlink 結構，清掉快取也不會壞。
  * drafter 目錄含 configuration_flashmap.py / modeling_flashmap.py，
    加上已安裝 flashmap 套件（會註冊 flashmap model type），trust_remote_code 兩種情況都能載入。
EOF
