# =============================================================================
# FlashMap 本機實驗 - WSL2 環境檢查 / 安裝
#
#   目的：讓「端到端 tok/s」（Phase 2，需要 vLLM）不必重開機進 Omarchy 也能跑。
#   原理：vLLM 官方僅支援 Linux；WSL2 + NVIDIA GPU passthrough 可執行 manylinux wheel。
#
#   用法：powershell -ExecutionPolicy Bypass -File setup_wsl2.ps1
# =============================================================================
$ErrorActionPreference = 'Continue'

function Log($m)  { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Warn($m) { Write-Host "[warn] $m" -ForegroundColor Yellow }

Log '1/4 檢查 WSL 狀態'
$status = (& wsl --status 2>&1 | Out-String)
Write-Host $status

$distros = (& wsl -l -v 2>&1 | Out-String)
Write-Host '已安裝的發行版：'
Write-Host $distros

if ($distros -match 'NAME\s+STATE' -and -not ($distros -match '\S+\s+(Running|Stopped)\s+\d')) {
    Warn '沒有安裝任何 Linux 發行版'
    Write-Host @'

安裝方式（需要重開機一次）：
  wsl --install -d Ubuntu-24.04

重開機後再執行本腳本，確認 GPU passthrough，然後在 WSL 內跑完整流程。

在 WSL 內執行實驗（repo 放在 Windows 磁碟可直接從 /mnt/c 取得）：
  cd /mnt/c/<你的帳號>/flashmap-local-bench/lab
  bash setup.sh
  bash run_all.sh

注意：Windows 與 WSL 共用同一張 GPU。跑實驗前請先關閉 Windows 上的
      遊戲/其他推論程式，否則 24GB VRAM 會不夠（Qwen3-8B 16.4GB + drafter 1GB）。
'@
    exit 0
}

Log '2/4 檢查 WSL 內的 NVIDIA GPU passthrough'
$gpu = (& wsl -e bash -lc 'nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader' 2>&1 | Out-String)
Write-Host $gpu
if ($LASTEXITCODE -ne 0 -or $gpu -match 'not found|No devices') {
    Warn 'WSL 內看不到 GPU，請確認 Windows 側 NVIDIA 驅動 >= 525 且已安裝 WSL'
    Write-Host '  驅動版本：' (& nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | Out-String)
} else {
    Write-Host '  [ok] GPU passthrough 正常' -ForegroundColor Green
}

Log '3/4 檢查 WSL 內的 Python 與磁碟'
& wsl -e bash -lc 'python3 --version; echo "---"; df -h ~ | tail -1; echo "---"; ls /mnt/c 2>/dev/null | head -3'

Log '4/4 後續步驟'
Write-Host @'
在 WSL 內執行完整實驗：

  cd /mnt/c/<你的帳號>/flashmap-local-bench/lab
  bash setup.sh       # 建 venv（uv + py3.12）、裝 vllm==0.31.0 + flashmap[vllm]、下載模型
  bash run_all.sh     # Phase 1 + Phase 2

若 /mnt/c 下讀寫太慢，可把整個 repo 複製到 WSL 的 ext4 檔案系統：
  cp -r /mnt/c/<你的帳號>/flashmap-local-bench ~/
  cd ~/flashmap-local-bench/lab

建議把模型快取也放進 WSL（ext4 速度遠快於 /mnt/c）：
  export HF_HOME=~/flashmap-lab/hf-cache
'@
