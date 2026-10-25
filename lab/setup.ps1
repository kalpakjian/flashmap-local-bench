# =============================================================================
# FlashMap 本機實驗 - Windows 版環境建置（Phase 1：接受長度 + 無損性）
#
#   為什麼只有 Phase 1？
#     vllm 0.31.0 / sglang 0.5.20 在 PyPI 只有 manylinux wheel，沒有 win_amd64，
#     所以「端到端 tok/s」（bench.py）**無法在 Windows 原生執行**，必須走 Linux。
#     但 flashmap 套件的核心 modeling_flashmap.py 是純 PyTorch（無 triton/flash-attn），
#     因此 Phase 1（acceptance.py）在 Windows 上完全可跑。
#
#   用法：
#     powershell -ExecutionPolicy Bypass -File setup.ps1
#     powershell -ExecutionPolicy Bypass -File setup.ps1 -SkipModels   # 只建環境，模型另外下
# =============================================================================
param(
    [switch]$SkipModels,
    # 一併下載 Qwen3-4B + FlashMap-Qwen3-4B，供「嚴格無損性檢查」(float32) 使用。
    # 8B 的 fp32 權重是 32 GiB，超過 24 GiB 的 RTX 4090，放不下 → 只能用 4B 做 dtype A/B。
    [switch]$With4B,
    # Windows 上 PyPI 的 torch 是 CPU-only，CUDA 版需從 PyTorch 自己的 index 裝。
    # cu130 需要驅動 >= 580；驅動較舊可改成 cu128。
    [string]$TorchIndex = 'https://download.pytorch.org/whl/cu130'
)
$ErrorActionPreference = 'Continue'   # 原生工具（uv/pip/huggingface_hub）會把進度寫到 stderr，
                                       # 在 PS 5.1 的 Stop 模式下會被誤判為錯誤而中止腳本。
                                       # 真正的失敗一律用 $LASTEXITCODE 檢查（見下方 Die 呼叫）。
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

$WorkDir   = if ($env:WORKDIR) { $env:WORKDIR } else { Join-Path $HOME 'flashmap-lab' }
$Venv      = Join-Path $WorkDir '.venv'
$VenvPy    = Join-Path $Venv 'Scripts\python.exe'
$TargetModel = 'Qwen/Qwen3-8B'
$DraftModel  = 'j-shen/FlashMap-Qwen3-8B'
if (-not $env:HF_HOME) { $env:HF_HOME = Join-Path $WorkDir 'hf-cache' }

function Log($m)  { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Warn($m) { Write-Host "[warn] $m" -ForegroundColor Yellow }
function Die($m)  { Write-Host "[fail] $m" -ForegroundColor Red; exit 1 }

New-Item -ItemType Directory -Force -Path $WorkDir, $env:HF_HOME | Out-Null

# ------------------------------------------------------------------ 0. 檢查
Log '0/4 環境檢查'
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    Die '找不到 nvidia-smi，請先安裝 NVIDIA 驅動（>= CUDA 13 支援版本）'
}
$nvidia = nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
Write-Host "  GPU : $nvidia"
$freeGB = [math]::Floor((Get-PSDrive -Name ((Get-Location).Drive.Name)).Free / 1GB)
Write-Host "  磁碟: $freeGB GB 可用"
if ($freeGB -lt 35) { Warn "可用空間 $freeGB GB 低於建議的 35GB（模型 17.5GB + 套件約 8GB）" }

# ------------------------------------------------------------------ 1. uv
Log '1/4 準備 uv'
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host '  安裝 uv ...'
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $uvPath = Join-Path $HOME '.local\bin'
    if (Test-Path $uvPath) { $env:PATH = "$uvPath;$env:PATH" }
}
$uv = (Get-Command uv -ErrorAction SilentlyContinue)
if (-not $uv) { Die 'uv 安裝失敗，請手動安裝：https://docs.astral.sh/uv/getting-started/installation/' }
uv --version

# ------------------------------------------------------------------ 2. venv
Log '2/4 建立 Python 3.12 虛擬環境'
if (-not (Test-Path $VenvPy)) {
    uv venv --python 3.12 $Venv
}
Write-Host "  venv: $Venv"

# 注意：這裡刻意不安裝 vllm（無 Windows wheel），因此也不使用 flashmap[vllm] extra
Log '3/4 安裝套件（torch / transformers / flashmap，不含 vLLM）'
# 先移除可能存在的 CPU-only torch，否則 uv 會認為需求已滿足而沿用舊版
uv pip uninstall --python $VenvPy torch | Out-Null
uv pip install --python $VenvPy torch --index-url $TorchIndex
if ($LASTEXITCODE -ne 0) { Die "torch 安裝失敗（index: $TorchIndex）" }
uv pip install --python $VenvPy 'transformers>=4.56' 'flashmap' 'huggingface_hub' 'datasets'
if ($LASTEXITCODE -ne 0) { Die 'transformers / flashmap 安裝失敗' }

& $VenvPy (Join-Path $PSScriptRoot 'env_check.py')
if ($LASTEXITCODE -ne 0) { Die '環境檢查失敗（詳見上方訊息）' }

# ------------------------------------------------------------------ 4. 模型
if ($SkipModels) {
    Warn '依參略過模型下載'
} else {
    Log '4/4 下載模型（首次約 17.5 GB）'
    $dlArgs = @((Join-Path $PSScriptRoot 'download_models.py'))
    if ($With4B) { $dlArgs += '--with-4b' }
    & $VenvPy @dlArgs
    if ($LASTEXITCODE -ne 0) { Die '模型下載失敗' }
}

Log '完成'
Write-Host @"

Windows 上可執行 Phase 1：
  & $VenvPy acceptance.py --prompts prompts.json --max-new-tokens 512

Phase 2（端到端 tok/s）需要 Linux，兩種作法：
  A. WSL2：powershell -ExecutionPolicy Bypass -File setup_wsl2.ps1
  B. 重開機進 Omarchy，直接跑 bash setup.sh && bash run_all.sh

模型快取：$env:HF_HOME
"@
