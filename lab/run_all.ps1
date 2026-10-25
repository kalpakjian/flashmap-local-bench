# =============================================================================
# FlashMap 本機實驗 - Windows 版執行器
#
#   Phase 1（接受長度 + 無損性）：Windows 原生可跑
#   Phase 2（端到端 tok/s）        ：需要 vLLM -> 僅限 Linux
#                                     會自動偵測；若目前就在 WSL2/Linux 環境則一併執行
#
#   用法：powershell -ExecutionPolicy Bypass -File run_all.ps1
# =============================================================================
param(
    [int]$MaxTokens = 512,
    [switch]$SkipPhase1,
    [switch]$SkipPhase2
)
$ErrorActionPreference = 'Continue'   # 見 setup.ps1 說明：原生工具的 stderr 進度輸出不可視為錯誤

$Here    = Split-Path -Parent $MyInvocation.MyCommand.Path
$WorkDir = if ($env:WORKDIR) { $env:WORKDIR } else { Join-Path $HOME 'flashmap-lab' }
$VenvPy  = Join-Path $WorkDir '.venv\Scripts\python.exe'
$Py      = if (Test-Path $VenvPy) { $VenvPy } else { 'python' }

function Log($m) { Write-Host "`n==> $m" -ForegroundColor Cyan }

Set-Location $Here
if (-not $env:HF_HOME) { $env:HF_HOME = Join-Path $WorkDir 'hf-cache' }

# ---------------------------------------------------------------- Phase 1
if (-not $SkipPhase1) {
    Log 'Phase 1 - 接受長度 γ + 無損性驗證'
    & $Py acceptance.py --prompts prompts.json --max-new-tokens $MaxTokens --num-steps 1
    if ($LASTEXITCODE -ne 0) {
        Write-Host '[warn] Phase 1 有項目未通過（退出碼非 0），請看上面的 lossless 標記' -ForegroundColor Yellow
    }
}

# ---------------------------------------------------------------- Phase 2
Log '偵測是否可以執行 Phase 2（端到端 tok/s）'
$vllmOK = $false
try {
    & $Py -c 'import vllm' 2>$null
    if ($LASTEXITCODE -eq 0) { $vllmOK = $true }
} catch { }

if ($SkipPhase2) {
    Write-Host '  依參數跳過 Phase 2'
} elseif ($vllmOK) {
    Write-Host '  偵測到 vLLM，執行 Phase 2 ...' -ForegroundColor Green
    & $Py bench.py --prompts prompts.json --max-tokens $MaxTokens
} else {
    Write-Host @'
  [skip] 目前環境沒有 vLLM，Phase 2 需要 Linux。

  兩種作法：
    A. WSL2（不需重開機進入別的 OS）
       powershell -ExecutionPolicy Bypass -File setup_wsl2.ps1
    B. 重開機進 Omarchy，在 Linux 內跑完整流程
       bash setup.sh && bash run_all.sh

  補充：vllm 0.31.0 / sglang 0.5.20 在 PyPI 只有 manylinux wheel，
        沒有 win_amd64，因此 Windows 原生無法執行 Phase 2。
'@ -ForegroundColor Yellow
}

Write-Host "`n產出：" -ForegroundColor Cyan
Get-ChildItem -Path $Here -Filter 'results_*.json' -ErrorAction SilentlyContinue |
    ForEach-Object { Write-Host "  $($_.Name)" }
if (Test-Path (Join-Path $Here 'report.md')) { Write-Host '  report.md' }
