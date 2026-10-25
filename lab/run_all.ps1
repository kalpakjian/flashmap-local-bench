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
    [switch]$SkipPhase2,
    # 嚴格「無損性」檢查：改用 Qwen3-4B + float32。
    # 8B 的 fp32 權重是 32 GiB，超過 24 GiB 的 RTX 4090，放不下。
    [switch]$Strict
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
    Log 'Phase 1a - 接受長度 γ（bf16）'
    & $Py acceptance.py --prompts prompts.json --max-new-tokens $MaxTokens --num-steps 1 --dtype bfloat16
    if ($LASTEXITCODE -ne 0) {
        Write-Host @'
[note] bf16 下出現 MISMATCH 是「預期中的數值假象」，不是演算法錯誤。
       推測解碼在 temperature=0 下是無損的（每個 token 都等於 target 的 argmax）；
       但 bf16 只有 8 個 mantissa bit，top-2 logit 差 < 1 個 ULP 時兩者會「精確並列」，
       不同 kernel 的 tie-break 方向可能相反，於是整段文字分岔。
       嚴格比對請用 --dtype float32（見下方 Phase 1b 或 FINDINGS.md）。
'@ -ForegroundColor Yellow
    }
}

if ($Strict) {
    Log 'Phase 1b - 嚴格無損性檢查（Qwen3-4B + float32，預期 12/12 逐 token 相同）'
    $env:TARGET_MODEL = 'Qwen/Qwen3-4B'
    $env:DRAFT_MODEL  = 'j-shen/FlashMap-Qwen3-4B'
    & $Py acceptance.py --prompts prompts.json --max-new-tokens $MaxTokens --num-steps 1 --dtype float32
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
