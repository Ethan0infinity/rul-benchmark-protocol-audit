param(
    [int]$WaitForPid = 0
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$Log = Join-Path $ProjectRoot "reports\review_revision_queue.log"

function Invoke-CheckedPython {
    param([string[]]$Arguments)
    "[QUEUE START] $Python $($Arguments -join ' ')" | Tee-Object -FilePath $Log -Append
    & $Python @Arguments 2>&1 | Tee-Object -FilePath $Log -Append
    if ($LASTEXITCODE -ne 0) {
        throw "Queued command failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
    "[QUEUE PASS] $($Arguments -join ' ')" | Tee-Object -FilePath $Log -Append
}

Set-Location -LiteralPath $ProjectRoot
if ($WaitForPid -gt 0) {
    "[QUEUE WAIT] pid=$WaitForPid" | Tee-Object -FilePath $Log -Append
    Wait-Process -Id $WaitForPid -ErrorAction SilentlyContinue
}

Invoke-CheckedPython @("scripts\run_design_sensitivity.py")
Invoke-CheckedPython @(
    "scripts\run_ncmapss_benchmark.py",
    "--cache", "data\external\processed\ncmapss_ds02_smp100_win50_val2.npz",
    "--models", "rast_gru", "rast_gru_v2",
    "--experiment-prefix", "ncmapss_ds02_val2",
    "--output-prefix", "ncmapss_ds02_val2"
)
Invoke-CheckedPython @(
    "scripts\run_ncmapss_benchmark.py",
    "--cache", "data\external\processed\ncmapss_ds02_smp100_win50_val10.npz",
    "--models", "rast_gru", "rast_gru_v2",
    "--experiment-prefix", "ncmapss_ds02_val10",
    "--output-prefix", "ncmapss_ds02_val10"
)
Invoke-CheckedPython @("scripts\analyze_ncmapss_split_sensitivity.py")
Invoke-CheckedPython @("scripts\build_stress_pair_extension.py")
"REVIEW_REVISION_QUEUE_PASS" | Tee-Object -FilePath $Log -Append
