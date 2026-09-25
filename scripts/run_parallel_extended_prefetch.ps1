$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
Set-Location -LiteralPath $ProjectRoot

function Invoke-CheckedPython {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    Write-Output ("[EXTENDED PREFETCH] {0} {1}" -f $Python, ($Arguments -join " "))
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

$factorial = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq "python.exe" -and
    $_.CommandLine -like "*run_split_training_seed_factorial.py*--points Core Asym*"
}
if ($factorial) {
    $ids = @($factorial.ProcessId)
    Write-Output ("[EXTENDED PREFETCH] waiting for Core/Asym factorial PIDs: {0}" -f ($ids -join ","))
    Wait-Process -Id $ids
}

Invoke-CheckedPython scripts\prepare_ncmapss_dev_rotation.py
Invoke-CheckedPython scripts\run_ncmapss_dev_rotation.py --points Core Asym
Write-Output "EXTENDED_PREFETCH_PASS"
