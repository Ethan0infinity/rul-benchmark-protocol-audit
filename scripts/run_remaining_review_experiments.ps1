$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
Set-Location -LiteralPath $ProjectRoot

function Invoke-CheckedPython {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    Write-Output ("[REVIEW PIPELINE] {0} {1}" -f $Python, ($Arguments -join " "))
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

function Wait-ForPythonScript {
    param([Parameter(Mandatory = $true)][string]$ScriptName)
    $running = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq "python.exe" -and $_.CommandLine -like "*$ScriptName*"
    }
    if ($running) {
        $ids = @($running.ProcessId)
        Write-Output ("[REVIEW PIPELINE] waiting for {0} PIDs: {1}" -f $ScriptName, ($ids -join ","))
        Wait-Process -Id $ids
    }
}

$runningAugmentation = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq "python.exe" -and
    $_.CommandLine -like "*run_augmentation_distribution_sensitivity.py*"
}
if ($runningAugmentation) {
    $ids = @($runningAugmentation.ProcessId)
    Write-Output ("[REVIEW PIPELINE] waiting for augmentation PIDs: {0}" -f ($ids -join ","))
    Wait-Process -Id $ids
}

# Re-run the grid driver to validate/reuse all completed cells and write one canonical 75-row manifest.
Invoke-CheckedPython scripts\run_augmentation_distribution_sensitivity.py
Invoke-CheckedPython scripts\analyze_augmentation_distribution_sensitivity.py

Wait-ForPythonScript "run_split_training_seed_factorial.py"
Invoke-CheckedPython scripts\run_split_training_seed_factorial.py
Invoke-CheckedPython scripts\analyze_split_training_seed_factorial.py

Wait-ForPythonScript "run_ncmapss_dev_rotation.py"
Invoke-CheckedPython scripts\prepare_ncmapss_dev_rotation.py
Invoke-CheckedPython scripts\run_ncmapss_dev_rotation.py
Invoke-CheckedPython scripts\analyze_ncmapss_dev_rotation.py

Invoke-CheckedPython scripts\check_extended_review_evidence.py
Write-Output "REMAINING_REVIEW_EXPERIMENTS_PASS"
