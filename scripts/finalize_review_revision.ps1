param(
    [int]$WaitForPid = 0
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$WorkspaceRoot = Split-Path -Parent $ProjectRoot
$ManuscriptDirectoryName = [string]::Concat([char]0x6B63, [char]0x6587)
$ManuscriptRoot = Join-Path $WorkspaceRoot $ManuscriptDirectoryName
$SupplementaryRoot = Join-Path $ProjectRoot "supplementary"
$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$Log = Join-Path $ProjectRoot "reports\review_revision_finalize.log"

function Write-Stage {
    param([string]$Message)
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message" | Tee-Object -FilePath $Log -Append
}

function Invoke-CheckedPython {
    param([string[]]$Arguments)
    Write-Stage "START python $($Arguments -join ' ')"
    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Python @Arguments 2>&1 | Tee-Object -FilePath $Log -Append
        $ExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
    if ($ExitCode -ne 0) {
        throw "Command failed with exit code ${ExitCode}: python $($Arguments -join ' ')"
    }
    Write-Stage "PASS python $($Arguments -join ' ')"
}

function Invoke-CheckedLatexmk {
    param([string]$Source, [string]$Directory = $ManuscriptRoot)
    Write-Stage "START latexmk $Source"
    Push-Location -LiteralPath $Directory
    try {
        $PreviousErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            & latexmk -xelatex -interaction=nonstopmode -halt-on-error $Source 2>&1 |
                Tee-Object -FilePath $Log -Append
            $ExitCode = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $PreviousErrorActionPreference
        }
        if ($ExitCode -ne 0) {
            throw "latexmk failed with exit code ${ExitCode}: $Source"
        }
    }
    finally {
        Pop-Location
    }
    Write-Stage "PASS latexmk $Source"
}

Set-Location -LiteralPath $ProjectRoot
if ($WaitForPid -gt 0) {
    Write-Stage "WAIT pid=$WaitForPid"
    Wait-Process -Id $WaitForPid -ErrorAction SilentlyContinue
}

Invoke-CheckedPython @(
    "scripts\analyze_release_statistics.py",
    "--max-t-calibration-simulations", "1000",
    "--max-t-calibration-bootstrap-reps", "999"
)
Invoke-CheckedPython @("scripts\analyze_preference_selection_stability.py")
Invoke-CheckedPython @("scripts\analyze_augmentation_distribution_sensitivity.py", "--reuse-stress")
Invoke-CheckedPython @("scripts\analyze_standard_protocol_track.py")
Invoke-CheckedPython @("scripts\analyze_ocm_attribution.py")
Invoke-CheckedPython @("scripts\analyze_design_sensitivity.py")
Invoke-CheckedPython @("scripts\run_endpoint_selection_confirmation.py", "--reuse")
Invoke-CheckedPython @("scripts\analyze_validation_endpoint_distributions.py")
Invoke-CheckedPython @("scripts\analyze_major_revision_evidence.py")
Invoke-CheckedPython @("scripts\analyze_condition_normalization.py")
Invoke-CheckedPython @("scripts\analyze_model_mechanisms.py", "--reuse")
Invoke-CheckedPython @("scripts\analyze_ncmapss_validation_only_k_selection.py")
Invoke-CheckedPython @("scripts\run_official_dual_mixer_baseline.py", "--reuse")
Invoke-CheckedPython @("scripts\analyze_official_dual_mixer_baseline.py")
Invoke-CheckedPython @("scripts\build_stress_pair_extension.py", "--reuse")
Invoke-CheckedPython @("scripts\extend_core_stress_evidence.py", "--reuse")
Invoke-CheckedPython @("scripts\build_engine_level_stress_evidence.py", "--reuse-predictions")
Invoke-CheckedPython @("scripts\analyze_cross_backbone_protocol_buildup.py")
Invoke-CheckedPython @("scripts\analyze_fast_cudnn_repeat_audit.py")
Invoke-CheckedPython @("scripts\build_revised_assets.py")
Invoke-CheckedPython @("scripts\build_advanced_evidence.py")
Invoke-CheckedPython @("scripts\analyze_final_revision_evidence.py")
Invoke-CheckedPython @("scripts\check_advanced_evidence.py")
Invoke-CheckedPython @("scripts\check_figure_data_consistency.py")
Invoke-CheckedPython @("scripts\check_method_figure_qa.py")
Invoke-CheckedPython @(
    "scripts\build_manuscript_evidence.py",
    "--copy-to", (Join-Path $ManuscriptRoot "generated")
)
Invoke-CheckedPython @("scripts\check_manuscript_evidence.py")
Invoke-CheckedPython @("scripts\build_release_evidence.py")
Invoke-CheckedPython @("scripts\release_evidence_gate.py")

$SupplementaryGenerated = Join-Path $SupplementaryRoot "generated"
New-Item -ItemType Directory -Path $SupplementaryGenerated -Force | Out-Null
Invoke-CheckedPython @(
    "scripts\build_manuscript_evidence.py",
    "--copy-to", $SupplementaryGenerated
)
Write-Stage "PASS synchronized supplementary generated evidence"

$AdvancedFigures = Join-Path $ProjectRoot "paper_outputs\advanced_evidence\figures"
$ManuscriptFigures = Join-Path $ManuscriptRoot "figures"
New-Item -ItemType Directory -Path $ManuscriptFigures -Force | Out-Null
Get-ChildItem -LiteralPath $AdvancedFigures -File | Where-Object { $_.Extension -in ".pdf", ".png", ".svg", ".tiff" } |
    ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $ManuscriptFigures -Force }
$AnalysisFigures = Join-Path $ProjectRoot "paper_outputs\analysis_working\figures"
Get-ChildItem -LiteralPath $AnalysisFigures -File | Where-Object { $_.Extension -in ".pdf", ".png", ".svg", ".tiff" } |
    ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $ManuscriptFigures -Force }
Write-Stage "PASS synchronized advanced figures"

$SupplementaryFigures = Join-Path $SupplementaryRoot "figures"
New-Item -ItemType Directory -Path $SupplementaryFigures -Force | Out-Null
Get-ChildItem -LiteralPath $ManuscriptFigures -File |
    ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $SupplementaryFigures -Force }
Write-Stage "PASS synchronized supplementary figures"

Invoke-CheckedPython @("scripts\sync_cross_document_release.py")
Invoke-CheckedPython @("scripts\validate_data.py")
Invoke-CheckedPython @("scripts\run_core_tests.py")
Invoke-CheckedLatexmk "main_revised.tex"
Invoke-CheckedLatexmk "main_zh.tex"
Invoke-CheckedLatexmk "supplement_en.tex" $SupplementaryRoot
Invoke-CheckedLatexmk "supplement_zh.tex" $SupplementaryRoot
Invoke-CheckedPython @("scripts\check_cross_document_sync.py")
Invoke-CheckedPython @(
    "scripts\create_replication_package.py",
    "--experiment-prefix", "paper_main_v3_seed"
)
Invoke-CheckedPython @(
    "scripts\verify_results_manifest.py",
    "--package-root", "replication_package",
    "--expected-rows", "260"
)
Invoke-CheckedPython @("scripts\create_submission_package.py")

Write-Stage "NONBLOCKING manuscript metadata audit"
$PreviousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    & $Python "scripts\check_submission_manuscripts.py" 2>&1 | Tee-Object -FilePath $Log -Append
    Write-Stage "NONBLOCKING final paper gate (public artifact and author-owned metadata may remain pending)"
    & $Python "scripts\paper_quality_gate.py" --stage final 2>&1 |
        Tee-Object -FilePath $Log -Append
}
finally {
    $ErrorActionPreference = $PreviousErrorActionPreference
}

Write-Stage "REVIEW_REVISION_FINALIZE_PASS"
