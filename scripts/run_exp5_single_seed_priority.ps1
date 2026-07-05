$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

& "$PSScriptRoot\run_exp5_paper.ps1" `
    -Conditions @("one_way", "duplex_direct", "duplex_curriculum") `
    -Folds @(0) `
    -Seeds @(0) `
    -LogDir "results/exp5_population_duplex/run_logs_single_seed_priority"
