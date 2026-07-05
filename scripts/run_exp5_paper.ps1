param(
    [string]$Config = "configs/exp5_population_duplex.yaml",
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string[]]$Conditions = @("no_comm", "one_way", "duplex_direct", "duplex_curriculum"),
    [int[]]$Folds = @(0, 1, 2, 3),
    [int[]]$Seeds = @(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    [string]$LogDir = "results/exp5_population_duplex/run_logs"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$masterLog = Join-Path $LogDir "paper_run_master.log"
"[$(Get-Date -Format o)] Starting Experiment 5 paper-scale run" | Tee-Object -FilePath $masterLog -Append

foreach ($condition in $Conditions) {
    foreach ($fold in $Folds) {
        foreach ($seed in $Seeds) {
            $checkpoint = "results/exp5_population_duplex/paper/$condition/fold_$fold/seed_$seed/population.pt"
            $runLog = Join-Path $LogDir "$condition`_fold_$fold`_seed_$seed.log"

            if (Test-Path $checkpoint) {
                "[$(Get-Date -Format o)] SKIP completed $condition fold=$fold seed=$seed" |
                    Tee-Object -FilePath $masterLog -Append
                continue
            }

            "[$(Get-Date -Format o)] RUN $condition fold=$fold seed=$seed" |
                Tee-Object -FilePath $masterLog -Append

            & $Python -m exp5_population_duplex.training `
                --config $Config `
                --condition $condition `
                --fold $fold `
                --seed $seed 2>&1 |
                Tee-Object -FilePath $runLog -Append

            if ($LASTEXITCODE -ne 0) {
                "[$(Get-Date -Format o)] FAILED $condition fold=$fold seed=$seed exit=$LASTEXITCODE" |
                    Tee-Object -FilePath $masterLog -Append
                exit $LASTEXITCODE
            }

            "[$(Get-Date -Format o)] DONE $condition fold=$fold seed=$seed" |
                Tee-Object -FilePath $masterLog -Append
        }
    }
}

"[$(Get-Date -Format o)] Experiment 5 paper-scale run complete" | Tee-Object -FilePath $masterLog -Append
