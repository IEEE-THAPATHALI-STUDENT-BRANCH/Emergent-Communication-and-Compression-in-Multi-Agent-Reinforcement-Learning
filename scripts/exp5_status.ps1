param(
    [string[]]$Conditions = @("no_comm", "one_way", "duplex_direct", "duplex_curriculum"),
    [int[]]$Folds = @(0, 1, 2, 3),
    [int[]]$Seeds = @(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
)

$total = 0
$done = 0
foreach ($condition in $Conditions) {
    foreach ($fold in $Folds) {
        foreach ($seed in $Seeds) {
            $total += 1
            $checkpoint = "results/exp5_population_duplex/paper/$condition/fold_$fold/seed_$seed/population.pt"
            if (Test-Path $checkpoint) {
                $done += 1
            }
        }
    }
}

[pscustomobject]@{
    Completed = $done
    Total = $total
    Remaining = $total - $done
    Percent = [math]::Round(100.0 * $done / $total, 2)
}
