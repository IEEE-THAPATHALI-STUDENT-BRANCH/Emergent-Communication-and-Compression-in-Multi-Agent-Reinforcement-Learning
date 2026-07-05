# Experiment 5: Population-Level Duplex Emergent Communication Under Constraints

This package is an isolated implementation of Experiment 5. It does not overwrite the earlier `referential_game` experiments or their results.

## Dependency

Training and evaluation require PyTorch:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

The configs are JSON-compatible YAML files, so no PyYAML dependency is required.

## Quick Smoke Test

```powershell
python -m exp5_population_duplex.training --config configs/exp5_population_duplex_quick.yaml --condition duplex_direct --fold 0 --seed 0
```

Evaluate the produced checkpoint:

```powershell
python -m exp5_population_duplex.evaluation --config configs/exp5_population_duplex_quick.yaml --checkpoint results/exp5_population_duplex/quick/duplex_direct/fold_0/seed_0/population.pt --fold 0 --seed 0 --split id --output results/exp5_population_duplex/quick_eval
```

Generate a cross-play matrix:

```powershell
python -m exp5_population_duplex.evaluation --config configs/exp5_population_duplex_quick.yaml --checkpoint results/exp5_population_duplex/quick/duplex_direct/fold_0/seed_0/population.pt --fold 0 --seed 0 --split ood --output results/exp5_population_duplex/quick_eval --crossplay
```

Generate tables and optional figures:

```powershell
python -m exp5_population_duplex.analysis --config configs/exp5_population_duplex_quick.yaml --results results/exp5_population_duplex --output results/exp5_population_duplex/analysis --crossplay_csv results/exp5_population_duplex/quick_eval/crossplay_ood.csv
```

## Single Seed Run

```powershell
python -m exp5_population_duplex.training --config configs/exp5_population_duplex.yaml --condition one_way --fold 0 --seed 0
python -m exp5_population_duplex.training --config configs/exp5_population_duplex.yaml --condition duplex_direct --fold 0 --seed 0
python -m exp5_population_duplex.training --config configs/exp5_population_duplex.yaml --condition duplex_curriculum --fold 0 --seed 0
```

## Single Fold Run

Run all configured seeds for one fold by launching one command per seed:

```powershell
foreach ($s in 0,1,2,3,4,5,6,7,8,9) {
  python -m exp5_population_duplex.training --config configs/exp5_population_duplex.yaml --condition duplex_direct --fold 0 --seed $s
}
```

## Full Paper Run

For each condition, run all configured folds and seeds:

```powershell
python -m exp5_population_duplex.training --config configs/exp5_population_duplex.yaml --condition no_comm --all
python -m exp5_population_duplex.training --config configs/exp5_population_duplex.yaml --condition one_way --all
python -m exp5_population_duplex.training --config configs/exp5_population_duplex.yaml --condition duplex_direct --all
python -m exp5_population_duplex.training --config configs/exp5_population_duplex.yaml --condition duplex_curriculum --all
```

Do not treat quick-mode results as final research evidence.

## Evaluation-Only From Checkpoints

```powershell
python -m exp5_population_duplex.evaluation --config configs/exp5_population_duplex.yaml --checkpoint <path-to-population.pt> --fold 0 --seed 0 --split id --output results/exp5_population_duplex/eval
python -m exp5_population_duplex.evaluation --config configs/exp5_population_duplex.yaml --checkpoint <path-to-population.pt> --fold 0 --seed 0 --split ood --output results/exp5_population_duplex/eval
python -m exp5_population_duplex.evaluation --config configs/exp5_population_duplex.yaml --checkpoint <path-to-population.pt> --fold 0 --seed 0 --split mixed --output results/exp5_population_duplex/eval
```

Feedback ablations:

```powershell
python -m exp5_population_duplex.evaluation --config configs/exp5_population_duplex.yaml --checkpoint <path-to-population.pt> --fold 0 --seed 0 --split ood --feedback_ablation empty --output results/exp5_population_duplex/eval
python -m exp5_population_duplex.evaluation --config configs/exp5_population_duplex.yaml --checkpoint <path-to-population.pt> --fold 0 --seed 0 --split ood --feedback_ablation shuffled --output results/exp5_population_duplex/eval
```

Noise robustness cells:

```powershell
python -m exp5_population_duplex.evaluation --config configs/exp5_population_duplex.yaml --checkpoint <path-to-population.pt> --fold 0 --seed 0 --split ood --p_noise 0.20 --output results/exp5_population_duplex/eval_noise
```

## Output Directories

- `results/exp5_population_duplex/<mode>/<condition>/fold_<n>/seed_<n>/`: checkpoints, config snapshot, training metrics, and summaries.
- `results/exp5_population_duplex/eval*/`: raw per-episode evaluation CSVs, metrics JSON files, and cross-play matrices.
- `results/exp5_population_duplex/analysis/`: generated figures, tables, and conflict-graph summaries.

## Expected Artifacts

- `population.pt`: six independently parameterized agents.
- `training_metrics.csv`: raw batch-level success, return, and message lengths.
- `evaluation_<split>_<ablation>.csv`: raw per-episode evaluation records.
- `metrics_<split>_<ablation>.json`: aggregate metrics derived from raw records.
- `crossplay_<split>.csv`: ordered informant-by-receiver cross-play matrix.
- `tables/`: generated hyperparameter, metrics, and conflict-graph tables.
- `fig*.png`: generated plots when inputs are provided to the analysis command.

## Assumptions

- The new implementation is intentionally isolated in `exp5_population_duplex/`.
- Config files use JSON syntax with a `.yaml` extension to avoid adding a YAML parser dependency.
- The quick config is for implementation checks only; paper-mode evidence requires the full configured seed/fold/evaluation budget.
- No conclusions are hardcoded. Metrics and plots are generated from saved raw logs.

