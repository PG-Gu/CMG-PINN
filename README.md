# CMG PINN experiments

## Environment

Set the DeepXDE backend before running:

```bash
export DDE_BACKEND=pytorch
python train.py --help
```

## Training

Run the complete learning-rate search for one task and method using all five
seeds (0-4):

```bash
python train.py --task burgers --method cmg_layer_match --gpu 0 --output results/search
```

Omit `--task` and `--method` to run the complete benchmark. Use `--seed 0` for an
individual seed. Use `--gpu 0 1 2` to distribute jobs, one process per GPU, or
`--gpu cpu` for CPU execution.

Method keys are `cmg_layer_match`, `cmg_gelu_match`, `vanilla_match`, `gelu_match`,
`react_match`, `sigmexd_match`, `swish_match`, `sine_match`,
`jagtap_adaptive_tanh_match` (L-LAAF), `piratenet`, and `fourier_feature_match`.
Task keys and every architecture, budget, metric, and search grid
are listed in `configs/benchmark.json`.

For the complete original learning-rate search:

```bash
python search.py --gpu 0 --output results/search
```

Search uses the stored candidate pairs for each task and method, with five seeds
per pair. Selection uses the pair with the lowest complete five-seed mean error.

## Figures and separability

Download the official implementations associated with
[Acevedo et al. (2022)](https://doi.org/10.1109/ACCESS.2022.3152789)
and [Acevedo et al. (2024)](https://doi.org/10.1371/journal.pcsy.0000012),
following the code-availability links in the papers.
Place them in `third_party/psi/` and `third_party/tsps/`, respectively.

```bash
python make_figures.py --results results/search --output figures
python separability.py --geometry figures/geometry.json --matlab matlab --concorde /path/to/concorde --output separability
python make_figures.py --results results/search --separability separability/scores.json --output figures
```

## Source layout

| Directory                      | Purpose                                                      |
| ------------------------------ | ------------------------------------------------------------ |
| `configs/`                   | Fixed paper configurations and analysis order                |
| `src/activations/`           | CMG, CMG-GELU and activation baselines         |
| `src/models/`                | MLP, PirateNet and Fourier-feature architectures             |
| `src/tasks/`                 | Geometry, PDE/BC/IC losses, transforms and reference metrics |
| `src/training/`              | Muon, sampling replay and training   |
| `src/analysis/`              | Seed aggregation, medoids, PCA and rendering         |
| `data/`                      | Required reference and derived training/evaluation data      |
| `scripts/`                | MATLAB metric driver                                         |
