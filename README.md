# Defining Optimal Schedule Review Periods Using EDM, Monte Carlo Simulation, and Tree-Based Machine Learning

Code and data for the article

> Carneiro, A. H. A., & Ho, L. L. *Defining Optimal Schedule Review Periods Using Earned Duration Management, Monte Carlo Simulation, and Tree-Based Machine Learning*. Quality and Reliability Engineering International (under review).

The framework finds the smallest set of review instants whose ability to predict a schedule delay is
statistically non-inferior (DeLong's test, margin δ) to monitoring at every candidate instant. It combines
a Monte Carlo simulator of the project network, the Duration Performance Index (DPI) of Earned Duration
Management, tree-based classifiers, and Boruta feature selection, under a train–validation–test protocol.

## Repository structure

| Folder | Content |
|---|---|
| `code/revision/mc_simulator.py` | Vectorized Monte Carlo simulator of the 32-activity EPC project (triangular, Beta-PERT, Gamma and Conway–Maxwell–Poisson durations); reproduces the original loop-based simulator bit by bit |
| `code/original_pipeline/` | Preprocessing, modeling and experiment pipeline used in the configuration experiment (1,728 configurations), the R script of the regression trees, and notebooks 1–3 |
| `code/revision/` | One script per analysis of the article (see the table below) and shared utilities (`common.py`, `pipeline_v2.py`, `inference.py`) |
| `data/project/` | Project parameter files: network, activity-duration distributions and costs (`db_original*.xlsx`) |
| `data/raw/` | Results of the configuration experiment (`df_exp_simulation.csv`, `features_regression_rpart.csv`) and baseline schedule (`gantt_chart.csv`) |
| `data/processed/` | `manifest.json` with generator, seed, size, share of late projects and SHA-256 of every dataset; the datasets themselves (Parquet, 2.8 GB) are archived at Zenodo |
| `results/` | Outputs (CSV/JSON) of every analysis, organized by experiment (E0–E11) |
| `figures/` | Figures of the article |

## Data

All simulated datasets (training set of the article, 20 Monte Carlo replications, alternative duration
distributions, and the independent validation and test sets of 100,000 projects) are available at Zenodo:
**DOI: [to be added]**. Download them into `data/processed/` and check them against `manifest.json`:

```bash
python code/revision/verify_datasets.py
```

Every dataset can also be regenerated from its seed with the simulator, e.g.

```python
import sys; sys.path.insert(0, "code/revision")
import mc_simulator as mc
dpi, result = mc.simulate(n_sim=10000, seed=2001, dist="pert")  # = pert_train
```

## Environment

Python 3.11; versions in `requirements.txt` (scikit-learn 1.5.2, XGBoost 2.1.1, LightGBM 4.5.0,
hyperopt 0.2.7, imbalanced-learn 0.12.4, BorutaPy 0.4.3).

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Reproducing the article

| Article | Script | Output |
|---|---|---|
| Simulated data (Figures 3–4) | `fig_eda.py` | `figures/` |
| Configuration experiment (Tables 3–4, Figure 5, Figure S1) | `code/original_pipeline/` (notebooks 1–3), `fig_eda.py`, `fig_regression_trees.R` | `data/raw/df_exp_simulation.csv`, `data/raw/features_regression_rpart.csv` |
| Review periods and non-inferiority test (Table 5, Figure 6) | `e3d_candidate_benchmark_rule.py`, `fig_tradeoff.py` | `results/E3_inference/` |
| Monte Carlo replications and distributions (Table 6, Figure 7) | `final_rule.py`, `e4b_e6_fixed_config.py`, `e4_aggregate_robustness.py` | `results/E4_mc_replicates/`, `results/E5_distributions/` |
| Train–test partitions | `e4b_e6_fixed_config.py` | `results/E6_splits/` |
| Selection procedures (Boruta per family, permutation importance, lasso) | `e7_selection_methods.py` | `results/E7_selection_methods/` |
| Correlation threshold | `e8_thresholds.py`, `e8a_corr095.py` | `results/E8_thresholds/` |
| Computation time (Table 7) | `mc_simulator.py`, timing in `results/E9_computation_time/` | `results/E9_computation_time/` |
| Interpretation and SHAP (Figures 8–9) | `e10_interpretation.py` | `results/E10_interpretation/` |
| Final model evaluation (Figures 10–11) | `e11_model_evaluation_figures.py` | `results/E3_inference/` |
| Baseline schedule (Figure 2) | `fig_gantt.py` | `figures/` |

Run the scripts from the repository root, e.g. `python code/revision/e3d_candidate_benchmark_rule.py`.

## Citation

See `CITATION.cff`. The Monte Carlo simulator and the case study were introduced in Carneiro, A. H. A., & Ho,
L. L. (2024), *Proposal and appraisal of new indicators and performance metrics for the project monitoring*
(original simulator: https://github.com/andre-carneiro/SPC-control-charts).

## License

Code: MIT License (`LICENSE`). Data (project parameter files, simulated datasets and results): Creative Commons Attribution 4.0 International (CC BY 4.0).
