# cost-aware-sales-prediction
Code and experiments for: Leakage-Controlled Re-evaluation and Cost-Aware Conformal Intervals for E-Commerce Sales Prediction

The study re-evaluates the sales-prediction experiment of Sayyad et al. (IEEE Access, 2024) on the public DataCo
supply-chain dataset. It separates algorithm capability from evaluation design by

1. auditing target leakage with three feature levels (L0 all columns, L1 clean, L2 strict),
2. comparing a temporal-grouped split with a grouped-random control,
3. training cost-aware (quantile-loss) forecasters under six tuning arms (default, TPE, random search, GWO, GWO→TPE),
4. building Cost-Asymmetric Split Conformal (CASC) prediction intervals and checking their coverage month by month.

5. The dataset is **not** included (see [Data](#data)).
6. link video: https://www.loom.com/share/9af3df7812f54b56a9c1805e79c9294e

## Quick start

### Requirements

- Python 3.x <!-- TODO: exact version -->
- `numpy`, `pandas`, `scipy`, `scikit-learn`, `lightgbm`, `catboost`, `optuna`, `matplotlib`
- Reported experiments were run on Google Colab (8 vCPUs, AMD EPYC 7B12, 51 GB RAM).

### Data

Download the DataCo Smart Supply Chain dataset from Kaggle:
<https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supplychain-for-big-data-analysis>
(file `DataCoSupplyChainDataset.csv`, 180,519 order lines, 53 attributes).

### Run

The script is configured by constants at the top of `src/gwo_optuna_v5.py` (section `[0] KONFIGURASI`, lines 64-76).
Paths are read from environment variables:

```bash
export DATACO_CSV=/path/to/DataCoSupplyChainDataset.csv
export RESULTS_DIR=results/run_T_temporal
```

Recommended order, as noted in the script docstring: `MODE='smoke'` (sanity check), then `'quick'`, then `'full'`.
All outputs (CSV tables, figures, selected-feature JSON) are written to `RESULTS_DIR`.

## Configuration of the reported runs

| Run | Role in the paper | `SPLIT_MODE` | `MAIN_LEVEL` | `MODE` | Notes |
|---|---|---|---|---|---|
| **G** | Exchangeable control | `'grouped_random'` | `'strict'` | `'full'` | 3 seeds {0,1,2}, 40 TPE trials (15 for Ridge) |
| **T** | Main temporal evaluation | `'temporal'` | `'strict'` | `'full'` | Same budget as Run G |
| **N** | Robustness check without calendar features | `'grouped_random'` <!-- TODO: confirm --> | `'strict_nocal'` | `'quick'` | 1 seed |

Other settings shared by all runs: `TARGET='Sales_per_customer'`, `C_OVER=0.10`, `C_UNDER=0.20` (so `TAU = 2/3`),
`SPLIT_Q=(0.65, 0.75, 0.85)`, `CONF_ALPHA=0.10`, `BUDGET_S=None` (trial-matched budget).

For Table VII (learning-rate sweep) set `RUN_LR_SWEEP=True`; it uses one seed and the `leak` level (`SWEEP_LEVELS`).

## Pipeline

1. **Data preparation.** Text and personal fields are dropped; calendar features (month, day, weekday, hour) are derived from the order
   timestamp; the year is excluded. Identifier-like numeric columns are treated as categorical.
2. **Four-way split** by order: train 65% | tuning-validation 10% | calibration 10% | test 15%. All items of an order stay in the same block
   (checked by an `assert`). Temporal mode cuts on the order's first timestamp (percentiles taken over order lines);
   grouped-random mode assigns whole orders at random. Preprocessing is fitted on the training block only.
3. **Leakage levels** (`LEVEL_DROP`): L0 keeps all columns; L1 removes price-derived columns and record identifiers;
   L2 additionally removes post-order information and product-identity proxies. `strict_nocal` removes the order calendar features on top of L2.
4. **Learners:** Ridge, LightGBM, CatBoost behind one interface (`fit_predict`); `tau=None` gives squared error, `tau=2/3` gives quantile loss.
5. **Six arms:** A default/MSE, B default/quantile, C TPE, D GWO-selected features, E GWO→TPE hybrid, F random search.
6. **Intervals:** range of Sayyad et al. (ŷ ± z·RMSE), symmetric split conformal, and CASC
   (`conformal_asym`: miscoverage split as α·c_o/(c_o+c_u) on the upper side).
7. **Evaluation:** point metrics, asymmetric cost, cost reduction against the constant τ-quantile forecast, order-level paired bootstrap, seed-level paired differences, per-month coverage.

8. ## Output files and the paper

| File (in `RESULTS_DIR`) | Content | Paper |
|---|---|---|
| `T1_leakage_ablation.csv` | R², RMSE, MSE (min-max) by level and learner | Table V |
| `F1_importance_leak.png` | LightGBM gain importance at L0 | Fig. 2 |
| `F2_leak_ablation.png` | R² by level and learner | Fig. 3 |
| `T7a_rival_replication.csv` | CatBoost with the parameters of Sayyad et al. | Table VI |
| `T7b_lr_sweep.csv` | Learning-rate sweep, fixed 300 iterations vs early stopping | Table VII |
| `T2_arms_raw.csv`, `T2_arms_aggregated.csv` | Six arms, per seed and aggregated | Table VIII |
| `F3_arms.png` | Cost reduction by arm | Fig. 4 |
| `F4_gwo_convergence.png` | GWO convergence per learner and seed | Fig. 5 |
| `T3b_seedlevel_pairs.csv` | Seed-level cost differences between arms | Table IX |
| `T3_bootstrap.csv` | Order-level paired bootstrap, 95% intervals | Section IV-C |
| `T4_sensitivity.csv`, `F5_sensitivity.png` | Sensitivity to the cost ratio (LightGBM) | Fig. 6 |
| `T5_intervals_raw.csv`, `T5_intervals_aggregated.csv` | Coverage, width, interval score | Table X |
| `T6_coverage_by_month.csv`, `F6_interval_coverage.png` | Monthly coverage (first seed only) | Fig. 7 |
| `T8_shift_by_month.csv` | Test-block distribution shift by month | Table XI |
| `T9a_gwo_feature_frequency.csv`, `T9b_gwo_jaccard.csv` | GWO selection stability | Section IV-C |
| `selected_features_and_config.json` | Configuration and GWO-selected features per learner and seed | Appendix |

Figure numbering in the script (`F1`-`F6`) differs from the paper (Fig. 2-7); Fig. 1 is the pipeline diagram.

## Headline results

Full tables and caveats are in the paper.

- With all columns (L0), a linear model reaches R² = 1.0000 because `Sales_per_customer` is identical to `Order_Item_Total`;
  at the strict level under the temporal split, the tree models reach R² ≈ 0.005.
- Switching from squared-error to quantile loss reduces total test cost by 4.7-8.7% under the temporal split; tuning contributes about 0.5% or less.
- GWO retains 5.3-8.3 of 20 candidate features; the GWO→TPE hybrid is not consistently more accurate than TPE alone.
- CASC reaches the nominal 90% coverage while data are exchangeable, and coverage collapses after the October 2017 catalogue break.

## Limitations of the code and study

The strict level is an information-restricted stress test, the unit costs are assumed, results use three seeds and one dataset,
and the time-matched comparison was not run. See Section IV-F of the paper.
