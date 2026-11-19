# MaskNet: Neural Imputation of Categorical Medical Data Trained on Incomplete Records

Code, data and results for the paper

> S. Cho, Y. Lee, and C. W. Nam, "Neural Imputation of Categorical Medical Data Trained on Incomplete Records: Class Weighting, Per-Variable Networks, and a Shared Masked Network" (under review).

The repository reproduces every table and figure of the paper: a controlled comparison of loss weighting, per-variable versus shared architectures, and training on incomplete versus complete records, in an inductive benchmark of 14 imputation methods on two public medical datasets under MCAR, MAR and MNAR missingness at 10%, 30% and 50% (54 settings), plus an ablation study and a training-size analysis.

## Contents

```
code/
  preprocess.py      raw UCI files -> fully categorical integer tables (data/*.npz)
  masks.py           MCAR / logistic MAR / self-masking MNAR generators
  nn.py              NumPy neural-network toolkit with hand-written gradients (MLP, Transformer, Adam)
  imputers.py        Mode, kNN, MICE, MICE-S, MissForest, GAIN, MIDAS-DAE, DAE-complete(-unw),
                     Noise DNN(-unw), MaskNet / MaskNet-S, ReMasker (categorical adaptation)
  metrics.py         accuracy, macro-F1, balanced accuracy, TVD, downstream coefficient RMSE
  run_benchmark.py   runs the main benchmark, the ablation and the training-size analysis
  analyze.py         statistics (Wilcoxon + Holm, cluster bootstrap CIs, rank-biserial r) and figures
  make_tables.py     LaTeX tables of the paper
  verify_claims.py   recomputes every number quoted in the paper's text (209 numbers) from the raw results
  vm_remasker.py, cloud_remasker.py, merge_remasker.py
                     helpers used to split the ReMasker runs over two machines (see below)
data/
  raw/               the two original UCI files (see "Data sources and licences")
  brfss.npz, diabetes.npz, *_meta.json   preprocessed tables and variable definitions
results/
  main/              one JSON file per setting (dataset_mechanism_rate_seed_ntrain.json)
  ablation/, size/   the same for the ablation study and the training-size analysis
  summary.json       aggregated results used for the tables
  verify_claims_output.txt   209 numbers of the paper checked against the raw results
figures/             Figures 2-4 of the paper
```

## Requirements

Python 3.10+ with NumPy, SciPy, pandas, scikit-learn and matplotlib (`pip install -r requirements.txt`). No GPU or deep-learning framework is needed: all neural models are implemented in NumPy, and their gradients were verified against finite differences.

## Reproducing the results

```bash
python code/preprocess.py data                 # data/raw -> data/brfss.npz, data/diabetes.npz
python code/run_benchmark.py main 0 1          # 54 settings x 14 methods
python code/run_benchmark.py ablation 0 1      # 18 settings x 10 MaskNet variants
python code/run_benchmark.py size 0 1          # 1,000 / 20,000 training records
python code/analyze.py                         # results/summary.json and figures/
python code/make_tables.py tables              # LaTeX tables
python code/verify_claims.py                   # check the numbers quoted in the paper
```

`run_benchmark.py <mode> <worker_id> <n_workers>` splits the settings over several processes (e.g. `main 0 2` and `main 1 2`), and every finished setting is written to its own JSON file, so interrupted runs resume where they stopped. All timings in the paper were measured on one CPU core (numerical libraries restricted to one thread).

Every random number is derived from fixed seeds (sample, masks and model initialisation per setting and method), so results are reproducible to the last digit on the same software stack. We checked this by re-running settings from scratch: the stored results were reproduced to six decimals, including a ReMasker run originally computed on a different machine.

**ReMasker.** ReMasker accounts for about 70% of the total run time (about 4-5 min per setting on one core). To save time, the paper's ReMasker runs were spread over two machines with `vm_remasker.py` (checkpointed, time-budgeted runs) and `cloud_remasker.py`, and merged into `results/main` with `merge_remasker.py`; times from the second machine were rescaled by MaskNet's time on both machines. Running `run_benchmark.py main` on a single machine gives the same imputations.

## Data sources and licences

| File | Source | Licence |
|---|---|---|
| `data/raw/uci296_diabetes_130us.zip` | Diabetes 130-US Hospitals for Years 1999-2008, UCI Machine Learning Repository, doi:10.24432/C5230J | CC BY 4.0 |
| `data/raw/uci891_cdc_diabetes_health_indicators.csv` | CDC Diabetes Health Indicators (BRFSS 2015 extract), UCI Machine Learning Repository, doi:10.24432/C53919; originally from the CDC Behavioral Risk Factor Surveillance System via Kaggle (alexteboul/diabetes-health-indicators-dataset) | See the linked source; BRFSS data are public-use data of the U.S. CDC |

The files are redistributed unchanged for reproducibility. If you use them, please cite the original sources:

- B. Strack et al., "Impact of HbA1c measurement on hospital readmission rates: Analysis of 70,000 clinical database patient records," *BioMed Research International*, 2014, Art. no. 781670.
- Centers for Disease Control and Prevention, Behavioral Risk Factor Surveillance System Survey Data, 2015.

`data/*.npz` and `results/` are derived from these files.

## Licence

The code is released under the MIT License (see `LICENSE`). The data files keep the licences of their sources listed above.
