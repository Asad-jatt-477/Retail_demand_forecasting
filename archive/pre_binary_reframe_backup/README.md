# 📦 Retail Demand Forecasting

Predicts the **quantity-class (1–5)** of an incoming retail order-item, using
only information genuinely available at the moment the order is placed —
built for warehouse / inventory planning (fast-pick shelving vs. bulk-zone
picking).

> **Status:** v2 — leakage-fixed, size-capped, evaluated, explained, and
> deployment-ready. See [`PROJECT_CHANGELOG.md`](PROJECT_CHANGELOG.md) for
> the full history of what changed and why.

## Results (test set, v2)

| Metric | Baseline | **Random Forest (winner)** | XGBoost |
|---|---|---|---|
| Accuracy | 0.350 | **0.579** | 0.574 |
| F1-Macro | 0.200 | **0.393** | 0.375 |
| F1-Weighted | — | **0.604** | 0.593 |
| Saved model size | — | **54 MB** | 2 MB |

Full per-class recall, confusion matrices, ROC/PR curves and the exact
tuned hyperparameters are in [`reports/metrics_v2.json`](reports/metrics_v2.json)
and `assets/plots/model_training/`.

**Honest caveat:** F1-Macro (0.393) is below the project's original >0.50
target. That target was set before a temporal-leakage issue was found and
fixed (see changelog) — 0.393 is the realistic ceiling for this feature set,
not a tuning failure. Qty=1 orders (54% of the data) are predicted well
(79% recall); Qty=3/4 remain hard to separate from their neighbors and
should get human review before any fully-automated use.

## Evaluation & Explainability (new)

Beyond the headline metrics above, two dedicated phases dig into *whether
the model can be trusted* and *why it predicts what it predicts*:

- **Model Evaluation** (`reports/evaluation_report.json`,
  `assets/plots/evaluation/`): a learning curve (train vs. cross-validation
  F1-Macro as training size grows — the gap is 0.12, i.e. mild, acceptable
  overfitting, not runaway memorization), a calibration curve per class
  (log-loss 0.908), and an accuracy breakdown by department-group that
  found accuracy ranges from **25% to 100%** depending on department —
  a much sharper signal than the single F1-Macro number.
- **Explainability / SHAP** (`reports/shap_feature_importance.json`,
  `assets/plots/explainability/`): SHAP TreeExplainer values on the winning
  Random Forest confirm `product_group` and `product_price_log` are the
  two real drivers of every prediction (not just `product_group` — impurity
  based importance alone overstates how dominant the target-encoded
  features are), with beeswarm plots showing the *direction* of each
  feature's effect and one worked example explaining an actual Qty=3
  prediction feature-by-feature.

## Repository structure

```
├── README.md, PROJECT_CHANGELOG.md, requirements.txt
├── .gitignore, .gitattributes
│
├── data/
│   ├── raw/                        # put Retail_demand_forecasting_dataset.csv here
│   └── processed/                  # each pipeline phase reads/writes here (gitignored — see data/README.md)
│
├── src/pipeline/                   # the 13-phase pipeline, run in order
│   ├── 01_data_cleaning.py
│   ├── 02_outlier_detection_visual.py
│   ├── 03_outlier_detection_statistical.py
│   ├── 04_outlier_treatment.py
│   ├── 05_exploratory_data_analysis.py
│   ├── 06_exploratory_data_analysis_financial.py
│   ├── 07_yeo_johnson_transformation.py
│   ├── 08_feature_engineering.py
│   ├── 09_feature_selection.py
│   ├── 10_model_training.py               # trains + saves the models
│   ├── 11_build_product_department_mapping.py  # builds the app's lookup table
│   ├── 12_model_evaluation.py              # learning curve, calibration, error segmentation
│   └── 13_model_explainability_shap.py     # SHAP global + per-class + worked example
│
├── app/
│   └── app.py                      # Streamlit app — run this
│
├── models/v2/                      # trained pipelines (Git LFS) + the app's lookup table
│   ├── random_forest_pipeline.pkl  (54 MB, winner)
│   ├── xgboost_pipeline.pkl        (2 MB)
│   └── group_id_mapping.json
│
├── reports/
│   ├── Retail_Demand_Forecasting_Project_Report.docx   # full write-up, incl. v2 Sections 10-12
│   ├── Previous_Retail_Demand_Preprocessing_Report.docx
│   ├── metrics_v2.json                     # Phase 10 test-set results
│   ├── evaluation_report.json              # Phase 12 learning curve / calibration / error breakdown
│   ├── shap_feature_importance.json        # Phase 13 global SHAP ranking + worked example
│   └── eda_*.csv                           # small EDA statistical-test outputs
│
├── assets/plots/                   # every diagnostic chart, one folder per phase
│   ├── eda/, feature_engineering/, feature_selection/
│   ├── outlier_visual/, outlier_statistical/, outlier_treatment/
│   ├── yeo_johnson/, model_training/
│   ├── evaluation/                 # learning curve, calibration curve, accuracy-by-department
│   ├── explainability/             # SHAP global importance, beeswarm plots
│   └── ../misc/price_tier_proof.png
│
└── archive/deprecated_v1/          # v1 modeling script + app, kept only for reference/diff
    ├── model_training_v1_deprecated.py     # 2.6GB model, leaky features — do not use
    ├── streamlit_app_v1_deprecated.py      # asks for post-order fields — do not deploy
    └── Modeling_Plots_v1/
```

Every script under `src/pipeline/` and `app/` resolves its paths from its own
location (`__file__`), not from the current working directory — so you can
run any of them from anywhere, e.g. both of these work identically:

```bash
python src/pipeline/10_model_training.py
cd src/pipeline && python 10_model_training.py
```

## Running the pipeline from scratch

```bash
pip install -r requirements.txt

# 1. Put the raw dataset here:
#    data/raw/Retail_demand_forecasting_dataset.csv

# 2. Run phases in order (each writes its output into data/processed/):
python src/pipeline/01_data_cleaning.py
python src/pipeline/02_outlier_detection_visual.py
python src/pipeline/03_outlier_detection_statistical.py
python src/pipeline/04_outlier_treatment.py
python src/pipeline/05_exploratory_data_analysis.py
python src/pipeline/06_exploratory_data_analysis_financial.py
python src/pipeline/07_yeo_johnson_transformation.py
python src/pipeline/08_feature_engineering.py
python src/pipeline/09_feature_selection.py

# 3. Train + save the models (writes reports/metrics_v2.json):
python src/pipeline/10_model_training.py

# 4. Build the app's product/department lookup table:
python src/pipeline/11_build_product_department_mapping.py

# 5. Evaluate the winning model in depth (learning curve, calibration, error segmentation):
python src/pipeline/12_model_evaluation.py

# 6. Explain the winning model's predictions with SHAP:
python src/pipeline/13_model_explainability_shap.py
```

More exhaustive hyperparameter search on stronger hardware:

```bash
MODEL_N_ITER=25 MODEL_CV_FOLDS=5 python src/pipeline/10_model_training.py
```

(The committed `models/v2/*.pkl`, `reports/metrics_v2.json`,
`reports/evaluation_report.json`, and `reports/shap_feature_importance.json`
were already produced this way — you only need to re-run the pipeline if you
want to regenerate everything from your own copy of the raw data.)

## Running the app

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

The app loads `models/v2/*.pkl` and `models/v2/group_id_mapping.json` (the
`.pkl` files are tracked via Git LFS — run `git lfs pull` after cloning if
they look like small text pointers instead of binaries).

`archive/deprecated_v1/streamlit_app_v1_deprecated.py` is the **original,
deprecated** version — kept only for reference/diff purposes. It asks for
post-order fields (delivery status, processing delay, realized profit) that
don't actually exist yet for a real incoming order; don't deploy it. See
`PROJECT_CHANGELOG.md` §3.

## Setting up Git LFS (once, before your first commit)

```bash
git lfs install
git add .gitattributes
git add models/v2/*.pkl
git commit -m "Track v2 model files with Git LFS"
```

## Known limitations

- Hyperparameter search in the committed run used a reduced budget
  (`n_iter=6, cv=3`) to fit this project's build environment (1 CPU core).
  Re-run with a larger budget on stronger hardware for a slightly more
  exhaustive search (see "Running the pipeline from scratch" above).
- `assets/plots/feature_selection/` ships `fs_01`–`fs_03` only. `fs_04`–`fs_07`
  depend on `09_feature_selection.py`'s RFECV step, which genuinely takes
  5-10 minutes to run — they weren't pre-rendered for this delivery, but will
  generate correctly the first time you run that script yourself (see
  `PROJECT_CHANGELOG.md` §9 for the full explanation).
- Qty=3/Qty=4 are the weakest-predicted classes, and accuracy varies from
  25% to 100% by department-group (see `reports/evaluation_report.json`) —
  see `PROJECT_CHANGELOG.md` §10 for a suggested follow-up (ordinal
  classification, or more order-time features).
- This is a planning aid, not an inventory-automation system — pair
  model output with human review for its weaker classes.
