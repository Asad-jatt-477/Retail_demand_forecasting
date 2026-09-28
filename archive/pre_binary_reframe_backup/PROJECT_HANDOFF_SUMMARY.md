# Retail Demand Forecasting — Project Handoff Summary

> Generated as a handoff document for continuing this project in Claude Code (VS Code).
> Covers the full audit/fix history so context isn't lost when switching tools.

---

## 1. Project Context & Goals

**Objective:** Predict `order_item_quantity` (a 5-class target: Qty = 1, 2, 3, 4, or 5) for an
**incoming** retail order, using only information genuinely available at the moment the order
is placed — intended use case is warehouse/inventory planning (fast-pick shelving vs. bulk-zone
picking).

**Dataset:** Retail/logistics order-item-level data (DataCo Supply Chain–style), ~178,396 rows
after cleaning, ~53 raw columns.

**Context:** This is a portfolio project reviewed by the user's mentor ("sir"), alongside other
projects (Hospital Readmission Prediction, a CLV project, and a planned Olist delivery-performance
project). The bar for this project is "professional, defensible, zero-mistakes" — it may be pushed
to GitHub.

**Repository layout (already in place):**
```
data/{raw,processed}/
src/pipeline/
  01_data_cleaning.py
  02_outlier_detection_visual.py
  03_outlier_detection_statistical.py
  04_outlier_treatment.py
  05_exploratory_data_analysis.py
  06_exploratory_data_analysis_financial.py
  07_yeo_johnson_transformation.py
  08_feature_engineering.py
  09_feature_selection.py
  10_model_training.py
  11_build_product_department_mapping.py
  12_model_evaluation.py
  13_model_explainability_shap.py
app/app.py                  (Streamlit prediction app)
models/v2/                  (trained pipelines + product/dept name→group-id mapping)
reports/                    (metrics_v2.json, evaluation_report.json, shap_feature_importance.json, docx reports)
assets/plots/                (one subfolder per phase)
archive/deprecated_v1/       (old, leaky, oversized v1 model + app — reference only)
```

---

## 2. Problems/Issues Identified So Far

### 2.1 Data Leakage (the big recurring theme)

| # | Issue | Status |
|---|---|---|
| A | **Temporal/"future-information" leakage** — 6 post-order columns (`processing_delay_days`, `late_delivery_risk`, `order_status_COMPLETE`, `order_status_PENDING_PAYMENT`, `benefit_per_order_yj`, `order_item_profit_ratio`) were present in the feature-selected dataset and usable by the model, even though none of them exist yet when an order is placed. | ✅ **Fixed** — dropped explicitly at the top of `10_model_training.py` (Step 1). |
| B | **Statistical target-encoding leakage** — `product_avg_qty` / `dept_avg_qty` were literally `groupby(product/dept)[target].mean()` computed on the *whole* dataset — a row could see its own target baked into its own feature. | ✅ **Fixed** — group *identity* (`product_group`/`dept_group`) is recovered via `pd.factorize`, then re-encoded inside the pipeline with a custom **out-of-fold `KFoldTargetEncoder`** (fit only on training folds). |
| C | **Duplicate/near-duplicate feature-combination contamination** (found by investigating "these results look too good") — the 18 final features are coarse/low-cardinality, so 178,396 rows collapse into only ~83,443 unique combinations (72.5% collision rate). A plain row-wise `train_test_split` (even stratified) randomly splits copies of the *same* combination across train and test. Measured: 45.5% of test rows had an exact combo-match in train that saw only one class there — model scored 78% accuracy on those vs. 51% on genuinely novel combos. | ✅ **Fixed** — `GroupShuffleSplit` (outer split) + `StratifiedGroupKFold` (CV during `RandomizedSearchCV`) grouped by the full feature-combination, guaranteeing **zero combo overlap** between train and test. Verified in a real run: `Feature-combinations shared between train and test: 0`. |
| D | **Split-consistency bug** (found while auditing) — `12_model_evaluation.py` and `13_model_explainability_shap.py` still used the *old* `train_test_split(stratify=y)` to "reproduce the Phase 10 split," which no longer matched the new group-aware split — their "test set" would have partly overlapped with the model's actual training data. | ✅ **Fixed** — both scripts updated to use the same `GroupShuffleSplit`/`combo_id` logic as `10_model_training.py`. |
| E | **Algebraic leakage almost introduced during feature experimentation** — testing `order_item_discount` (dollar) + `order_item_discount_rate` (%) + `product_price` together gave a suspicious F1-Macro of 0.88. Verified: 93.8% of rows satisfy `quantity = discount_dollar / (price × discount_rate)` exactly — a pure arithmetic identity, not learning. | ✅ **Caught before implementation, rejected.** Documented as a standing rule: never combine dollar-discount + discount-rate + price in the same feature set. |
| F | **Latent/unresolved leakage-guarding gap in Feature Selection** — `09_feature_selection.py` has an explicit `LEAKY_FEATURES` list, but it only covers the *target-encoding* leakage type (`product_avg_qty`, `category_avg_qty`, `dept_avg_qty`, `is_bulk_category`, `is_single_unit_category`). Only one feature (`is_fulfilled`) is explicitly dropped for *temporal* leakage. Other genuinely post-order features engineered in Phase 8 (`delivery_delay_days`, `is_late_delivery`, `is_digital_payment`) are **not** in the final `FS.csv` today, but this currently appears to be **statistical coincidence** (low MI/importance scores), not a deliberate, comprehensive guard. | 🟡 **Open — was mid-investigation when this handoff was written.** Needs a decision: add these to an explicit leakage-guard list in Phase 9 so future re-runs (with different data/thresholds) can't silently select them. |

### 2.2 Outliers

- Visual (Phase 2) and statistical (Phase 3) outlier detection methods were **verified formula-correct** against standard references (Tukey IQR, Grubbs' test, Generalized ESD per NIST handbook, Modified Z-score per Iglewicz–Hoaglin) — no bugs found.
- Outlier **treatment** (Phase 4) — winsorization (1st/99th percentile capping) and `log1p` transforms are correctly implemented, with a proper negative-value safety assertion before `log1p`. Target variable is correctly *never* outlier-treated.
- **Minor, low-severity note:** winsorization caps and the Yeo-Johnson lambda (Phase 7) are both fit on the *full* dataset before the train/test split exists (Phases 1–9 run before any split). This is a mild, theoretical "global preprocessing" leakage — very low severity (no target variable involved, and cap/lambda values barely change between 80% and 100% of 178K rows) and is standard/accepted practice, but is *not* strictly train-only.

### 2.3 Feature Selection

- **Real, verified finding:** `order_item_discount_rate` (continuous %) was engineered in Phase 8 but **dropped by Phase 9** because, in isolation, its correlation with the target is ≈0.0000. It was screened out by Phase 9's univariate methods (Mutual Information / correlation / RF importance individually), which — a known limitation of filter-based feature selection — **miss features whose value only appears through interaction with another feature** (here, `product_group`). Confirmed via controlled experiment: adding it back raised F1-Macro from 0.393 → **0.467** (every class improved, `Qty=3` recall more than doubled: 0.09 → 0.22). Confirmed *not* leakage (0 correlation with quantity even within a single product).
- **Other candidates tested and rejected** (no meaningful gain, some slightly worse — likely overfitting/noise given the fixed search budget):
  - `category_name` group → redundant with `product_group`/`dept_group` (F1-Macro slightly down, 0.386)
  - `shipping_mode` as full one-hot (replacing the ordinal urgency score) → negligible (+0.002)
  - `customer_zipcode` (leak-safe target-encoded) → worse (0.454)
  - `order_state` (leak-safe target-encoded) → worse (0.459)
  - payment `type` one-hot → worse (0.460)
- **Not yet implemented:** the `order_item_discount_rate` fix has been verified in an isolated experiment only — it has **not yet been added to `08_feature_engineering.py`/`09_feature_selection.py`/`10_model_training.py`** as of this handoff.

### 2.4 Class Imbalance

- **Method used:** Class Weighting (cost-sensitive learning) — `class_weight='balanced'` for Random Forest, `compute_sample_weight('balanced')` passed as `sample_weight` for XGBoost. **SMOTE was never used.**
- **Verified properly resolved**, using multiple independent checks:
  - Per-class recall is non-zero and reasonable for all 5 classes (not just the majority class).
  - Confusion matrix: predicted-Qty=1 column has (near-)zero false positives from other classes.
  - A small synthetic-data test confirmed `sample_weight` really is applied per-fold inside `RandomizedSearchCV` for XGBoost (CV score changes measurably with vs. without the weight array) — ruling out the common "silently ignored fit_param" bug.
- **Remaining weakness is *not* imbalance** — it's that `Qty=2/3/4/5` are already balanced *relative to each other* (~11.4% each) yet still hard to tell apart. That's a feature-signal problem (see §2.3), not a re-balancing problem. (Do **not** add SMOTE — it was considered and explicitly rejected as unnecessary/inappropriate here.)

---

## 3. Phases Reviewed & Decisions Made

A systematic, evidence-based re-audit of Phases 1–10 was started (per the user's request to verify
the *entire* pipeline rather than rebuild from scratch, since no phase has shown evidence of being
fundamentally broken — every issue found so far has been subtle/statistical, not a foundational
error like wrong dtypes, corrupted target, or bad joins).

| Phase | Verdict | Notes |
|---|---|---|
| **1 — Data Cleaning** | ✅ 95% correct | All documented claims verified against raw data (duplicate-column drop, missing-% claims, PII drops, 2018 partial-year drop, zero full-duplicate rows — all confirmed exactly). **Found but not yet fixed:** two more exact-duplicate column pairs were missed: `Sales per customer == Order Item Total` and `Order Item Product Price == Product Price` (both 100% identical, verified). These got redundantly processed through Phases 2–8 (explains why some later charts showed two "different" columns with identical statistics) but were harmlessly filtered out by Feature Selection before reaching the final model — **no impact on final results**, just wasted compute/redundant plots. Low-priority cleanup. |
| **2 — Outlier Detection (Visual)** | ✅ Clean | Standard Tukey IQR formula. No data is written out (diagnostic plots only) — no correctness risk. |
| **3 — Outlier Detection (Statistical)** | ✅ Clean | Grubbs' test, Generalized ESD, and Modified Z-score formulas verified line-by-line against standard references (NIST, Iglewicz–Hoaglin) — all textbook-correct. |
| **4 — Outlier Treatment** | ✅ Mostly clean | Target never treated (correct). `log1p` has a proper negative-value guard. Phase 1's missed duplicates propagate here (5 columns log-transformed, only 2 truly unique — cosmetic, not a correctness bug). Minor "global preprocessing before split" note (see §2.2). |
| **5–6 — EDA** | ✅ Clean | Chi-square test and Cramér's V formula verified correct. Read-only analysis — no data mutation risk. |
| **7 — Yeo-Johnson** | ✅ Clean | Uses scipy's built-in `stats.yeojohnson()` correctly. Same minor "fit before split" note as Phase 4. |
| **8 — Feature Engineering** | 🟡 Audit interrupted mid-way | Binary/derived features spot-checked look logically correct. Found the leakage-guard gap described in §2.1-F (`delivery_delay_days`, `is_late_delivery`, `is_digital_payment` are post-order features that currently don't make it to the final model, but not via an explicit, comprehensive guard). **This phase's audit was not finished** — need to resume checking the remaining engineered features. |
| **9 — Feature Selection** | 🟡 Partially reviewed | `LEAKY_FEATURES` list and the explicit `is_fulfilled` temporal-leakage drop reviewed (see §2.1-F). Already separately confirmed the `order_item_discount_rate` false-negative (§2.3). Full method-by-method review (Mutual Information, RF importance, RFECV, VIF/multicollinearity step) **not yet completed** in this audit pass. |
| **10 — Model Training** | ✅ Already extensively audited/fixed earlier (leakage fixes A, B, C above) | Not yet re-visited in *this* specific re-audit pass, but has already been through multiple rounds of scrutiny and real verification runs (see §2.1). |
| **11–13** | Not yet included in this audit pass | `11_build_product_department_mapping.py` is independent of splitting/training (just builds a name→ID lookup for the app) — low risk. `12`/`13` already had the split-consistency bug found and fixed (§2.1-D). |

**Key philosophy established during this review:** every issue found so far has required *deliberate,
deep investigation* to surface (statistical duplication, interaction-only feature value, algebraic
identities) — none has been a "the foundation is broken" class of bug (wrong data types, corrupted
target, bad merges, etc.). This is evidence the pipeline is fundamentally sound and being refined,
not evidence it needs to be rebuilt from scratch.

---

## 4. Pending Tasks & Immediate Next Steps

### Immediate (resume the interrupted audit)
1. **Finish auditing Phase 8 (Feature Engineering)** — review remaining engineered features beyond the binary flags already spot-checked.
2. **Finish auditing Phase 9 (Feature Selection)** — review the Mutual Information, RF-importance, RFECV, and VIF/multicollinearity steps method-by-method for correctness (not just the leakage-list gap already found).
3. **Decide on and implement a fix for the leakage-guard gap (§2.1-F)** — add `delivery_delay_days`, `is_late_delivery`, `is_digital_payment` (and audit for any others) to an explicit, comprehensive leaky-feature guard in Phase 9, rather than relying on them being statistically screened out by coincidence.

### Feature improvement (verified, not yet implemented)
4. **Add `order_item_discount_rate` as a real feature** through the actual pipeline (not just the isolated test harness):
   - `08_feature_engineering.py`: ensure the raw `order_item_discount_rate` column survives feature engineering.
   - `09_feature_selection.py`: make sure it isn't re-filtered out (the whole point — it was wrongly dropped here originally).
   - `10_model_training.py`: add it to the final feature list; re-run full training (`N_ITER=25, CV_FOLDS=5`) and update `models/v2/*.pkl` + `metrics_v2.json`.
   - Re-run `12_model_evaluation.py` and `13_model_explainability_shap.py` afterward (they depend on the trained model and must use the same split).
   - **Never** add `order_item_discount` (dollar amount) alongside this — confirmed algebraic leakage with price + rate (§2.1-E).
5. **Low-priority cleanup:** drop the two newly-found duplicate column pairs in `01_data_cleaning.py` (`Sales per customer`, `Order Item Product Price`) for cleanliness — confirmed zero impact on final results, so this is cosmetic/efficiency only, not urgent.

### Once the above is done
6. Re-run the full pipeline end-to-end (01 → 13) on the actual project machine to confirm everything is consistent with the fixes above.
7. Update `PROJECT_CHANGELOG.md`, `README.md`, and the Word project report with: the group-aware split fix, the discount-rate feature fix, and the Phase 1–9 audit findings.
8. Consider (optional, discussed but not started): ordinal classification approach for `Qty=2/3/4/5`, since they have a natural order the current plain multi-class classifier ignores.

### Explicitly rejected — do not revisit unless new evidence appears
- **SMOTE** — class imbalance is already correctly handled via class weighting; SMOTE was considered and rejected as unnecessary.
- **`category_name` grouping** — tested, redundant with existing `product_group`/`dept_group`.
- **`order_item_discount` (dollar) combined with rate + price** — confirmed algebraic leakage.
- Full pipeline rebuild-from-scratch — evaluated and rejected; targeted audit-and-fix is the right approach given no foundational issues have been found.
