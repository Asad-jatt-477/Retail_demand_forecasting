# =============================================================================
#  RETAIL DEMAND FORECASTING — MODELING PHASE (v3 — BINARY, LEAKAGE-SAFE)
#  Phase 10: Random Forest + XGBoost (Binary Classification)
#  Input  : Retail_demand_FS.csv          (178,396 rows x N columns)
#  Output : Trained models + evaluation report + plots + metrics_v3.json
#  Target : is_multi_unit = (order_item_quantity >= 2)
#             0 = Single-unit order (Qty = 1)     -> fast-pick shelf
#             1 = Multi-unit order  (Qty 2..5)    -> bulk-pick zone
# =============================================================================
#
#  WHAT CHANGED FROM v2 (and why) — see PROJECT_CHANGELOG.md for full detail
#  ─────────────────────────────────────────────────────────────────────────
#  FIX E — QUANTITY-DERIVED FEATURE REMOVED (discount_intensity)
#    discount_intensity was a 4-bucket version of the DOLLAR discount. In this
#    dataset  dollar discount = quantity x product_price x discount_rate
#    (94.1% of discounted rows satisfy quantity == discount$ / (price x rate)
#    exactly), so it is only known once the quantity is already decided. It was
#    the ONLY feature that separated Qty 2..5 at all, and it did so through
#    arithmetic, not customer behaviour. With it removed, accuracy on true
#    Qty 2..5 rows falls to chance (0.24 vs 0.25 for a random guess). It is
#    removed upstream (Phase 8 no longer creates it, Phase 9 has an explicit
#    guard) and dropped here again as a safety net against a stale FS file.
#    Adding order_item_discount_rate on top of discount_intensity looked like a
#    gain (F1-Macro 0.385 -> 0.459) but that was the same leak re-opened, not a
#    new signal: without discount_intensity the rate adds nothing (0.346 -> 0.348).
#
#  FIX F — TARGET REFRAMED FROM 5 CLASSES TO 2 (Single vs Multi unit)
#    What the data actually contains (measured, see PROJECT_CHANGELOG.md):
#      * 54 of 118 products are ALWAYS sold as Qty=1 (42.9% of all rows):
#        every product priced above $100, plus a few cheap generic items.
#      * On the other 64 products (all priced <= $100) Qty 1,2,3,4,5 each
#        occur ~20% of the time, for EVERY product, and no available
#        pre-order feature (price, month, region, segment, shipping mode...)
#        predicts which one (chi-square p > 0.05 for all of them; a label-
#        shuffled control scores the same as the model).
#    So "how many units exactly" (Qty 2 vs 3 vs 4 vs 5) is not learnable from
#    what is known when an order is placed, while "one unit or several?" is.
#    That also matches the business use-case: the warehouse decision is fast-
#    pick shelf (single unit) vs bulk-pick zone (multiple units) — it does not
#    need the exact count.
#
#  FIX G — SPLIT: STRATIFIED ROW SPLIT + UNSEEN-PRODUCT STRESS TEST
#    v2's FIX D grouped rows by their full feature combination
#    (GroupShuffleSplit). That was right for 18 coarse features with many
#    order-context noise columns (a model could memorize noisy duplicates), but
#    it depends on HOW MANY features the model uses. Phase 9 now removes the
#    noise features, leaving the model with product identity and price only
#    (~110 distinct inputs). With so few distinct inputs a "feature combination"
#    IS a product, so the group split silently turns into "hold out whole
#    products": on the same lean model F1-Macro swung from 0.45 to 0.85 with the
#    random seed and the test set from 25k to 104k rows, versus a stable
#    0.886-0.887 on a stratified row split. Those two splits answer different
#    questions:
#      * row split      -> "a NEW ORDER for a product we already sell"  (the real
#                          deployment question; repeated inputs are normal and a
#                          114-input model cannot memorize individual rows —
#                          checked: same lean model scores 0.8846 under the
#                          strictest full-context group split and 0.886 here)
#      * product holdout -> "an order for a product NEVER seen in training"
#                          (cold start) — reported separately in Step 8b.
#
#  FROM v2 — KEPT (already correct)
#  ─────────────────────────────────────────────────────────────────────────
#  * FIX A  6 post-order / "future information" columns dropped.
#  * FIX B  Random Forest size-capped (max_depth <= 15, n_estimators <= 300).
#  * Leak-safe out-of-fold target encoding for product_group / dept_group.
#  * Class weighting (class_weight='balanced' / balanced sample_weight) and
#    F1-Macro as the tuning/selection metric. SMOTE is deliberately not used.
#
#  MODELING PLAN
#  ─────────────────────────────────────────────────────────────────────────
#  STEP 0  : Load FS dataset
#  STEP 1  : Drop temporal-leakage + quantity-derived columns      (FIX A, E)
#  STEP 2  : Fix statistical target-encoding leakage (group-identity recovery)
#  STEP 3  : Build binary target + stratified train/test split     (FIX F, G)
#  STEP 4  : Baselines (Dummy + train-only price rule)
#  STEP 5  : Leakage-safe K-Fold target encoder (custom transformer)
#  STEP 6  : Build pipelines — Random Forest & XGBoost (size-capped) (FIX B)
#  STEP 7  : Hyperparameter tuning — RandomizedSearchCV (stratified CV)
#  STEP 8  : Final fit + test-set evaluation (+ 8b: unseen-product stress test)
#  STEP 9  : ROC / PR curves, feature importance
#  STEP 10 : Model comparison + business recommendation
#  STEP 11 : Save trained models + machine-readable metrics_v3.json
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless-safe; no blocking plt.show() windows
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import pickle
import json
import time

from sklearn.model_selection import (
    StratifiedKFold, RandomizedSearchCV, StratifiedGroupKFold, train_test_split,
)
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score,
    balanced_accuracy_score, precision_recall_fscore_support,
    roc_curve, roc_auc_score, precision_recall_curve, average_precision_score,
)
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# ─── PROJECT-ROOT-RELATIVE PATHS (works regardless of current working dir) ────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))      # .../src/pipeline
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))       # project root
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'processed', 'Retail_demand_FS.csv')

palette = {
    'primary': '#2C3E50', 'accent': '#E74C3C', 'highlight': '#3498DB',
    'success': '#27AE60', 'warn': '#F39C12', 'bg': '#FAFAFA',
    'grid': '#ECEFF1', 'purple': '#8E44AD', 'teal': '#16A085',
}
plt.rcParams.update({
    'figure.facecolor': palette['bg'], 'axes.facecolor': palette['bg'],
    'axes.edgecolor': palette['primary'], 'axes.labelcolor': palette['primary'],
    'xtick.color': palette['primary'], 'ytick.color': palette['primary'],
    'grid.color': palette['grid'], 'font.family': 'DejaVu Sans',
    'axes.titleweight': 'bold', 'axes.titlesize': 11,
})

RANDOM_STATE = 42
TARGET = 'order_item_quantity'          # raw column in the FS dataset
BIN_TARGET = 'is_multi_unit'            # derived here: 1 if Qty >= 2 else 0
CLASSES = [0, 1]
CLASS_NAMES = ['Single-unit (Qty=1)', 'Multi-unit (Qty 2-5)']
CLASS_KEYS = ['single_unit', 'multi_unit']

PLOTS_DIR = os.path.join(ROOT_DIR, 'assets', 'plots', 'model_training')
MODELS_DIR = os.path.join(ROOT_DIR, 'models', 'v3')
REPORTS_DIR = os.path.join(ROOT_DIR, 'reports')
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# === RUN-SPEED CONFIG =========================================================
# Raise these on a stronger machine for a more exhaustive search.
N_ITER   = int(os.environ.get('MODEL_N_ITER', 25))
CV_FOLDS = int(os.environ.get('MODEL_CV_FOLDS', 5))
# ===============================================================================

print("=" * 70)
print(f"CONFIG -> N_ITER={N_ITER}  CV_FOLDS={CV_FOLDS}")
print("=" * 70)

# =============================================================================
# STEP 0: LOAD DATASET
# =============================================================================
print("\n" + "=" * 70)
print("STEP 0: LOAD DATASET")
print("=" * 70)
df = pd.read_csv(DATA_PATH)
print(f"  Dataset loaded : {df.shape[0]:,} rows x {df.shape[1]} columns")

# =============================================================================
# STEP 1: DROP TEMPORAL-LEAKAGE + QUANTITY-DERIVED COLUMNS   (FIX A, FIX E)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 1: DROP POST-ORDER / QUANTITY-DERIVED COLUMNS (FIX A, FIX E)")
print("=" * 70)

FUTURE_INFO_COLS = [
    'processing_delay_days',       # = shipping_date - order_date -> only known after shipment
    'late_delivery_risk',          # raw dataset's post-delivery outcome flag
    'order_status_COMPLETE',       # order status is only final after fulfillment
    'order_status_PENDING_PAYMENT',
    'benefit_per_order_yj',        # realized profit -> only known after the sale is booked
    'order_item_profit_ratio',     # realized profit ratio -> same issue
    'shipping_year', 'shipping_month',   # year / month of the SHIPPING date -> not known at order time
]
# FIX E: features computed from a dollar amount that already contains the
# quantity (dollar discount = quantity x price x rate). Should no longer be in
# the FS file at all (Phase 8/9 removed it) — kept here as a safety net.
QUANTITY_DERIVED_COLS = ['discount_intensity']

present = [c for c in FUTURE_INFO_COLS + QUANTITY_DERIVED_COLS if c in df.columns]
print(f"  Dropping (not available at order-placement time / derived from quantity): {present}")
df.drop(columns=present, inplace=True)
print(f"  Remaining columns: {df.shape[1]}")

class_dist = df[TARGET].value_counts().sort_index()
print(f"\n  Raw quantity distribution (for reference):")
for cls, cnt in class_dist.items():
    print(f"    Qty={cls}  {cnt:>7,}  ({cnt / len(df) * 100:5.1f}%)")

# =============================================================================
# STEP 2: FIX STATISTICAL TARGET-ENCODING LEAKAGE (group-identity recovery)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: FIX STATISTICAL LEAKAGE (product_avg_qty [, dept_avg_qty])")
print("=" * 70)

# product_avg_qty / dept_avg_qty are per-group means of the target. What the model
# may use is only the group IDENTITY (recovered here via factorize) — the value is
# re-encoded out-of-fold inside the pipeline. dept_avg_qty is optional: Phase 9
# drops it when the department adds nothing beyond the product (each product
# belongs to exactly one department), which is what the data shows.
assert 'product_avg_qty' in df.columns, "Phase 9 must keep the product identity feature."
GROUP_SOURCES = {'product_group': 'product_avg_qty', 'dept_group': 'dept_avg_qty'}
LEAKY_GROUP_COLS = []
for _g, _src in GROUP_SOURCES.items():
    if _src not in df.columns:
        print(f"  {_src} not in the feature set (dropped by Phase 9) — skipped")
        continue
    _chk = df.groupby(_src)[TARGET].mean()
    assert (_chk - _chk.index.to_series()).abs().max() < 1e-6, \
        "Group identity assumption violated — do not proceed with this shortcut."
    df[_g] = pd.factorize(df[_src])[0]
    df.drop(columns=[_src], inplace=True)
    LEAKY_GROUP_COLS.append(_g)
    print(f"  Recovered {df[_g].nunique()} groups for {_g}")
print(f"  Will be safely re-encoded from: {LEAKY_GROUP_COLS} (inside CV folds only)")

# =============================================================================
# STEP 3: BINARY TARGET + STRATIFIED TRAIN / TEST SPLIT   (FIX F, FIX G)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: BINARY TARGET + STRATIFIED TRAIN / TEST SPLIT (FIX F, FIX G)")
print("=" * 70)

X = df.drop(columns=[TARGET])
y_qty = df[TARGET]
y = (y_qty >= 2).astype(int).rename(BIN_TARGET)

print(f"  Target '{BIN_TARGET}':  0 = Qty=1 (single-unit)   1 = Qty 2..5 (multi-unit)")
for cls, name in zip(CLASSES, CLASS_NAMES):
    print(f"    {cls} = {name:<22} {int((y == cls).sum()):>8,}  ({(y == cls).mean() * 100:5.1f}%)")

# FIX G: stratified random row split (see the header for why the v2 group-by-
# feature-combination split is no longer used). How many distinct model inputs
# there are, and how often a test row's exact input also occurs in train, is
# printed so the reader can see the model has no room to memorize rows.
combo_id = X.groupby(list(X.columns)).ngroup()
n_unique_combos = combo_id.nunique()
print(f"\n  {len(X):,} rows collapse into {n_unique_combos:,} distinct model inputs "
      f"({len(X) / n_unique_combos:.1f}x average repeat rate)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
test_combo_seen = float(combo_id.loc[X_test.index].isin(set(combo_id.loc[X_train.index])).mean())
print(f"  Test rows whose exact input also occurs in train: {test_combo_seen * 100:.1f}% "
      f"(expected for a repeat-purchase catalog; the model has only "
      f"{n_unique_combos:,} distinct inputs to learn from)")

print(f"  Train set : {X_train.shape[0]:,} rows  |  Test set : {X_test.shape[0]:,} rows")
print(f"  Final feature count used for training: {X_train.shape[1]}")
print(f"  Features: {list(X_train.columns)}")

print("  Class balance after stratified split (train / test %):")
for cls, name in zip(CLASSES, CLASS_NAMES):
    tr_pct = (y_train == cls).mean() * 100
    te_pct = (y_test == cls).mean() * 100
    print(f"    {name:<22} train {tr_pct:5.1f}%   test {te_pct:5.1f}%")


def summarize(y_true, pred, proba=None):
    """One place that computes every reported metric, so baselines and models
    are always scored identically. proba = P(multi-unit) or None."""
    prec, rec, f1c, sup = precision_recall_fscore_support(
        y_true, pred, labels=CLASSES, zero_division=0)
    out = {
        'f1_macro': float(f1_score(y_true, pred, average='macro')),
        'f1_weighted': float(f1_score(y_true, pred, average='weighted')),
        'accuracy': float(accuracy_score(y_true, pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, pred)),
        'per_class': {
            key: {'precision': float(p), 'recall': float(r), 'f1': float(f), 'support': int(s)}
            for key, p, r, f, s in zip(CLASS_KEYS, prec, rec, f1c, sup)
        },
    }
    if proba is not None:
        out['roc_auc'] = float(roc_auc_score(y_true, proba))
        out['pr_auc_multi'] = float(average_precision_score(y_true, proba))
    return out


# =============================================================================
# STEP 4: BASELINES
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4: BASELINES (Dummy + train-only price rule)")
print("=" * 70)

dummy = DummyClassifier(strategy='stratified', random_state=RANDOM_STATE)
dummy.fit(X_train, y_train)
dummy_res = summarize(y_test, dummy.predict(X_test), dummy.predict_proba(X_test)[:, 1])
print(f"  Dummy (random, class-frequency) ->  F1-Macro: {dummy_res['f1_macro']:.4f}  "
      f"|  Accuracy: {dummy_res['accuracy']:.4f}")

# Simple business rule a planner could apply by hand, learned from TRAIN ONLY:
# "multi-unit orders were only ever seen up to price X in training; anything
# priced above X is a single-unit order, everything else is multi-unit."
# The ML models must beat THIS to show they add value over a price threshold.
max_multi_price_log = float(X_train.loc[y_train == 1, 'product_price_log'].max())
rule_pred = (X_test['product_price_log'] <= max_multi_price_log).astype(int).values
rule_res = summarize(y_test, rule_pred)
print(f"  Price rule (train-only: multi-unit only up to ${np.expm1(max_multi_price_log):,.2f}) "
      f"->  F1-Macro: {rule_res['f1_macro']:.4f}  |  Accuracy: {rule_res['accuracy']:.4f}")

# =============================================================================
# STEP 5: LEAKAGE-SAFE K-FOLD TARGET ENCODER
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5: LEAKAGE-SAFE TARGET ENCODER")
print("=" * 70)


class KFoldTargetEncoder(BaseEstimator, TransformerMixin):
    """Out-of-fold target encoder. fit() learns the production mapping on all
    of X passed to it; fit_transform() (training only) builds an out-of-fold
    version so a row's own target never leaks into its own encoded value, even
    during training. With the binary target the encoded value is the smoothed
    P(multi-unit | group)."""

    def __init__(self, cols, n_splits=5, smoothing=10, random_state=42):
        self.cols = cols
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.random_state = random_state

    def _smoothed_means(self, group_col, y):
        agg = pd.DataFrame({'g': group_col.values, 't': np.asarray(y)})
        agg = agg.groupby('g')['t'].agg(['mean', 'count'])
        global_mean = float(np.mean(y))
        smoothed = (agg['mean'] * agg['count'] + global_mean * self.smoothing) \
            / (agg['count'] + self.smoothing)
        return smoothed, global_mean

    def fit(self, X, y):
        X = X.reset_index(drop=True)
        y = pd.Series(np.asarray(y)).reset_index(drop=True)
        self.mapping_ = {}
        self.global_mean_ = float(y.mean())
        for col in self.cols:
            smoothed, _ = self._smoothed_means(X[col], y)
            self.mapping_[col] = smoothed
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            X[col] = X[col].map(self.mapping_[col]).fillna(self.global_mean_)
        return X

    def fit_transform(self, X, y=None):
        X_orig = X.reset_index(drop=True).copy()
        y_ser = pd.Series(np.asarray(y)).reset_index(drop=True)
        self.fit(X_orig, y_ser)
        X_out = X_orig.copy()
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                               random_state=self.random_state)
        for col in self.cols:
            oof = np.zeros(len(X_orig))
            for tr_idx, val_idx in skf.split(X_orig, y_ser):
                smoothed, gmean = self._smoothed_means(
                    X_orig[col].iloc[tr_idx], y_ser.iloc[tr_idx]
                )
                oof[val_idx] = X_orig[col].iloc[val_idx].map(smoothed).fillna(gmean).values
            X_out[col] = oof
        return X_out


print("  KFoldTargetEncoder ready.")

# =============================================================================
# STEP 6: CLASS IMBALANCE + BUILD PIPELINES (RF size-capped — FIX B)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 6: PIPELINES (class-imbalance aware, size-capped RF)")
print("=" * 70)

sample_weights_train = compute_sample_weight(class_weight='balanced', y=y_train)

rf_pipeline = Pipeline([
    ('target_enc', KFoldTargetEncoder(cols=LEAKY_GROUP_COLS, n_splits=5,
                                       smoothing=10, random_state=RANDOM_STATE)),
    ('rf', RandomForestClassifier(class_weight='balanced',
                                   random_state=RANDOM_STATE, n_jobs=-1)),
])
xgb_pipeline = Pipeline([
    ('target_enc', KFoldTargetEncoder(cols=LEAKY_GROUP_COLS, n_splits=5,
                                       smoothing=10, random_state=RANDOM_STATE)),
    ('xgb', XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                           random_state=RANDOM_STATE, n_jobs=-1)),
])

# FIX B: max_depth is capped (no `None` / fully-grown trees) and n_estimators
# is capped at 300 -> keeps the pickled model deployable-size.
rf_param_dist = {
    'rf__n_estimators':      [100, 150, 200, 300],
    'rf__max_depth':         [8, 10, 12, 15],
    'rf__min_samples_split': [5, 10],
    'rf__min_samples_leaf':  [2, 4],
    'rf__max_features':      ['sqrt'],
}
xgb_param_dist = {
    'xgb__n_estimators':     [150, 200, 300],
    'xgb__max_depth':        [3, 4, 5, 6],
    'xgb__learning_rate':    [0.03, 0.05, 0.1],
    'xgb__subsample':        [0.7, 0.8, 0.9],
    'xgb__colsample_bytree': [0.7, 0.8, 1.0],
    'xgb__min_child_weight': [1, 3, 5],
}

# =============================================================================
# STEP 7: HYPERPARAMETER TUNING
# =============================================================================
print("\n" + "=" * 70)
print(f"STEP 7: HYPERPARAMETER TUNING (n_iter={N_ITER}, cv={CV_FOLDS}, scoring=F1-Macro)")
print("=" * 70)

cv_strategy = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

print("\n  -> Tuning Random Forest ...")
t0 = time.time()
rf_search = RandomizedSearchCV(
    rf_pipeline, param_distributions=rf_param_dist, n_iter=N_ITER,
    scoring='f1_macro', cv=cv_strategy, random_state=RANDOM_STATE,
    n_jobs=1, verbose=1, refit=True,
)
rf_search.fit(X_train, y_train)
rf_tune_time = time.time() - t0
print(f"  RF tuned in {rf_tune_time:.1f}s | Best CV F1-Macro: {rf_search.best_score_:.4f}")
print(f"  Best params: {rf_search.best_params_}")

print("\n  -> Tuning XGBoost ...")
t0 = time.time()
xgb_search = RandomizedSearchCV(
    xgb_pipeline, param_distributions=xgb_param_dist, n_iter=N_ITER,
    scoring='f1_macro', cv=cv_strategy, random_state=RANDOM_STATE,
    n_jobs=1, verbose=1, refit=True,
)
xgb_search.fit(X_train, y_train, xgb__sample_weight=sample_weights_train)
xgb_tune_time = time.time() - t0
print(f"  XGB tuned in {xgb_tune_time:.1f}s | Best CV F1-Macro: {xgb_search.best_score_:.4f}")
print(f"  Best params: {xgb_search.best_params_}")

best_rf = rf_search.best_estimator_
best_xgb = xgb_search.best_estimator_

# =============================================================================
# STEP 8: TEST-SET EVALUATION
# =============================================================================
print("\n" + "=" * 70)
print("STEP 8: TEST-SET EVALUATION")
print("=" * 70)
print("  (Stratified row split: the question answered is \"a NEW ORDER for a product we")
print("   already sell\". The model has only a few hundred distinct inputs, so it cannot")
print("   memorize rows. Step 8b separately tests products the model has never seen.)")

rf_classes = list(best_rf.named_steps['rf'].classes_)
xgb_classes = list(best_xgb.named_steps['xgb'].classes_)
assert rf_classes == CLASSES and xgb_classes == CLASSES, "Unexpected class order"

rf_proba = best_rf.predict_proba(X_test)[:, 1]
xgb_proba = best_xgb.predict_proba(X_test)[:, 1]
rf_pred = best_rf.predict(X_test)
xgb_pred = best_xgb.predict(X_test)

results = {
    'Random Forest': summarize(y_test, rf_pred, rf_proba),
    'XGBoost': summarize(y_test, xgb_pred, xgb_proba),
}
for name, pred in [('Random Forest', rf_pred), ('XGBoost', xgb_pred)]:
    r = results[name]
    print(f"\n  -- {name} --")
    print(f"    Accuracy: {r['accuracy']:.4f}  F1-Macro: {r['f1_macro']:.4f}  "
          f"Balanced-Acc: {r['balanced_accuracy']:.4f}  ROC-AUC: {r['roc_auc']:.4f}  "
          f"PR-AUC(multi): {r['pr_auc_multi']:.4f}")
    print(classification_report(y_test, pred, target_names=CLASS_NAMES, digits=3))

fig, axes = plt.subplots(1, 3, figsize=(19, 5.8))
for ax, (name, pred) in zip(axes, [('Random Forest', rf_pred), ('XGBoost', xgb_pred),
                                    ('Price rule (baseline)', rule_pred)]):
    cm = confusion_matrix(y_test, pred, labels=CLASSES)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=['Single', 'Multi'], yticklabels=['Single', 'Multi'])
    ax.set_title(f'{name} — Confusion Matrix', fontsize=11)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'model_01_confusion_matrices.png'), dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# STEP 8b: STRESS TEST — PRODUCTS THE MODEL HAS NEVER SEEN (cold start)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 8b: STRESS TEST — WHOLE PRODUCTS HELD OUT (5 folds)")
print("=" * 70)
print("  Each fold trains on ~80% of the products and tests on the other ~20% — the")
print("  question is \"an order for a product that was never in training\" (new SKU).")

prod_cols = [c for c in ['product_group', 'product_price_log'] if c in X.columns]
product_key = X.groupby(prod_cols).ngroup()
n_products = int(product_key.nunique())
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
stress = {'Random Forest': [], 'XGBoost': []}
for tr_i, te_i in sgkf.split(X, y, groups=product_key):
    Xtr, Xte = X.iloc[tr_i], X.iloc[te_i]
    ytr, yte = y.iloc[tr_i], y.iloc[te_i]
    rf_fold = clone(best_rf).fit(Xtr, ytr)
    xgb_fold = clone(best_xgb).fit(Xtr, ytr, xgb__sample_weight=compute_sample_weight('balanced', ytr))
    stress['Random Forest'].append(float(f1_score(yte, rf_fold.predict(Xte), average='macro')))
    stress['XGBoost'].append(float(f1_score(yte, xgb_fold.predict(Xte), average='macro')))

stress_out = {'n_products': n_products, 'folds': 5, 'grouped_by': prod_cols}
for name, scores in stress.items():
    arr = np.array(scores)
    stress_out[name] = {'f1_macro_per_fold': [round(s, 4) for s in scores],
                        'mean': float(arr.mean()), 'min': float(arr.min()), 'max': float(arr.max())}
    print(f"  {name:<14} F1-Macro per fold: {[round(s, 3) for s in scores]}  "
          f"mean={arr.mean():.3f}  min={arr.min():.3f}  max={arr.max():.3f}")
print(f"  ({n_products} distinct products, identified by {prod_cols}.)")
print("  Read with care: the score swings a lot between folds because a few products carry")
print("  tens of thousands of rows each, and product_group ids were derived from a whole-")
print("  dataset target mean (all always-single products share one id), which makes this")
print("  test optimistic for those products. Treat it as a warning about new-SKU use, not")
print("  as a precise number.")

# =============================================================================
# STEP 9: ROC / PR CURVES + FEATURE IMPORTANCE
# =============================================================================
print("\n" + "=" * 70)
print("STEP 9: ROC / PR CURVES + FEATURE IMPORTANCE")
print("=" * 70)

model_colors = {'Random Forest': palette['highlight'], 'XGBoost': palette['accent']}
model_probas = {'Random Forest': rf_proba, 'XGBoost': xgb_proba}

fig, ax = plt.subplots(figsize=(7.5, 6.5))
for name, proba in model_probas.items():
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax.plot(fpr, tpr, color=model_colors[name], lw=2,
            label=f"{name} (AUC={results[name]['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], color='grey', lw=1, linestyle=':', label='Random guess')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC — Multi-unit vs Single-unit', fontsize=11)
ax.legend(loc='lower right', fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'model_02_roc_curves.png'), dpi=300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(7.5, 6.5))
for name, proba in model_probas.items():
    precision, recall, _ = precision_recall_curve(y_test, proba)
    ax.plot(recall, precision, color=model_colors[name], lw=2,
            label=f"{name} (AP={results[name]['pr_auc_multi']:.3f})")
ax.axhline(float(y_test.mean()), color='grey', lw=1, linestyle=':',
           label=f'Prevalence of multi-unit ({y_test.mean():.2f})')
ax.set_xlabel('Recall (multi-unit)'); ax.set_ylabel('Precision (multi-unit)')
ax.set_title('Precision-Recall — Multi-unit class', fontsize=11)
ax.legend(loc='lower left', fontsize=9); ax.grid(alpha=0.3); ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'model_03_pr_curves.png'), dpi=300, bbox_inches='tight')
plt.close()

feature_names = list(X_train.columns)
rf_importances = best_rf.named_steps['rf'].feature_importances_
xgb_importances = best_xgb.named_steps['xgb'].feature_importances_
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
for ax, (name, importances) in zip(axes, [('Random Forest', rf_importances), ('XGBoost', xgb_importances)]):
    imp_series = pd.Series(importances, index=feature_names).sort_values(ascending=True)
    colors = [palette['warn'] if 'group' in f else palette['teal'] for f in imp_series.index]
    ax.barh(imp_series.index, imp_series.values, color=colors, edgecolor='white')
    ax.set_title(f'{name} — Feature Importance (v3)', fontsize=11)
    ax.set_xlabel('Importance'); ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'model_04_feature_importance.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: confusion matrices, ROC, PR curves, feature importance (v3)")

# =============================================================================
# STEP 10: MODEL COMPARISON
# =============================================================================
print("\n" + "=" * 70)
print("STEP 10: FINAL MODEL COMPARISON")
print("=" * 70)

comparison = {
    'Dummy (random)':        dummy_res,
    'Price rule (train-only)': rule_res,
    'Random Forest':         results['Random Forest'],
    'XGBoost':               results['XGBoost'],
}
print(f"  {'Model':<26}{'F1-Macro':>10}{'Accuracy':>10}{'Recall(Single)':>16}{'Recall(Multi)':>15}")
for name, r in comparison.items():
    print(f"  {name:<26}{r['f1_macro']:>10.4f}{r['accuracy']:>10.4f}"
          f"{r['per_class']['single_unit']['recall']:>16.4f}"
          f"{r['per_class']['multi_unit']['recall']:>15.4f}")

best_model_name = 'Random Forest' if results['Random Forest']['f1_macro'] >= results['XGBoost']['f1_macro'] else 'XGBoost'
best_f1 = results[best_model_name]['f1_macro']
print(f"\n  WINNER (by F1-Macro): {best_model_name}")
print(f"  F1-Macro vs random baseline   : {best_f1 - dummy_res['f1_macro']:+.4f}")
print(f"  F1-Macro vs train-only price rule: {best_f1 - rule_res['f1_macro']:+.4f}  "
      f"(how much the ML model adds over a simple price threshold)")

# =============================================================================
# STEP 11: SAVE MODELS + MACHINE-READABLE METRICS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 11: SAVE MODELS + METRICS")
print("=" * 70)

with open(os.path.join(MODELS_DIR, 'random_forest_pipeline.pkl'), 'wb') as f:
    pickle.dump(best_rf, f)
with open(os.path.join(MODELS_DIR, 'xgboost_pipeline.pkl'), 'wb') as f:
    pickle.dump({'pipeline': best_xgb}, f)

rf_size_mb = os.path.getsize(os.path.join(MODELS_DIR, 'random_forest_pipeline.pkl')) / (1024 ** 2)
xgb_size_mb = os.path.getsize(os.path.join(MODELS_DIR, 'xgboost_pipeline.pkl')) / (1024 ** 2)

metrics_out = {
    'generated_by': '10_model_training.py',
    'problem': 'binary: is_multi_unit = (order_item_quantity >= 2); 0 = single-unit (fast-pick), 1 = multi-unit (bulk zone)',
    'class_names': CLASS_NAMES,
    'random_state': RANDOM_STATE,
    'n_iter': N_ITER,
    'cv_folds': CV_FOLDS,
    'train_rows': int(X_train.shape[0]),
    'test_rows': int(X_test.shape[0]),
    'test_multi_unit_share': float(y_test.mean()),
    'features_used': feature_names,
    'dropped_future_info_columns': [c for c in present if c in FUTURE_INFO_COLS],
    'dropped_quantity_derived_columns': [c for c in present if c in QUANTITY_DERIVED_COLS],
    'split': {
        'type': 'stratified random row split, 80/20 (FIX G)',
        'note': 'Answers "a new order for a product we already sell". The model has only a '
                'few hundred distinct inputs so it cannot memorize rows; see the script '
                'header (FIX G) and stress_test_unseen_products for the cold-start question.',
        'distinct_model_inputs': int(n_unique_combos),
        'total_rows': int(len(X)),
        'avg_repeat_rate': round(len(X) / n_unique_combos, 2),
        'test_rows_with_exact_input_seen_in_train': round(test_combo_seen, 4),
    },
    'stress_test_unseen_products': stress_out,
    'baseline_dummy': dummy_res,
    'baseline_price_rule': {
        **rule_res,
        'rule': 'predict single-unit when product price is above the highest price at which '
                'a multi-unit order was seen in TRAIN; otherwise multi-unit',
        'train_max_multi_unit_price': float(np.expm1(max_multi_price_log)),
    },
    'random_forest': {
        **results['Random Forest'],
        'best_params': rf_search.best_params_,
        'cv_best_f1_macro': rf_search.best_score_,
        'tuning_seconds': rf_tune_time,
        'model_size_mb': round(rf_size_mb, 1),
    },
    'xgboost': {
        **results['XGBoost'],
        'best_params': xgb_search.best_params_,
        'cv_best_f1_macro': xgb_search.best_score_,
        'tuning_seconds': xgb_tune_time,
        'model_size_mb': round(xgb_size_mb, 1),
    },
    'winner': best_model_name,
}
with open(os.path.join(REPORTS_DIR, 'metrics_v3.json'), 'w') as f:
    json.dump(metrics_out, f, indent=2)

print(f"  Saved: {MODELS_DIR}/random_forest_pipeline.pkl  ({rf_size_mb:.1f} MB)")
print(f"  Saved: {MODELS_DIR}/xgboost_pipeline.pkl  ({xgb_size_mb:.1f} MB)")
print(f"  Saved: metrics_v3.json")
print("\n" + "=" * 70)
print("MODELING PHASE v3 COMPLETE")
print("=" * 70)
