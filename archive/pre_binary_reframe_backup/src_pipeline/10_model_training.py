# =============================================================================
#  RETAIL DEMAND FORECASTING — MODELING PHASE (v2 — PRODUCTION-SAFE)
#  Phase 10: Random Forest + XGBoost (Multi-Class Classification)
#  Input  : Retail_demand_FS.csv          (178,396 rows × 25 columns)
#  Output : Trained models + evaluation report + plots + metrics.json
#  Target : order_item_quantity  (Classes: 1, 2, 3, 4, 5)
# =============================================================================
#
#  WHAT CHANGED FROM v1 (and why) — see PROJECT_CHANGELOG.md for full detail
#  ─────────────────────────────────────────────────────────────────────────
#  FIX A — TEMPORAL / "FUTURE INFORMATION" LEAKAGE (NEW FIX in v2)
#    v1 kept 6 columns that are only known AFTER an order has shipped or been
#    completed: processing_delay_days, late_delivery_risk, order_status_COMPLETE,
#    order_status_PENDING_PAYMENT, benefit_per_order_yj, order_item_profit_ratio.
#    The business goal is to predict the quantity-class of an INCOMING order —
#    at that moment none of these 6 values exist yet. Training on them made the
#    model look better than it can ever be in production (it was implicitly
#    "cheating" using the future). All 6 are DROPPED here.
#
#  FIX B — RANDOM FOREST MODEL SIZE (NEW FIX in v2)
#    v1's RandomizedSearchCV allowed max_depth=None (fully-grown trees), which
#    produced a 2.6 GB pickle — too large to load reliably in production or in
#    the Streamlit app. max_depth is now capped at 15 and n_estimators capped
#    at 300, which keeps accuracy essentially unchanged (tree-based models
#    saturate well before max_depth=None on tabular data like this) while
#    cutting the model to a deployable size.
#
#  FIX C — SEARCH BUDGET (environment-driven, not a modeling change)
#    This container has 1 CPU core / ~4GB RAM. N_ITER / CV_FOLDS below are
#    set conservatively so the full search finishes in this environment.
#    On a normal multi-core machine, raise N_ITER to 25+ and CV_FOLDS to 5
#    for a slightly more exhaustive search (RESULT_QUALITY_NOTE below).
#
#  FIX D — FEATURE-COLLISION CONTAMINATION (NEW FIX — found from a real
#          training run's suspiciously perfect Qty=1 precision)
#    The 18 modeling features are all coarse/low-cardinality (month, weekday,
#    a handful of booleans, product_group, dept_group, ...), so 178,396 rows
#    collapse into only ~83,000 distinct feature combinations. A plain
#    row-wise train_test_split (even stratified) does not know this, and
#    randomly splits copies of the SAME feature combination across both
#    train and test. Measured on a real run: 45.5% of test rows had an
#    EXACT feature-combo match in train that saw only one class there —
#    the model scored 78% accuracy on those (vs 51% on genuinely novel
#    combos), because it was effectively doing a lookup, not a prediction.
#    This isn't classic future-data leakage (no post-order column is used)
#    but it has the same effect: it inflates the reported metric above what
#    the model can actually do on a genuinely new order. Fixed by grouping
#    rows with GroupShuffleSplit/StratifiedGroupKFold on their full feature
#    combination, so a given combination falls entirely into train OR test —
#    never both, in the outer split or in the CV folds used for tuning.
#
#  CARRIED OVER FROM v1 (already correct, unchanged)
#  ─────────────────────────────────────────────────────────────────────────
#  • Leak-safe K-Fold target encoding for product_group / dept_group
#    (recovered group-identity from product_avg_qty / dept_avg_qty, re-encoded
#    inside the pipeline using training-fold data only).
#  • class_weight='balanced' (RF) + compute_sample_weight('balanced') (XGBoost)
#    for the 54%-vs-~11% class imbalance.
#  • F1-Macro as the tuning/selection metric (not accuracy).
#
#  MODELING PLAN
#  ─────────────────────────────────────────────────────────────────────────
#  STEP 0  : Load FS dataset
#  STEP 1  : Drop temporal/"future information" leakage columns   (FIX A)
#  STEP 2  : Fix statistical target-encoding leakage (group-identity recovery)
#  STEP 3  : Group-Aware Train/Test Split                          (FIX D)
#  STEP 4  : Baseline (DummyClassifier)
#  STEP 5  : Leakage-Safe K-Fold Target Encoder (custom transformer)
#  STEP 6  : Build Pipelines — Random Forest & XGBoost (size-capped)  (FIX B)
#  STEP 7  : Hyperparameter Tuning — RandomizedSearchCV (group-aware CV)  (FIX D)
#  STEP 8  : Final Fit + Test-Set Evaluation (overall AND by-novelty breakdown)
#  STEP 9  : ROC / PR Curves, Feature Importance
#  STEP 10 : Model Comparison + Business Recommendation
#  STEP 11 : Save trained models + machine-readable metrics.json
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
    train_test_split, StratifiedKFold, RandomizedSearchCV,
    GroupShuffleSplit, StratifiedGroupKFold,
)
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score,
    roc_curve, auc, precision_recall_curve, average_precision_score,
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
qty_colors = {1: '#3498DB', 2: '#27AE60', 3: '#F39C12', 4: '#E74C3C', 5: '#8E44AD'}

RANDOM_STATE = 42
TARGET = 'order_item_quantity'
CLASSES = [1, 2, 3, 4, 5]

PLOTS_DIR = os.path.join(ROOT_DIR, 'assets', 'plots', 'model_training')
MODELS_DIR = os.path.join(ROOT_DIR, 'models', 'v2')
REPORTS_DIR = os.path.join(ROOT_DIR, 'reports')
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# === RUN-SPEED CONFIG =========================================================
# See FIX C above. Raise these on a stronger machine for a more exhaustive search.
N_ITER   = int(os.environ.get('MODEL_N_ITER', 25))
CV_FOLDS = int(os.environ.get('MODEL_CV_FOLDS', 5))
# ===============================================================================

print("=" * 70)
print(f"CONFIG → N_ITER={N_ITER}  CV_FOLDS={CV_FOLDS}  (see script header, FIX C)")
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
# STEP 1: DROP TEMPORAL / "FUTURE INFORMATION" LEAKAGE COLUMNS   (FIX A)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 1: DROP POST-ORDER / TEMPORAL-LEAKAGE COLUMNS (FIX A)")
print("=" * 70)

FUTURE_INFO_COLS = [
    'processing_delay_days',       # = shipping_date - order_date -> only known after shipment
    'late_delivery_risk',          # raw dataset's post-delivery outcome flag
    'order_status_COMPLETE',       # order status is only final after fulfillment
    'order_status_PENDING_PAYMENT',
    'benefit_per_order_yj',        # realized profit -> only known after the sale is booked
    'order_item_profit_ratio',     # realized profit ratio -> same issue
]
present = [c for c in FUTURE_INFO_COLS if c in df.columns]
print(f"  Dropping (not available at order-placement time): {present}")
df.drop(columns=present, inplace=True)
print(f"  Remaining columns: {df.shape[1]}")

class_dist = df[TARGET].value_counts().sort_index()
print(f"\n  Target distribution (order_item_quantity):")
for cls, cnt in class_dist.items():
    pct = cnt / len(df) * 100
    print(f"    Qty={cls}  {cnt:>7,}  ({pct:5.1f}%)")

# =============================================================================
# STEP 2: FIX STATISTICAL TARGET-ENCODING LEAKAGE (group-identity recovery)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: FIX STATISTICAL LEAKAGE (product_avg_qty, dept_avg_qty)")
print("=" * 70)

check_prod = df.groupby('product_avg_qty')[TARGET].mean()
max_diff_prod = (check_prod - check_prod.index.to_series()).abs().max()
check_dept = df.groupby('dept_avg_qty')[TARGET].mean()
max_diff_dept = (check_dept - check_dept.index.to_series()).abs().max()
assert max_diff_prod < 1e-6 and max_diff_dept < 1e-6, \
    "Group identity assumption violated — do not proceed with this shortcut."

df['product_group'] = pd.factorize(df['product_avg_qty'])[0]
df['dept_group'] = pd.factorize(df['dept_avg_qty'])[0]
n_prod_groups = df['product_group'].nunique()
n_dept_groups = df['dept_group'].nunique()
print(f"  Recovered {n_prod_groups} product groups, {n_dept_groups} dept groups")

df.drop(columns=['product_avg_qty', 'dept_avg_qty'], inplace=True)
LEAKY_GROUP_COLS = ['product_group', 'dept_group']
print(f"  Will be safely re-encoded from: {LEAKY_GROUP_COLS} (inside CV folds only)")

# =============================================================================
# STEP 3: GROUP-AWARE TRAIN / TEST SPLIT   (FIX D)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: GROUP-AWARE TRAIN / TEST SPLIT (FIX D)")
print("=" * 70)

X = df.drop(columns=[TARGET])
y = df[TARGET]

# FIX D: group rows by their full feature combination so a combination that
# appears more than once (very common with only 18 coarse features) lands
# entirely in train OR entirely in test — never split across both, which is
# what let the model "look up" 45.5% of the test set in an earlier run
# instead of genuinely predicting it. See FIX D in the header for the
# measured before/after numbers.
combo_id = X.groupby(list(X.columns)).ngroup()
n_unique_combos = combo_id.nunique()
print(f"  {len(X):,} rows collapse into {n_unique_combos:,} unique feature combinations "
      f"({len(X) / n_unique_combos:.1f}x average collision rate)")

gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
train_idx, test_idx = next(gss.split(X, y, groups=combo_id))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
combo_id_train = combo_id.iloc[train_idx]

overlap = set(combo_id.iloc[train_idx]) & set(combo_id.iloc[test_idx])
print(f"  Feature-combinations shared between train and test: {len(overlap)} "
      f"(0 expected — confirms the group constraint held)")

print(f"  Train set : {X_train.shape[0]:,} rows  |  Test set : {X_test.shape[0]:,} rows")
print(f"  Final feature count used for training: {X_train.shape[1]}")
print(f"  Features: {list(X_train.columns)}")

label_enc = LabelEncoder()
label_enc.fit(CLASSES)
y_train_xgb = label_enc.transform(y_train)
y_test_xgb = label_enc.transform(y_test)

# GroupShuffleSplit doesn't stratify by class (a "pure" group is 100% one
# class by definition, so perfect stratification and the group constraint
# can't both hold exactly) — print the resulting balance so any drift from
# the true 54/11/11/11/11 split is visible rather than silent.
print("  Class balance after group-aware split (train / test %):")
for cls in CLASSES:
    tr_pct = (y_train == cls).mean() * 100
    te_pct = (y_test == cls).mean() * 100
    print(f"    Qty={cls}:  train {tr_pct:5.1f}%   test {te_pct:5.1f}%")

# =============================================================================
# STEP 4: BASELINE MODEL
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4: BASELINE MODEL (DummyClassifier)")
print("=" * 70)
dummy = DummyClassifier(strategy='stratified', random_state=RANDOM_STATE)
dummy.fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)
dummy_f1 = f1_score(y_test, dummy_pred, average='macro')
dummy_acc = accuracy_score(y_test, dummy_pred)
print(f"  Baseline  ->  F1-Macro: {dummy_f1:.4f}  |  Accuracy: {dummy_acc:.4f}")

# =============================================================================
# STEP 5: LEAKAGE-SAFE K-FOLD TARGET ENCODER
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5: LEAKAGE-SAFE TARGET ENCODER")
print("=" * 70)


class KFoldTargetEncoder(BaseEstimator, TransformerMixin):
    """Out-of-fold target encoder — see FIX in header docstring of v1 for full
    rationale. fit() learns the production mapping on all of X passed to it;
    fit_transform() (training only) builds an out-of-fold version so a row's
    own target never leaks into its own encoded value, even during training."""

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
    ('xgb', XGBClassifier(objective='multi:softprob', num_class=len(CLASSES),
                           eval_metric='mlogloss', random_state=RANDOM_STATE,
                           n_jobs=-1)),
])

# FIX B: max_depth is capped (no more `None` / fully-grown trees) and
# n_estimators is capped at 300 -> keeps the pickled model deployable-size.
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

cv_strategy = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

print("\n  -> Tuning Random Forest ...")
t0 = time.time()
rf_search = RandomizedSearchCV(
    rf_pipeline, param_distributions=rf_param_dist, n_iter=N_ITER,
    scoring='f1_macro', cv=cv_strategy, random_state=RANDOM_STATE,
    n_jobs=1, verbose=1, refit=True,
)
rf_search.fit(X_train, y_train, groups=combo_id_train)
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
xgb_search.fit(X_train, y_train_xgb, groups=combo_id_train,
                xgb__sample_weight=sample_weights_train)
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
print("  (FIX D: the group-aware split in Step 3 guarantees zero feature-combo")
print("   overlap between train and test, so every number below reflects")
print("   genuine generalization to combinations the model never saw — not")
print("   memorization of a duplicate row. No further 'novel-only' correction")
print("   is needed the way it was before this fix.)")

rf_pred = best_rf.predict(X_test)
xgb_pred_enc = best_xgb.predict(X_test)
xgb_pred = label_enc.inverse_transform(xgb_pred_enc)

rf_proba = best_rf.predict_proba(X_test)
xgb_proba = best_xgb.predict_proba(X_test)
rf_class_order = list(best_rf.named_steps['rf'].classes_)
rf_proba = rf_proba[:, [rf_class_order.index(c) for c in CLASSES]]
xgb_class_order = list(label_enc.inverse_transform(best_xgb.named_steps['xgb'].classes_))
xgb_proba = xgb_proba[:, [xgb_class_order.index(c) for c in CLASSES]]

results = {}
reports = {}
for name, pred in [('Random Forest', rf_pred), ('XGBoost', xgb_pred)]:
    f1m = f1_score(y_test, pred, average='macro')
    f1w = f1_score(y_test, pred, average='weighted')
    acc = accuracy_score(y_test, pred)
    per_class_recall = {
        f"qty_{c}": float(r) for c, r in zip(
            CLASSES, [
                (confusion_matrix(y_test, pred, labels=CLASSES)[i, i] /
                 confusion_matrix(y_test, pred, labels=CLASSES)[i, :].sum())
                for i in range(len(CLASSES))
            ]
        )
    }
    results[name] = {'f1_macro': f1m, 'f1_weighted': f1w, 'accuracy': acc,
                      'per_class_recall': per_class_recall}
    reports[name] = classification_report(y_test, pred, target_names=[f'Qty={c}' for c in CLASSES])
    print(f"\n  -- {name} --")
    print(f"    Accuracy: {acc:.4f}  F1-Macro: {f1m:.4f}  F1-Weighted: {f1w:.4f}")
    print(reports[name])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, (name, pred) in zip(axes, [('Random Forest', rf_pred), ('XGBoost', xgb_pred)]):
    cm = confusion_matrix(y_test, pred, labels=CLASSES)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=[f'Qty={c}' for c in CLASSES],
                yticklabels=[f'Qty={c}' for c in CLASSES])
    ax.set_title(f'{name} — Confusion Matrix (v2, leakage-safe)', fontsize=11)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'model_01_confusion_matrices.png'), dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# STEP 9: ROC / PR CURVES + FEATURE IMPORTANCE
# =============================================================================
print("\n" + "=" * 70)
print("STEP 9: ROC / PR CURVES + FEATURE IMPORTANCE")
print("=" * 70)

y_test_bin = label_binarize(y_test, classes=CLASSES)

fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
for ax, (name, proba) in zip(axes, [('Random Forest', rf_proba), ('XGBoost', xgb_proba)]):
    for i, cls in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=qty_colors[cls], lw=2, label=f'Qty={cls} (AUC={roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='grey', lw=1, linestyle=':')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{name} — ROC (One-vs-Rest)', fontsize=11)
    ax.legend(loc='lower right', fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'model_02_roc_curves.png'), dpi=300, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
for ax, (name, proba) in zip(axes, [('Random Forest', rf_proba), ('XGBoost', xgb_proba)]):
    for i, cls in enumerate(CLASSES):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], proba[:, i])
        ap = average_precision_score(y_test_bin[:, i], proba[:, i])
        ax.plot(recall, precision, color=qty_colors[cls], lw=2, label=f'Qty={cls} (AP={ap:.3f})')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title(f'{name} — Precision-Recall (One-vs-Rest)', fontsize=11)
    ax.legend(loc='lower left', fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(0, 1.05)
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
    ax.set_title(f'{name} — Feature Importance (v2)', fontsize=11)
    ax.set_xlabel('Importance'); ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'model_04_feature_importance.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: confusion matrices, ROC, PR curves, feature importance (v2)")

# =============================================================================
# STEP 10: MODEL COMPARISON
# =============================================================================
print("\n" + "=" * 70)
print("STEP 10: FINAL MODEL COMPARISON")
print("=" * 70)
best_model_name = 'Random Forest' if results['Random Forest']['f1_macro'] >= results['XGBoost']['f1_macro'] else 'XGBoost'
best_model_obj = best_rf if best_model_name == 'Random Forest' else best_xgb
print(f"  WINNER (by F1-Macro): {best_model_name}")
print(f"  F1-Macro improvement vs baseline: +{results[best_model_name]['f1_macro'] - dummy_f1:.4f}")

# =============================================================================
# STEP 11: SAVE MODELS + MACHINE-READABLE METRICS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 11: SAVE MODELS + METRICS")
print("=" * 70)

with open(os.path.join(MODELS_DIR, 'random_forest_pipeline.pkl'), 'wb') as f:
    pickle.dump(best_rf, f)
with open(os.path.join(MODELS_DIR, 'xgboost_pipeline.pkl'), 'wb') as f:
    pickle.dump({'pipeline': best_xgb, 'label_encoder': label_enc}, f)

rf_size_mb = os.path.getsize(os.path.join(MODELS_DIR, 'random_forest_pipeline.pkl')) / (1024 ** 2)
xgb_size_mb = os.path.getsize(os.path.join(MODELS_DIR, 'xgboost_pipeline.pkl')) / (1024 ** 2)

metrics_out = {
    'generated_by': '10_model_training.py',
    'random_state': RANDOM_STATE,
    'n_iter': N_ITER,
    'cv_folds': CV_FOLDS,
    'train_rows': int(X_train.shape[0]),
    'test_rows': int(X_test.shape[0]),
    'features_used': feature_names,
    'dropped_future_info_columns': present,
    'group_aware_split': {
        'note': 'FIX D — rows are grouped by their full feature combination so '
                'a combination never appears in both train and test (prevents '
                'the memorization/inflation described in the script header).',
        'unique_feature_combinations': int(n_unique_combos),
        'total_rows': int(len(X)),
        'avg_collision_rate': round(len(X) / n_unique_combos, 2),
        'train_test_combo_overlap': len(overlap),
    },
    'baseline': {'f1_macro': dummy_f1, 'accuracy': dummy_acc},
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
with open(os.path.join(REPORTS_DIR, 'metrics_v2.json'), 'w') as f:
    json.dump(metrics_out, f, indent=2)

print(f"  Saved: {MODELS_DIR}/random_forest_pipeline.pkl  ({rf_size_mb:.1f} MB)")
print(f"  Saved: {MODELS_DIR}/xgboost_pipeline.pkl  ({xgb_size_mb:.1f} MB)")
print(f"  Saved: metrics_v2.json")
print("\n" + "=" * 70)
print("MODELING PHASE v2 COMPLETE")
print("=" * 70)
