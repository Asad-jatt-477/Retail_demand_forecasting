# =============================================================================
#  RETAIL DEMAND FORECASTING — PHASE 12: MODEL EVALUATION
#  Input  : data/processed/Retail_demand_FS.csv, models/v2/*.pkl
#  Output : reports/evaluation_report.json + assets/plots/evaluation/*.png
#
#  WHY A SEPARATE PHASE
#  ─────────────────────────────────────────────────────────────────────────
#  10_model_training.py reports test-set accuracy/F1/confusion matrices —
#  enough to pick a winner, but not enough to trust it in production. This
#  phase asks three questions Phase 10 doesn't answer:
#
#  1. Is the winning model OVERFITTING? (learning curve: train vs
#     cross-validation score as training-set size grows)
#  2. Are its predicted PROBABILITIES trustworthy? (calibration curve — when
#     the model says "70% confident", is it right about 70% of the time?)
#  3. WHERE does it fail? (error breakdown by product/department group,
#     not just by quantity class)
#
#  This script re-derives the exact train/test split used in
#  10_model_training.py (same random_state, same dropped columns, same
#  group-id recovery, same GroupShuffleSplit — FIX D) so "test set" here
#  means the same held-out rows the saved model has genuinely never seen.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless-safe; no blocking plt.show() windows
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, learning_curve, GroupShuffleSplit,
)
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss, f1_score, precision_recall_fscore_support
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# ─── PROJECT-ROOT-RELATIVE PATHS ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'processed', 'Retail_demand_FS.csv')
MODELS_DIR = os.path.join(ROOT_DIR, 'models', 'v2')
REPORTS_DIR = os.path.join(ROOT_DIR, 'reports')
PLOTS_DIR = os.path.join(ROOT_DIR, 'assets', 'plots', 'evaluation')
os.makedirs(PLOTS_DIR, exist_ok=True)

RANDOM_STATE = 42
TARGET = 'order_item_quantity'
CLASSES = [1, 2, 3, 4, 5]
FUTURE_INFO_COLS = [
    'processing_delay_days', 'late_delivery_risk', 'order_status_COMPLETE',
    'order_status_PENDING_PAYMENT', 'benefit_per_order_yj', 'order_item_profit_ratio',
]


class KFoldTargetEncoder(BaseEstimator, TransformerMixin):
    """Must match 10_model_training.py exactly to unpickle the saved pipelines."""

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
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for col in self.cols:
            oof = np.zeros(len(X_orig))
            for tr_idx, val_idx in skf.split(X_orig, y_ser):
                smoothed, gmean = self._smoothed_means(X_orig[col].iloc[tr_idx], y_ser.iloc[tr_idx])
                oof[val_idx] = X_orig[col].iloc[val_idx].map(smoothed).fillna(gmean).values
            X_out[col] = oof
        return X_out


print("=" * 70)
print("PHASE 12: MODEL EVALUATION")
print("=" * 70)

# ── Reproduce the exact Phase 10 train/test split (FIX D: group-aware) ────────
df = pd.read_csv(DATA_PATH)
present = [c for c in FUTURE_INFO_COLS if c in df.columns]
df.drop(columns=present, inplace=True)

df['product_group'] = pd.factorize(df['product_avg_qty'])[0]
df['dept_group'] = pd.factorize(df['dept_avg_qty'])[0]
df.drop(columns=['product_avg_qty', 'dept_avg_qty'], inplace=True)

X = df.drop(columns=[TARGET])
y = df[TARGET]

# Must match 10_model_training.py's Step 3 exactly, or this "test set" would
# include rows the saved model was actually trained on.
combo_id = X.groupby(list(X.columns)).ngroup()
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
train_idx, test_idx = next(gss.split(X, y, groups=combo_id))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

label_enc = LabelEncoder().fit(CLASSES)
y_test_xgb = label_enc.transform(y_test)
print(f"  Reproduced split: {X_train.shape[0]:,} train / {X_test.shape[0]:,} test rows "
      f"(must match 10_model_training.py's numbers)")

# ── Load trained pipelines ─────────────────────────────────────────────────────
with open(os.path.join(MODELS_DIR, 'random_forest_pipeline.pkl'), 'rb') as f:
    best_rf = pickle.load(f)
with open(os.path.join(MODELS_DIR, 'xgboost_pipeline.pkl'), 'rb') as f:
    xgb_bundle = pickle.load(f)
    best_xgb = xgb_bundle['pipeline']

with open(os.path.join(REPORTS_DIR, 'metrics_v2.json')) as f:
    prior_metrics = json.load(f)

# =============================================================================
# 1. LEARNING CURVE (winner model only — Random Forest, per metrics_v2.json)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 1: LEARNING CURVE (is the winner overfitting?)")
print("=" * 70)

rf_best_params = {k.replace('rf__', ''): v for k, v in prior_metrics['random_forest']['best_params'].items()}
lc_pipeline = Pipeline([
    ('target_enc', KFoldTargetEncoder(cols=['product_group', 'dept_group'], n_splits=5,
                                       smoothing=10, random_state=RANDOM_STATE)),
    ('rf', RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE,
                                   n_jobs=-1, **rf_best_params)),
])

train_sizes, train_scores, val_scores = learning_curve(
    lc_pipeline, X_train, y_train,
    train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0],
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
    scoring='f1_macro', n_jobs=1, random_state=RANDOM_STATE,
)
train_mean, val_mean = train_scores.mean(axis=1), val_scores.mean(axis=1)
gap = train_mean - val_mean
print(f"  Final train F1-Macro: {train_mean[-1]:.3f}  |  Final CV F1-Macro: {val_mean[-1]:.3f}  "
      f"|  Gap: {gap[-1]:.3f}")
overfit_verdict = (
    "small gap (<0.05): no meaningful overfitting" if gap[-1] < 0.05 else
    "moderate gap (0.05-0.15): mild overfitting, acceptable" if gap[-1] < 0.15 else
    "large gap (>0.15): overfitting — consider regularizing further"
)
print(f"  Verdict: {overfit_verdict}")

plt.figure(figsize=(8, 5.5))
plt.plot(train_sizes, train_mean, 'o-', color='#3498DB', label='Training score')
plt.plot(train_sizes, val_mean, 'o-', color='#E74C3C', label='Cross-validation score')
plt.fill_between(train_sizes, train_mean, val_mean, alpha=0.1, color='grey')
plt.xlabel('Training examples')
plt.ylabel('F1-Macro')
plt.title('Learning Curve — Random Forest (winner)')
plt.legend(loc='best')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'eval_01_learning_curve.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {PLOTS_DIR}/eval_01_learning_curve.png")

# =============================================================================
# 2. CALIBRATION CURVE (are predicted probabilities trustworthy?)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: CALIBRATION CURVE (Random Forest, one-vs-rest per class)")
print("=" * 70)

rf_proba = best_rf.predict_proba(X_test)
rf_class_order = list(best_rf.named_steps['rf'].classes_)

fig, ax = plt.subplots(figsize=(7, 7))
ax.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
qty_colors = {1: '#3498DB', 2: '#27AE60', 3: '#F39C12', 4: '#E74C3C', 5: '#8E44AD'}
for cls in CLASSES:
    y_true_bin = (y_test.values == cls).astype(int)
    col_idx = rf_class_order.index(cls)
    frac_pos, mean_pred = calibration_curve(y_true_bin, rf_proba[:, col_idx], n_bins=8, strategy='quantile')
    ax.plot(mean_pred, frac_pos, 'o-', color=qty_colors[cls], label=f'Qty={cls}')
ax.set_xlabel('Mean predicted probability')
ax.set_ylabel('Fraction of positives (actual)')
ax.set_title('Calibration — Random Forest (one-vs-rest per class)')
ax.legend(loc='best', fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'eval_02_calibration_curve.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {PLOTS_DIR}/eval_02_calibration_curve.png")

rf_logloss = log_loss(y_test, rf_proba, labels=rf_class_order)
print(f"  Log-loss (lower is better-calibrated): {rf_logloss:.4f}")

# =============================================================================
# 3. ERROR SEGMENTATION (where does the model fail, beyond per-class recall?)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: ERROR SEGMENTATION BY PRODUCT/DEPARTMENT GROUP")
print("=" * 70)

rf_pred = best_rf.predict(X_test)
err_df = X_test.copy()
err_df['y_true'] = y_test.values
err_df['y_pred'] = rf_pred
err_df['correct'] = err_df['y_true'] == err_df['y_pred']

by_dept = err_df.groupby('dept_group')['correct'].agg(['mean', 'count']).rename(
    columns={'mean': 'accuracy', 'count': 'n_test_rows'}
).sort_values('accuracy')
print("\n  Accuracy by department-group (lowest first):")
print(by_dept.round(3).to_string())

plt.figure(figsize=(8, 5))
by_dept_sorted = by_dept.sort_values('accuracy')
plt.barh(by_dept_sorted.index.astype(str), by_dept_sorted['accuracy'], color='#3498DB')
plt.xlabel('Accuracy on test set')
plt.ylabel('dept_group (internal id)')
plt.title('Random Forest — Accuracy by Department Group')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'eval_03_accuracy_by_dept_group.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {PLOTS_DIR}/eval_03_accuracy_by_dept_group.png")

# ── Detailed per-class precision/recall/f1/support table ──────────────────────
precision, recall, f1, support = precision_recall_fscore_support(y_test, rf_pred, labels=CLASSES)
per_class_table = {
    f"qty_{c}": {'precision': float(p), 'recall': float(r), 'f1': float(f), 'support': int(s)}
    for c, p, r, f, s in zip(CLASSES, precision, recall, f1, support)
}

# =============================================================================
# SAVE EVALUATION REPORT
# =============================================================================
evaluation_report = {
    'generated_by': '12_model_evaluation.py',
    'reproduced_split_matches_training': {
        'train_rows': int(X_train.shape[0]),
        'test_rows': int(X_test.shape[0]),
    },
    'learning_curve': {
        'train_sizes': train_sizes.tolist(),
        'train_f1_macro': train_mean.tolist(),
        'cv_f1_macro': val_mean.tolist(),
        'final_train_cv_gap': float(gap[-1]),
        'verdict': overfit_verdict,
    },
    'calibration': {
        'log_loss': float(rf_logloss),
    },
    'error_by_dept_group': by_dept.reset_index().to_dict(orient='records'),
    'per_class_detail': per_class_table,
}
with open(os.path.join(REPORTS_DIR, 'evaluation_report.json'), 'w') as f:
    json.dump(evaluation_report, f, indent=2)

print("\n" + "=" * 70)
print(f"  Saved: {REPORTS_DIR}/evaluation_report.json")
print("PHASE 12 COMPLETE")
print("=" * 70)
