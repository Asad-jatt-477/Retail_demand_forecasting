# =============================================================================
#  RETAIL DEMAND FORECASTING — PHASE 12: MODEL EVALUATION (v3, binary)
#  Input  : data/processed/Retail_demand_FS.csv, models/v3/*.pkl
#  Output : reports/evaluation_report.json + assets/plots/evaluation/*.png
#
#  WHY A SEPARATE PHASE
#  ─────────────────────────────────────────────────────────────────────────
#  10_model_training.py reports test-set accuracy/F1/confusion matrices —
#  enough to pick a winner, but not enough to trust it in production. This
#  phase asks three questions Phase 10 doesn't answer:
#
#  1. Is the model OVERFITTING? (learning curve: train vs cross-validation
#     score as training-set size grows)
#  2. Are its predicted PROBABILITIES trustworthy? (calibration curve — when
#     the model says "70% multi-unit", is it right about 70% of the time?)
#  3. WHERE does it fail? (error breakdown by department group and by price
#     zone, not just by class)
#
#  This script re-derives the exact train/test split used in
#  10_model_training.py (same random_state, same dropped columns, same
#  group-id recovery, same binary target, same stratified train_test_split) so
#  "test set" here means the same held-out rows the saved model has genuinely
#  never seen.
#
#  v3 changes: binary target (Single vs Multi unit); stratified row split (see
#  10_model_training.py, FIX G — the model now has only a few hundred distinct
#  inputs, so it cannot memorize rows, and a group-by-feature-combination split
#  would hold out whole products instead).
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless-safe; no blocking plt.show() windows
import matplotlib.pyplot as plt
import json
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, learning_curve, train_test_split
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss, brier_score_loss, precision_recall_fscore_support
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# ─── PROJECT-ROOT-RELATIVE PATHS ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'processed', 'Retail_demand_FS.csv')
MODELS_DIR = os.path.join(ROOT_DIR, 'models', 'v3')
REPORTS_DIR = os.path.join(ROOT_DIR, 'reports')
PLOTS_DIR = os.path.join(ROOT_DIR, 'assets', 'plots', 'evaluation')
os.makedirs(PLOTS_DIR, exist_ok=True)

RANDOM_STATE = 42
TARGET = 'order_item_quantity'
CLASSES = [0, 1]
CLASS_KEYS = ['single_unit', 'multi_unit']
# Must match 10_model_training.py exactly.
FUTURE_INFO_COLS = [
    'processing_delay_days', 'late_delivery_risk', 'order_status_COMPLETE',
    'order_status_PENDING_PAYMENT', 'benefit_per_order_yj', 'order_item_profit_ratio',
    'shipping_year', 'shipping_month',
]
QUANTITY_DERIVED_COLS = ['discount_intensity']


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
print("PHASE 12: MODEL EVALUATION (binary: Single vs Multi unit)")
print("=" * 70)

# ── Reproduce the exact Phase 10 train/test split ─────────────────────────────
df = pd.read_csv(DATA_PATH)
present = [c for c in FUTURE_INFO_COLS + QUANTITY_DERIVED_COLS if c in df.columns]
df.drop(columns=present, inplace=True)

GROUP_SOURCES = {'product_group': 'product_avg_qty', 'dept_group': 'dept_avg_qty'}
GROUP_COLS = []
for _g, _src in GROUP_SOURCES.items():
    if _src in df.columns:
        df[_g] = pd.factorize(df[_src])[0]
        df.drop(columns=[_src], inplace=True)
        GROUP_COLS.append(_g)

X = df.drop(columns=[TARGET])
y = (df[TARGET] >= 2).astype(int)

# Must match 10_model_training.py's Step 3 exactly, or this "test set" would
# include rows the saved model was actually trained on.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
print(f"  Reproduced split: {X_train.shape[0]:,} train / {X_test.shape[0]:,} test rows "
      f"(must match 10_model_training.py's numbers)")

# ── Load trained pipelines ─────────────────────────────────────────────────────
with open(os.path.join(MODELS_DIR, 'random_forest_pipeline.pkl'), 'rb') as f:
    best_rf = pickle.load(f)
with open(os.path.join(MODELS_DIR, 'xgboost_pipeline.pkl'), 'rb') as f:
    best_xgb = pickle.load(f)['pipeline']

with open(os.path.join(REPORTS_DIR, 'metrics_v3.json')) as f:
    prior_metrics = json.load(f)
assert prior_metrics['train_rows'] == X_train.shape[0] and prior_metrics['test_rows'] == X_test.shape[0], \
    "Reproduced split does not match the Phase 10 split — do not trust this evaluation."

WINNER = prior_metrics['winner']
winner_pipe = best_rf if WINNER == 'Random Forest' else best_xgb
print(f"  Winner per metrics_v3.json: {WINNER}")

# =============================================================================
# 1. LEARNING CURVE (Random Forest)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 1: LEARNING CURVE (is the model overfitting?)")
print("=" * 70)
print("  (Random Forest is used for the learning curve because class_weight is native;")
print("   XGBoost would need per-subset sample weights.)")

rf_best_params = {k.replace('rf__', ''): v for k, v in prior_metrics['random_forest']['best_params'].items()}
lc_pipeline = Pipeline([
    ('target_enc', KFoldTargetEncoder(cols=GROUP_COLS, n_splits=5,
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
plt.title('Learning Curve — Random Forest (binary)')
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
print(f"STEP 2: CALIBRATION CURVE ({WINNER}, P(multi-unit))")
print("=" * 70)

proba = winner_pipe.predict_proba(X_test)[:, 1]
frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy='quantile')

fig, ax = plt.subplots(figsize=(7, 7))
ax.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
ax.plot(mean_pred, frac_pos, 'o-', color='#3498DB', label='Multi-unit probability')
ax.set_xlabel('Mean predicted probability of multi-unit')
ax.set_ylabel('Fraction actually multi-unit')
ax.set_title(f'Calibration — {WINNER}')
ax.legend(loc='best', fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'eval_02_calibration_curve.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {PLOTS_DIR}/eval_02_calibration_curve.png")

ll = log_loss(y_test, proba)
brier = brier_score_loss(y_test, proba)
print(f"  Log-loss: {ll:.4f}   Brier score: {brier:.4f}  (lower is better-calibrated)")

# =============================================================================
# 3. ERROR SEGMENTATION (where does the model fail?)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: ERROR SEGMENTATION (price zone and price band)")
print("=" * 70)

pred = winner_pipe.predict(X_test)
err_df = X_test.copy()
err_df['y_true'] = y_test.values
err_df['y_pred'] = pred
err_df['correct'] = err_df['y_true'] == err_df['y_pred']

# Price band (the model's real driver): how accurate is it, and how often are orders
# actually multi-unit, in each band?
_price = np.expm1(err_df['product_price_log'])
_bins = [0, 25, 50, 100.0001, 200, 500, np.inf]
_labels = ['<= $25', '$25-50', '$50-100', '$100-200', '$200-500', '> $500']
err_df['price_band'] = pd.cut(_price, bins=_bins, labels=_labels)
by_band = err_df.groupby('price_band', observed=True).agg(
    n_test_rows=('correct', 'size'),
    accuracy=('correct', 'mean'),
    share_multi_unit=('y_true', 'mean'),
    predicted_multi_share=('y_pred', 'mean'),
)
print("\n  By price band:")
print(by_band.round(3).to_string())

# Price zone, using the SAME train-only threshold as the Phase 10 price rule:
# above the highest price at which a multi-unit order was seen in training,
# multi-unit orders essentially do not occur (business "single-unit only" zone).
max_multi_price_log = float(X_train.loc[y_train == 1, 'product_price_log'].max())
err_df['price_zone'] = np.where(
    err_df['product_price_log'] <= max_multi_price_log,
    f'multi-unit possible (price <= ${np.expm1(max_multi_price_log):,.0f})',
    f'above ${np.expm1(max_multi_price_log):,.0f}',
)
by_zone = err_df.groupby('price_zone').agg(
    n_test_rows=('correct', 'size'),
    accuracy=('correct', 'mean'),
    share_multi_unit=('y_true', 'mean'),
    predicted_multi_share=('y_pred', 'mean'),
)
print("\n  By price zone (share_multi_unit = how often orders there are actually multi-unit):")
print(by_zone.round(3).to_string())

plt.figure(figsize=(8, 5))
plt.bar(by_band.index.astype(str), by_band['accuracy'], color='#3498DB', label='Accuracy')
plt.plot(by_band.index.astype(str), by_band['share_multi_unit'], 'o--', color='#E74C3C',
         label='Share of orders that are multi-unit')
plt.ylim(0, 1.05)
plt.xlabel('Product price band')
plt.ylabel('Test set')
plt.title(f'{WINNER} — Accuracy by Price Band')
plt.legend(loc='center left', fontsize=9)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'eval_03_accuracy_by_price_band.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {PLOTS_DIR}/eval_03_accuracy_by_price_band.png")

precision, recall, f1, support = precision_recall_fscore_support(y_test, pred, labels=CLASSES)
per_class_table = {
    key: {'precision': float(p), 'recall': float(r), 'f1': float(f), 'support': int(s)}
    for key, p, r, f, s in zip(CLASS_KEYS, precision, recall, f1, support)
}

# =============================================================================
# SAVE EVALUATION REPORT
# =============================================================================
evaluation_report = {
    'generated_by': '12_model_evaluation.py',
    'model_evaluated': WINNER,
    'reproduced_split_matches_training': {
        'train_rows': int(X_train.shape[0]),
        'test_rows': int(X_test.shape[0]),
    },
    'learning_curve': {
        'model': 'Random Forest',
        'cv': 'StratifiedKFold(3)',
        'train_sizes': train_sizes.tolist(),
        'train_f1_macro': train_mean.tolist(),
        'cv_f1_macro': val_mean.tolist(),
        'final_train_cv_gap': float(gap[-1]),
        'verdict': overfit_verdict,
    },
    'calibration': {
        'log_loss': float(ll),
        'brier_score': float(brier),
    },
    'error_by_price_band': by_band.reset_index().astype({'price_band': str}).to_dict(orient='records'),
    'error_by_price_zone': by_zone.reset_index().to_dict(orient='records'),
    'per_class_detail': per_class_table,
}
with open(os.path.join(REPORTS_DIR, 'evaluation_report.json'), 'w') as f:
    json.dump(evaluation_report, f, indent=2)

print("\n" + "=" * 70)
print(f"  Saved: {REPORTS_DIR}/evaluation_report.json")
print("PHASE 12 COMPLETE")
print("=" * 70)
