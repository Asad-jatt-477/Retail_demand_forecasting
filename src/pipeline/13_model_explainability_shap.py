# =============================================================================
#  RETAIL DEMAND FORECASTING — PHASE 13: EXPLAINABILITY (SHAP) (v3, binary)
#  Input  : data/processed/Retail_demand_FS.csv, models/v3/*.pkl
#  Output : reports/shap_feature_importance.json + assets/plots/explainability/*.png
#
#  WHY THIS PHASE EXISTS
#  ─────────────────────────────────────────────────────────────────────────
#  10_model_training.py's "feature importance" chart is the model's built-in
#  impurity-based importance — fast, but known to be biased toward high-
#  cardinality / continuous features (exactly what product_group and
#  dept_group are), and it says nothing about DIRECTION (does a higher value
#  push the prediction toward "multi-unit" or toward "single-unit"?) or about
#  individual predictions.
#
#  SHAP (SHapley Additive exPlanations) fixes both: it gives a
#  theoretically-grounded, per-prediction attribution for every feature, so
#  this phase can answer two things Phase 10/12 cannot:
#    1. Which features actually drive the "multi-unit" prediction, and in
#       which direction?
#    2. For individual orders, what does the model look at?
#
#  The explained model is the winner recorded in metrics_v3.json. SHAP values
#  are for the P(multi-unit) output: positive = pushes toward Multi-unit
#  (bulk zone), negative = pushes toward Single-unit (fast-pick shelf).
#  (RF explains probability units; XGBoost explains log-odds units — compare
#  features within one model, not magnitudes across the two.)
#
#  A random stratified subsample of the test set is used (a few hundred rows
#  is enough for a stable importance ranking and keeps SHAP fast).
#
#  Requires the `shap` package (pip install shap).
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

import shap
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.base import BaseEstimator, TransformerMixin

# ─── PROJECT-ROOT-RELATIVE PATHS ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'processed', 'Retail_demand_FS.csv')
MODELS_DIR = os.path.join(ROOT_DIR, 'models', 'v3')
REPORTS_DIR = os.path.join(ROOT_DIR, 'reports')
PLOTS_DIR = os.path.join(ROOT_DIR, 'assets', 'plots', 'explainability')
os.makedirs(PLOTS_DIR, exist_ok=True)

RANDOM_STATE = 42
TARGET = 'order_item_quantity'
SHAP_SAMPLE_SIZE = 500  # rows sampled from the test set for SHAP (speed vs. stability trade-off)
# Must match 10_model_training.py exactly.
FUTURE_INFO_COLS = [
    'processing_delay_days', 'late_delivery_risk', 'order_status_COMPLETE',
    'order_status_PENDING_PAYMENT', 'benefit_per_order_yj', 'order_item_profit_ratio',
    'shipping_year', 'shipping_month',
]
QUANTITY_DERIVED_COLS = ['discount_intensity']


class KFoldTargetEncoder(BaseEstimator, TransformerMixin):
    """Must match 10_model_training.py exactly to unpickle the saved pipeline."""

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
        return self.fit(X, y).transform(X)


print("=" * 70)
print("PHASE 13: EXPLAINABILITY (SHAP) — binary: Single vs Multi unit")
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

# Must match 10_model_training.py's Step 3 exactly, or the SHAP sample would
# be drawn partly from rows the saved model was actually trained on.
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)

# ── Load the winning model, per metrics_v3.json ───────────────────────────────
with open(os.path.join(REPORTS_DIR, 'metrics_v3.json')) as f:
    prior_metrics = json.load(f)
assert prior_metrics['test_rows'] == len(X_test), \
    "Reproduced split does not match the Phase 10 split — do not trust this SHAP sample."
WINNER = prior_metrics['winner']

if WINNER == 'Random Forest':
    with open(os.path.join(MODELS_DIR, 'random_forest_pipeline.pkl'), 'rb') as f:
        pipeline = pickle.load(f)
    model_step = 'rf'
else:
    with open(os.path.join(MODELS_DIR, 'xgboost_pipeline.pkl'), 'rb') as f:
        pipeline = pickle.load(f)['pipeline']
    model_step = 'xgb'
print(f"  Explaining: {WINNER} (winner per metrics_v3.json)")

model = pipeline.named_steps[model_step]
target_enc = pipeline.named_steps['target_enc']

# Stratified sample of the test set (SHAP is run on the ENCODED features — i.e.
# after the same target-encoding step the trained model actually sees — using
# the mapping learned during training, not re-fit here)
per_class = SHAP_SAMPLE_SIZE // 2
sample_idx = X_test.groupby(y_test).apply(
    lambda g: g.sample(min(len(g), per_class), random_state=RANDOM_STATE)
).index.get_level_values(-1)
X_sample_raw = X_test.loc[sample_idx]
y_sample = y_test.loc[sample_idx]
X_sample_encoded = target_enc.transform(X_sample_raw)
print(f"  SHAP sample: {len(X_sample_encoded)} rows (stratified: single-unit / multi-unit)")

# =============================================================================
# COMPUTE SHAP VALUES (TreeExplainer — exact, fast for tree ensembles)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 1: COMPUTE SHAP VALUES")
print("=" * 70)

explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(X_sample_encoded)
# Normalize to a single (n_samples, n_features) array for the multi-unit output:
#   RF binary  -> list of 2 arrays, or ndarray (n, features, 2)  -> take class 1
#   XGB binary -> ndarray (n, features)                          -> already the positive class
if isinstance(sv, list):
    shap_multi = np.asarray(sv[1])
elif isinstance(sv, np.ndarray) and sv.ndim == 3:
    shap_multi = sv[:, :, 1]
else:
    shap_multi = np.asarray(sv)
assert shap_multi.shape == X_sample_encoded.shape, \
    f"Unexpected SHAP shape {shap_multi.shape} vs data {X_sample_encoded.shape}"
print(f"  Computed SHAP values for the multi-unit output: {shap_multi.shape}")

# =============================================================================
# STEP 2: GLOBAL FEATURE IMPORTANCE (mean |SHAP value|)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: GLOBAL FEATURE IMPORTANCE (SHAP)")
print("=" * 70)

feature_names = list(X_sample_encoded.columns)
overall_importance = np.abs(shap_multi).mean(axis=0)
importance_order = np.argsort(overall_importance)[::-1]

print("\n  Feature ranking (mean |SHAP value| for the multi-unit output):")
for rank, idx in enumerate(importance_order, start=1):
    print(f"    {rank:2d}. {feature_names[idx]:<30} {overall_importance[idx]:.4f}")

plt.figure(figsize=(9, 7))
sorted_names = [feature_names[i] for i in importance_order]
sorted_vals = [overall_importance[i] for i in importance_order]
colors = ['#F39C12' if 'group' in n else '#16A085' for n in sorted_names]
plt.barh(sorted_names[::-1], sorted_vals[::-1], color=colors[::-1])
plt.xlabel('Mean |SHAP value| (effect on the multi-unit prediction)')
plt.title(f'Global Feature Importance — SHAP ({WINNER}, winner)')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'shap_01_global_importance.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"\n  Saved: {PLOTS_DIR}/shap_01_global_importance.png")

# =============================================================================
# STEP 3: BEESWARM PLOT (direction of effect)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: BEESWARM PLOT (direction of effect on P(multi-unit))")
print("=" * 70)

plt.figure()
shap.summary_plot(shap_multi, X_sample_encoded, show=False, plot_size=(9, 7))
plt.title('SHAP Beeswarm — multi-unit output (right = toward Multi-unit / bulk zone)')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'shap_02_beeswarm_multi_unit.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {PLOTS_DIR}/shap_02_beeswarm_multi_unit.png")

# =============================================================================
# STEP 4: WORKED EXAMPLES (one true Single-unit and one true Multi-unit order)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4: WORKED EXAMPLES — ONE SINGLE-UNIT AND ONE MULTI-UNIT ORDER")
print("=" * 70)

worked_examples = {}
for label, cls in [('single_unit_order', 0), ('multi_unit_order', 1)]:
    rows = np.where(y_sample.values == cls)[0]
    if len(rows) == 0:
        worked_examples[label] = []
        print(f"  No true {label} rows in this sample — skipped.")
        continue
    row_i = rows[0]
    row_shap = shap_multi[row_i]
    top5 = np.argsort(np.abs(row_shap))[::-1][:5]
    print(f"\n  Example row (true {label}) — top 5 features moving the multi-unit prediction:")
    entries = []
    for idx in top5:
        entry = {
            'feature': feature_names[idx],
            'value': float(X_sample_encoded.iloc[row_i, idx]),
            'shap_contribution': float(row_shap[idx]),
        }
        entries.append(entry)
        direction = "toward Multi-unit" if entry['shap_contribution'] > 0 else "toward Single-unit"
        print(f"    {entry['feature']:<25} = {entry['value']:.3f}  -> pushes {direction} "
              f"(SHAP={entry['shap_contribution']:+.4f})")
    worked_examples[label] = entries

# =============================================================================
# SAVE REPORT
# =============================================================================
shap_report = {
    'generated_by': '13_model_explainability_shap.py',
    'model_explained': f'{WINNER.lower()} (winner per metrics_v3.json)',
    'output_explained': 'multi-unit prediction (positive SHAP = toward Qty 2..5, negative = toward Qty=1)',
    'sample_size': int(X_sample_encoded.shape[0]),
    'global_feature_importance': {
        feature_names[i]: float(overall_importance[i]) for i in importance_order
    },
    'worked_examples': worked_examples,
}
with open(os.path.join(REPORTS_DIR, 'shap_feature_importance.json'), 'w') as f:
    json.dump(shap_report, f, indent=2)

print("\n" + "=" * 70)
print(f"  Saved: {REPORTS_DIR}/shap_feature_importance.json")
print("PHASE 13 COMPLETE")
print("=" * 70)
