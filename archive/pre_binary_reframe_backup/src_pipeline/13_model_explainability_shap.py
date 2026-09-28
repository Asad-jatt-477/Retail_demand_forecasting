# =============================================================================
#  RETAIL DEMAND FORECASTING — PHASE 13: EXPLAINABILITY (SHAP)
#  Input  : data/processed/Retail_demand_FS.csv, models/v2/*.pkl
#  Output : reports/shap_feature_importance.json + assets/plots/explainability/*.png
#
#  WHY THIS PHASE EXISTS
#  ─────────────────────────────────────────────────────────────────────────
#  10_model_training.py's "feature importance" chart is Random Forest's
#  built-in impurity-based importance — fast, but known to be biased toward
#  high-cardinality / continuous features (exactly what product_group and
#  dept_group are), and it says nothing about DIRECTION (does a higher value
#  push the prediction toward Qty=1 or away from it?) or about individual
#  predictions.
#
#  SHAP (SHapley Additive exPlanations) fixes both: it gives a
#  theoretically-grounded, per-prediction attribution for every feature, so
#  this phase can answer two things Phase 10/12 cannot:
#    1. Which features actually drive each quantity-class prediction, with
#       direction (not just an impurity-based ranking)?
#    2. For the model's weakest classes (Qty=3/4 — see Phase 12), what does
#       the model look at when it gets a prediction right vs. wrong?
#
#  A random subsample of the test set is used (SHAP on a full 35k-row test
#  set with 5 output classes is unnecessarily slow for tree models; a few
#  hundred rows is enough for a stable importance ranking).
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
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin

# ─── PROJECT-ROOT-RELATIVE PATHS ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'processed', 'Retail_demand_FS.csv')
MODELS_DIR = os.path.join(ROOT_DIR, 'models', 'v2')
REPORTS_DIR = os.path.join(ROOT_DIR, 'reports')
PLOTS_DIR = os.path.join(ROOT_DIR, 'assets', 'plots', 'explainability')
os.makedirs(PLOTS_DIR, exist_ok=True)

RANDOM_STATE = 42
TARGET = 'order_item_quantity'
CLASSES = [1, 2, 3, 4, 5]
SHAP_SAMPLE_SIZE = 500  # rows sampled from the test set for SHAP (speed vs. stability trade-off)
FUTURE_INFO_COLS = [
    'processing_delay_days', 'late_delivery_risk', 'order_status_COMPLETE',
    'order_status_PENDING_PAYMENT', 'benefit_per_order_yj', 'order_item_profit_ratio',
]


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
print("PHASE 13: EXPLAINABILITY (SHAP)")
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

# Must match 10_model_training.py's Step 3 exactly, or the SHAP sample would
# be drawn partly from rows the saved model was actually trained on.
combo_id = X.groupby(list(X.columns)).ngroup()
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
_, test_idx = next(gss.split(X, y, groups=combo_id))
X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

# ── Load the winning model (Random Forest, per metrics_v2.json) ───────────────
with open(os.path.join(MODELS_DIR, 'random_forest_pipeline.pkl'), 'rb') as f:
    rf_pipeline = pickle.load(f)

rf_model = rf_pipeline.named_steps['rf']
target_enc = rf_pipeline.named_steps['target_enc']

# Take a stratified sample of the test set (SHAP is run on the ENCODED
# features — i.e. after the same target-encoding step the trained model
# actually sees — using the mapping learned during training, not re-fit here)
sample_idx = X_test.groupby(y_test).apply(
    lambda g: g.sample(min(len(g), SHAP_SAMPLE_SIZE // len(CLASSES)), random_state=RANDOM_STATE)
).index.get_level_values(-1)
X_sample_raw = X_test.loc[sample_idx]
y_sample = y_test.loc[sample_idx]
X_sample_encoded = target_enc.transform(X_sample_raw)
print(f"  SHAP sample: {len(X_sample_encoded)} rows (stratified across Qty=1..5)")

# =============================================================================
# COMPUTE SHAP VALUES (TreeExplainer — exact, fast for tree ensembles)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 1: COMPUTE SHAP VALUES")
print("=" * 70)

explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_sample_encoded)
# shap_values shape for multiclass RF: (n_samples, n_features, n_classes) in recent shap versions,
# or a list of (n_samples, n_features) arrays in older ones — normalize to the list form.
if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    shap_values = [shap_values[:, :, i] for i in range(shap_values.shape[2])]

print(f"  Computed SHAP values for {len(shap_values)} classes x {X_sample_encoded.shape[0]} rows")

# =============================================================================
# STEP 2: GLOBAL FEATURE IMPORTANCE (mean |SHAP value|, aggregated across classes)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: GLOBAL FEATURE IMPORTANCE (SHAP, all classes combined)")
print("=" * 70)

feature_names = list(X_sample_encoded.columns)
mean_abs_shap_per_class = np.stack([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)  # (n_classes, n_features)
overall_importance = mean_abs_shap_per_class.mean(axis=0)  # average across the 5 one-vs-rest outputs
importance_order = np.argsort(overall_importance)[::-1]

print("\n  Feature ranking (mean |SHAP value| across all 5 classes):")
for rank, idx in enumerate(importance_order, start=1):
    print(f"    {rank:2d}. {feature_names[idx]:<30} {overall_importance[idx]:.4f}")

plt.figure(figsize=(9, 7))
sorted_names = [feature_names[i] for i in importance_order]
sorted_vals = [overall_importance[i] for i in importance_order]
colors = ['#F39C12' if 'group' in n else '#16A085' for n in sorted_names]
plt.barh(sorted_names[::-1], sorted_vals[::-1], color=colors[::-1])
plt.xlabel('Mean |SHAP value| (avg. across Qty=1..5 one-vs-rest outputs)')
plt.title('Global Feature Importance — SHAP (Random Forest, winner)')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'shap_01_global_importance.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"\n  Saved: {PLOTS_DIR}/shap_01_global_importance.png")

# =============================================================================
# STEP 3: BEESWARM PLOTS — Qty=1 (majority) and Qty=5 (hardest bulk class)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: BEESWARM PLOTS (direction of effect, per class)")
print("=" * 70)

class_order = list(rf_model.classes_)  # matches the order of shap_values list
for cls_of_interest in [1, 5]:
    ci = class_order.index(cls_of_interest)
    plt.figure()
    shap.summary_plot(shap_values[ci], X_sample_encoded, show=False, plot_size=(9, 7))
    plt.title(f'SHAP Beeswarm — Qty={cls_of_interest} (one-vs-rest)')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'shap_02_beeswarm_qty{cls_of_interest}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {PLOTS_DIR}/shap_02_beeswarm_qty{cls_of_interest}.png")

# =============================================================================
# STEP 4: ONE WORKED EXAMPLE (a single Qty=3 prediction, the model's weakest class)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4: WORKED EXAMPLE — ONE Qty=3 PREDICTION EXPLAINED")
print("=" * 70)

qty3_rows = np.where(y_sample.values == 3)[0]
if len(qty3_rows) > 0:
    row_i = qty3_rows[0]
    ci3 = class_order.index(3)
    row_shap = shap_values[ci3][row_i]
    top5 = np.argsort(np.abs(row_shap))[::-1][:5]
    print(f"  Example row (true Qty=3) — top 5 features pushing the Qty=3 probability:")
    worked_example = []
    for idx in top5:
        entry = {
            'feature': feature_names[idx],
            'value': float(X_sample_encoded.iloc[row_i, idx]),
            'shap_contribution': float(row_shap[idx]),
        }
        worked_example.append(entry)
        direction = "increases" if entry['shap_contribution'] > 0 else "decreases"
        print(f"    {entry['feature']:<25} = {entry['value']:.3f}  -> {direction} Qty=3 probability "
              f"(SHAP={entry['shap_contribution']:+.4f})")
else:
    worked_example = []
    print("  No Qty=3 rows in this sample — skipped.")

# =============================================================================
# SAVE REPORT
# =============================================================================
shap_report = {
    'generated_by': '13_model_explainability_shap.py',
    'model_explained': 'random_forest (winner per metrics_v2.json)',
    'sample_size': int(X_sample_encoded.shape[0]),
    'global_feature_importance': {
        feature_names[i]: float(overall_importance[i]) for i in importance_order
    },
    'worked_example_qty3_prediction': worked_example,
}
with open(os.path.join(REPORTS_DIR, 'shap_feature_importance.json'), 'w') as f:
    json.dump(shap_report, f, indent=2)

print("\n" + "=" * 70)
print(f"  Saved: {REPORTS_DIR}/shap_feature_importance.json")
print("PHASE 13 COMPLETE")
print("=" * 70)
