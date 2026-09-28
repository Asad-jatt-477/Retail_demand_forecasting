# =============================================================================
#  RETAIL DEMAND FORECASTING — FEATURE SELECTION
#  Phase 9: Complete Feature Selection Pipeline
#  Input  : Retail_demand_FE.csv          (178,396 rows × 47 columns)
#  Output : Retail_demand_FS.csv          (178,396 rows × selected columns)
#  Target : order_item_quantity  (1, 2, 3, 4, 5)
# =============================================================================
#
#  WHAT IS FEATURE SELECTION?
#  ─────────────────────────────────────────────────────────────────────────────
#  Feature Engineering mein humne raw columns se smart features banaye.
#  Feature Selection mein hum decide karte hain ke un features mein se
#  KAUN SE model ke liye actually useful hain.
#
#  Too many features = noise, overfitting, slow training
#  Too few features  = underfitting, loss of signal
#  Goal              = optimal subset jo model ko best performance de
#
#  APPROACH: 3-METHOD CONSENSUS VOTING
#  ─────────────────────────────────────────────────────────────────────────────
#  Koi ek method akela reliable nahi hota. Isliye 3 independent methods use:
#
#  Method 1 → FILTER    : Mutual Information (MI Score)
#             Measures non-linear statistical dependency between feature & target
#             Kisi bhi model se independent — pure statistical test
#
#  Method 2 → EMBEDDED  : Random Forest Feature Importance
#             Tree-based model khud feature importance calculate karta hai
#             during training — captures interaction effects
#
#  Method 3 → WRAPPER   : RFECV (Recursive Feature Elimination with CV)
#             Systematically features remove karta hai, cross-validation se
#             check karta hai — most rigorous but computationally expensive
#
#  CONSENSUS RULE:
#    ≥ 2/3 methods mein selected → KEEP  ✓
#    < 2/3 methods mein selected → DROP  ✗
#    Negative permutation importance → UNCONDITIONAL DROP (model ko hurt karta hai)
#
#  FEATURE ENGINEERING PHASE SE CARRIED-OVER TASKS:
#  ─────────────────────────────────────────────────────────────────────────────
#  1. OHE (One-Hot Encoding) for remaining categoricals:
#     customer_segment (3), market (5), order_region (23), order_status (9)
#  2. is_fulfilled DROP — temporal leakage (order status known after delivery)
#  3. Leaky features (product_avg_qty, category_avg_qty, dept_avg_qty,
#     is_bulk_category, is_single_unit_category) — evaluate karein ge lekin
#     Modeling phase mein properly fix honge via sklearn Pipeline + TargetEncoder
#
#  FEATURE SELECTION PLAN:
#  ─────────────────────────────────────────────────────────────────────────────
#  STEP 0 : Load FE dataset + Pre-selection fixes
#           (OHE categoricals, drop temporal leakage, handle duplicates)
#  STEP 1 : Variance Threshold
#           (near-zero variance features drop karo)
#  STEP 2 : Multicollinearity Check
#           (r > 0.85 pairs mein se weaker feature drop)
#  STEP 3 : Method 1 — Mutual Information Score
#  STEP 4 : Method 2 — Random Forest Feature Importance + Permutation Importance
#  STEP 5 : Method 3 — RFECV (Recursive Feature Elimination with CV)
#  STEP 6 : Consensus Voting (2-of-3 rule)
#  STEP 7 : Final Feature Set — Save Retail_demand_FS.csv
#  STEP 8 : Summary Report + Plots
#
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless-safe; no blocking plt.show() windows
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import warnings
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(ROOT_DIR, 'data', 'processed')

from sklearn.feature_selection import (
    mutual_info_classif,
    VarianceThreshold,
    RFECV,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# ── Color Palette (consistent with all previous phases) ──────────────────────
palette = {
    'primary':   '#2C3E50',
    'accent':    '#E74C3C',
    'highlight': '#3498DB',
    'success':   '#27AE60',
    'warn':      '#F39C12',
    'bg':        '#FAFAFA',
    'grid':      '#ECEFF1',
    'purple':    '#8E44AD',
    'teal':      '#16A085',
}

plt.rcParams.update({
    'figure.facecolor': palette['bg'],
    'axes.facecolor':   palette['bg'],
    'axes.edgecolor':   palette['primary'],
    'axes.labelcolor':  palette['primary'],
    'xtick.color':      palette['primary'],
    'ytick.color':      palette['primary'],
    'grid.color':       palette['grid'],
    'font.family':      'DejaVu Sans',
    'axes.titleweight': 'bold',
    'axes.titlesize':   11,
})

qty_colors = {1: '#3498DB', 2: '#27AE60', 3: '#F39C12', 4: '#E74C3C', 5: '#8E44AD'}

PLOTS_DIR = os.path.join(ROOT_DIR, 'assets', 'plots', 'feature_selection')
os.makedirs(PLOTS_DIR, exist_ok=True)
print(f"📁 Output folder ready: '{PLOTS_DIR}/'")


# =============================================================================
# STEP 0: LOAD DATASET + PRE-SELECTION FIXES
# =============================================================================
# Pre-selection fixes woh kaam hain jo Feature Engineering phase mein
# deliberately chodey gaye they — OHE, temporal leakage drop, etc.
# =============================================================================

print("\n" + "="*65)
print("STEP 0: LOAD DATASET + PRE-SELECTION FIXES")
print("="*65)

df = pd.read_csv(os.path.join(DATA_DIR, 'Retail_demand_FE.csv'), encoding='latin-1')
TARGET = 'order_item_quantity'

print(f"\n  Input shape  : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"  Missing vals : {df.isnull().sum().sum()}")
print(f"  Target       : {TARGET}  |  Classes: {sorted(df[TARGET].unique())}")
print(f"\n  Target distribution:")
vc = df[TARGET].value_counts().sort_index()
for qty, cnt in vc.items():
    print(f"    Qty {qty}  →  {cnt:,}  ({cnt/len(df)*100:.1f}%)")

# ── FIX 1: Drop is_fulfilled — Temporal Leakage ──────────────────────────────
# WHY DROP: is_fulfilled = 1 jab order COMPLETE ya CLOSED ho.
# Lekin yeh status order deliver hone KE BAAD pata chalta hai.
# Naye orders pe prediction karte waqt order sirf PENDING hoga —
# COMPLETE status future mein hai. Yeh training-serving skew create karta hai.
# ─────────────────────────────────────────────────────────────────────────────
if 'is_fulfilled' in df.columns:
    df.drop(columns=['is_fulfilled'], inplace=True)
    print(f"\n  [FIX 1] is_fulfilled DROPPED — temporal leakage (future information)")

# ── FIX 2: Drop order_item_product_price_log — Duplicate of product_price_log ─
# WHY DROP: product_price aur order_item_product_price perfect duplicates they
# (r=1.00 confirmed in EDA). Unke log versions bhi identical hain.
# product_price_log already rakha hua hai — duplicate drop karo.
# ─────────────────────────────────────────────────────────────────────────────
if 'order_item_product_price_log' in df.columns:
    dup_corr = df['order_item_product_price_log'].corr(df['product_price_log'])
    print(f"\n  [FIX 2] order_item_product_price_log DROPPED")
    print(f"          Correlation with product_price_log: {dup_corr:.4f} (exact duplicate)")
    df.drop(columns=['order_item_product_price_log'], inplace=True)

# ── FIX 3: Drop order_item_discount — Already replaced by engineered features ─
# WHY: Group G mein has_discount + discount_intensity banaye gaye.
# Raw order_item_discount column reh gaya tha dataset mein — ab drop karo.
# ─────────────────────────────────────────────────────────────────────────────
if 'order_item_discount' in df.columns:
    print(f"\n  [FIX 3] order_item_discount DROPPED — replaced by has_discount + discount_intensity")
    df.drop(columns=['order_item_discount'], inplace=True)

# ── FIX 4: Drop product_price — raw version, product_price_log better ─────────
# WHY: product_price_log is more compressed and handles skew better.
# is_high_price aur price_tier_encoded already $100 boundary capture kar chuke.
# ─────────────────────────────────────────────────────────────────────────────
if 'product_price' in df.columns:
    print(f"\n  [FIX 4] product_price DROPPED — product_price_log + is_high_price already present")
    df.drop(columns=['product_price'], inplace=True)

# ── FIX 5: OHE — Remaining Categorical Columns ────────────────────────────────
# WHY OHE: customer_segment (3), market (5) — low cardinality, safe for OHE.
# order_region (23) — medium cardinality but geographically meaningful.
# order_status (9) — fulfillment stage of order.
#
# NOTE on order_status: is_fulfilled already bana tha (ab drop kar diya).
# order_status raw mein 9 values hain — OHE se 9 dummies banenge.
# Feature Selection decide karegi ke kaunsi dummies useful hain.
#
# drop_first=False kyun:
# Tree-based models (XGBoost, LightGBM) ko dummy trap ki zaroorat nahi —
# woh non-linear hain. drop_first=True sirf linear models ke liye zaroori hai.
# ─────────────────────────────────────────────────────────────────────────────
cat_cols_for_ohe = ['customer_segment', 'market', 'order_region', 'order_status']
ohe_cols_created = []

print(f"\n  [FIX 5] One-Hot Encoding for categorical columns:")
for col in cat_cols_for_ohe:
    if col in df.columns:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False, dtype=int)
        df = pd.concat([df, dummies], axis=1)
        df.drop(columns=[col], inplace=True)
        ohe_cols_created.extend(dummies.columns.tolist())
        print(f"    OHE: {col:<20} → {dummies.shape[1]} dummy columns created")

print(f"\n  Total OHE columns created: {len(ohe_cols_created)}")

# ── FIX 6: Drop sales_per_customer_log — Target Leakage (added after Modeling review) ─
# WHY DROP: sales_per_customer / order_item_total / Sales are near-identical
# (r=0.99, confirmed in Feature Engineering Step "Financial Duplicates").
# In this dataset, Sales = Order Item Quantity x Product Price — so
# sales_per_customer_log is essentially a transformed version of the TARGET
# itself (quantity is baked into its formula). Modeling phase results showed
# this exact symptom: 99.9% accuracy, AUC=1.000 across all classes, and
# sales_per_customer_log as the #1 feature (42% importance in Random Forest) —
# the model was reverse-calculating quantity, not predicting it.
# ─────────────────────────────────────────────────────────────────────────────
for leak_col in ['sales_per_customer_log', 'sales_per_customer']:
    if leak_col in df.columns:
        leak_corr = df[leak_col].corr(df[TARGET])
        print(f"\n  [FIX 6] {leak_col} DROPPED — target leakage")
        print(f"          Correlation with {TARGET}: {leak_corr:.4f}")
        print(f"          (Sales = Quantity x Price in this dataset, so this feature")
        print(f"           is a near-direct transform of the target — must not be used.)")
        df.drop(columns=[leak_col], inplace=True)

print(f"  Dataset shape after pre-selection fixes: {df.shape[0]:,} × {df.shape[1]}")

# ── Define X and y ────────────────────────────────────────────────────────────
X = df.drop(columns=[TARGET])
y = df[TARGET]

print(f"\n  X shape : {X.shape}")
print(f"  y shape : {y.shape}")
print(f"  Features going into selection: {X.shape[1]}")

# ── Note on Leaky Features ────────────────────────────────────────────────────
# product_avg_qty, category_avg_qty, dept_avg_qty → full df par target mean se bane
# is_bulk_category, is_single_unit_category → category_avg_qty se derived
# Yeh features selection mein evaluate honge lekin Modeling phase mein
# sklearn Pipeline + TargetEncoder se properly replace honge.
# Abhi inko selection mein shamil karte hain — unka signal strength dekhna zaroori hai.
LEAKY_FEATURES = [
    'product_avg_qty', 'category_avg_qty', 'dept_avg_qty',
    'is_bulk_category', 'is_single_unit_category'
]
print(f"\n  ⚠️  Leaky features (will be fixed in Modeling via TargetEncoder Pipeline):")
for lf in LEAKY_FEATURES:
    if lf in X.columns:
        print(f"      {lf}")


# =============================================================================
# STEP 1: VARIANCE THRESHOLD
# =============================================================================
# WHY FIRST?
# Near-zero variance features model ko kuch nahi sikhate — agar ek feature
# ki 95%+ values same hain, toh woh effectively constant hai.
# Inhe sabse pehle hatao — baaki methods ka time waste na ho.
#
# Threshold = 0.01:
# Matlab ek feature rakha jayega sirf agar uski variance > 0.01 ho.
# Binary feature jo 95% ek value hai: variance = 0.95 × 0.05 = 0.0475 → rakha jayega
# Binary feature jo 99% ek value hai: variance = 0.99 × 0.01 = 0.0099 → DROP
# =============================================================================

print("\n" + "="*65)
print("STEP 1: VARIANCE THRESHOLD")
print("="*65)

VARIANCE_THRESHOLD = 0.01

selector_var = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
selector_var.fit(X)

low_var_mask   = ~selector_var.get_support()
low_var_cols   = X.columns[low_var_mask].tolist()
kept_var_cols  = X.columns[selector_var.get_support()].tolist()

print(f"\n  Variance Threshold : {VARIANCE_THRESHOLD}")
print(f"  Features before    : {X.shape[1]}")
print(f"  Features dropped   : {len(low_var_cols)}")
print(f"  Features kept      : {len(kept_var_cols)}")

if low_var_cols:
    print(f"\n  LOW VARIANCE — DROPPED:")
    for col in low_var_cols:
        var = X[col].var()
        vc_pct = X[col].value_counts(normalize=True).iloc[0] * 100
        print(f"    DROP: {col:<40} | var={var:.6f} | dominant value={vc_pct:.1f}%")
else:
    print(f"\n  No features dropped by Variance Threshold (all variance > {VARIANCE_THRESHOLD})")

print(f"\n  All features and their variance:")
variances = X.var().sort_values()
for col, var in variances.items():
    status = "DROP" if col in low_var_cols else "KEEP"
    marker = " ← LOW VARIANCE" if col in low_var_cols else ""
    print(f"    [{status}] {col:<45} var={var:.6f}{marker}")

# Apply variance threshold
X_var = X[kept_var_cols].copy()
print(f"\n  ✓ After Variance Threshold: {X_var.shape[1]} features remain")


# =============================================================================
# STEP 2: MULTICOLLINEARITY CHECK
# =============================================================================
# WHY?
# Jab do features highly correlated hain (r > 0.85), model ko confuse karta hai:
#   - Feature importance dilute ho jaati hai
#   - Model ek hi signal ko do baar count karta hai
#
# APPROACH:
# Correlation matrix banao → r > 0.85 pairs dhundho → har pair mein se
# woh feature drop karo jis ka TARGET se correlation kam ho.
# Agar dono ka target corr same ho → jo FE mein later bana woh drop karo.
#
# Threshold = 0.85 (strict nahi, kyunki tree models multicollinearity
# se kam suffer karte hain compared to linear models — lekin saaf rakhna better)
# =============================================================================

print("\n" + "="*65)
print("STEP 2: MULTICOLLINEARITY CHECK (r > 0.85)")
print("="*65)

MULTICOLL_THRESHOLD = 0.85

numeric_X = X_var.select_dtypes(include=['int64', 'float64', 'int32'])
corr_matrix = numeric_X.corr().abs()

# Upper triangle only (har pair ek baar)
upper_tri = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

high_corr_pairs = []
for col in upper_tri.columns:
    for row in upper_tri.index:
        val = upper_tri.loc[row, col]
        if val > MULTICOLL_THRESHOLD:
            high_corr_pairs.append((row, col, val))

high_corr_pairs.sort(key=lambda x: x[2], reverse=True)

print(f"\n  Threshold: r > {MULTICOLL_THRESHOLD}")
print(f"  High-correlation pairs found: {len(high_corr_pairs)}")

multicoll_drop = set()

if high_corr_pairs:
    print(f"\n  High-Correlation Pairs:")
    for feat1, feat2, r_val in high_corr_pairs:
        corr1 = abs(X_var[feat1].corr(y))
        corr2 = abs(X_var[feat2].corr(y))

        # Drop the one with lower target correlation
        # (if both already in drop set, skip)
        if feat1 in multicoll_drop or feat2 in multicoll_drop:
            drop_candidate = feat1 if feat1 in multicoll_drop else feat2
        elif corr1 >= corr2:
            drop_candidate = feat2
        else:
            drop_candidate = feat1

        keep_candidate = feat2 if drop_candidate == feat1 else feat1
        multicoll_drop.add(drop_candidate)

        print(f"\n    Pair: {feat1}  ↔  {feat2}")
        print(f"           r = {r_val:.4f}")
        print(f"           {feat1:<45} target_corr={corr1:.4f}")
        print(f"           {feat2:<45} target_corr={corr2:.4f}")
        print(f"           → DROP: {drop_candidate}  |  KEEP: {keep_candidate}")

multicoll_drop = list(multicoll_drop)
kept_multicoll = [c for c in X_var.columns if c not in multicoll_drop]

print(f"\n  Features dropped due to multicollinearity : {len(multicoll_drop)}")
for col in multicoll_drop:
    print(f"    DROP: {col}")
print(f"  Features remaining: {len(kept_multicoll)}")

X_mc = X_var[kept_multicoll].copy()
print(f"\n  ✓ After Multicollinearity Check: {X_mc.shape[1]} features remain")

# ── Plot: Correlation Heatmap ─────────────────────────────────────────────────
print(f"\n  Generating correlation heatmap...")
corr_plot = X_mc.select_dtypes(include=['int64','float64','int32']).corr()
n_feat_mc = len(corr_plot.columns)
heat_side = max(14, min(24, 0.42 * n_feat_mc + 6))
annot_size = max(4.5, min(8, 210 / max(n_feat_mc, 1)))

fig, ax = plt.subplots(figsize=(heat_side, heat_side * 0.82))
mask = np.triu(np.ones_like(corr_plot, dtype=bool))
sns.heatmap(
    corr_plot, mask=mask, ax=ax,
    cmap='RdBu_r', center=0, vmin=-1, vmax=1,
    annot=True, fmt='.2f', annot_kws={'size': annot_size},
    linewidths=0.3, linecolor='white',
    cbar_kws={'shrink': 0.7, 'label': 'Pearson r'}
)
ax.set_title(
    'Correlation Heatmap — Post Multicollinearity Removal\n'
    '(Features entering Method 1/2/3 Selection)',
    fontsize=14, fontweight='bold', pad=14
)
ax.tick_params(axis='x', rotation=45, labelsize=8)
ax.tick_params(axis='y', rotation=0,  labelsize=8)
for label in ax.get_xticklabels():
    label.set_ha('right')
fig.tight_layout(rect=[0, 0, 1, 0.96])
save_path = os.path.join(PLOTS_DIR, 'fs_01_correlation_heatmap.png')
fig.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {save_path}")


# =============================================================================
# STEP 3: METHOD 1 — MUTUAL INFORMATION SCORE
# =============================================================================
# WHY MUTUAL INFORMATION?
# Pearson correlation sirf LINEAR relationships pakadti hai.
# Mutual Information (MI) non-linear relationships bhi pakadta hai —
# jo tree-based models ke liye zyada relevant hai.
#
# MI = 0        → feature aur target bilkul independent hain
# MI > 0        → feature mein target ke baare mein kuch information hai
# (MI ki koi upper bound nahi — relative ranking zaroori hai)
#
# Selection Rule: Top 60% features by MI score → METHOD 1 SELECTED
# Baaki 40% → method 1 ne reject kiya
# =============================================================================

print("\n" + "="*65)
print("STEP 3: METHOD 1 — MUTUAL INFORMATION SCORE")
print("="*65)

X_mc_values = X_mc.values
y_values    = y.values

print(f"\n  Computing Mutual Information scores...")
print(f"  (random_state=42 for reproducibility)")

mi_scores = mutual_info_classif(
    X_mc_values, y_values,
    discrete_features='auto',
    random_state=42
)
mi_series = pd.Series(mi_scores, index=X_mc.columns).sort_values(ascending=False)

# Selection threshold: top 60% features
mi_threshold_count = int(len(mi_series) * 0.60)
mi_selected = set(mi_series.head(mi_threshold_count).index.tolist())

print(f"\n  Total features evaluated: {len(mi_series)}")
print(f"  Selection threshold: Top {mi_threshold_count} features (top 60%)")
print(f"  MI Selected: {len(mi_selected)}")

print(f"\n  MI Scores (sorted by importance):")
for i, (col, score) in enumerate(mi_series.items(), 1):
    status = "✓ SELECT" if col in mi_selected else "✗ REJECT"
    leaky  = " ⚠️ LEAKY" if col in LEAKY_FEATURES else ""
    print(f"    {i:2}. [{status}] {col:<45} MI={score:.6f}{leaky}")

# ── Plot: MI Scores ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, max(8, len(mi_series) * 0.28)))
colors = [palette['success'] if c in mi_selected else palette['accent'] for c in mi_series.index]
bars = ax.barh(range(len(mi_series)), mi_series.values, color=colors, edgecolor='white', alpha=0.85)
ax.set_yticks(range(len(mi_series)))
ax.set_yticklabels(mi_series.index, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel('Mutual Information Score', fontsize=11)
ax.set_title(
    'Method 1: Mutual Information Scores\n'
    f'(Green = Selected top {mi_threshold_count}, Red = Rejected)',
    fontsize=12, fontweight='bold'
)
ax.axvline(x=mi_series.iloc[mi_threshold_count-1], color=palette['warn'],
           linestyle='--', linewidth=1.5, label=f'Threshold (top {mi_threshold_count})')
ax.legend(fontsize=9)
ax.grid(axis='x', linestyle='--', alpha=0.4)
for bar, val in zip(bars, mi_series.values):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'fs_02_mutual_information.png'), dpi=300, bbox_inches='tight')
plt.close('all')
print(f"\n  Saved: {PLOTS_DIR}/fs_02_mutual_information.png")


# =============================================================================
# STEP 4: METHOD 2 — RANDOM FOREST IMPORTANCE + PERMUTATION IMPORTANCE
# =============================================================================
# WHY RANDOM FOREST IMPORTANCE?
# RF internally tracks how much each feature reduces impurity (Gini) across
# all trees. Yeh EMBEDDED method hai — model training ke andar hi hota hai.
# Non-linear interactions capture karta hai.
#
# WHY ALSO PERMUTATION IMPORTANCE?
# RF's built-in importance high-cardinality aur correlated features ko
# artificially inflate kar sakti hai.
# Permutation importance ek feature ko randomly shuffle karta hai aur
# dekhu hai ke model performance kitni girti hai.
# NEGATIVE permutation importance = feature model ko HURT karta hai = UNCONDITIONAL DROP
#
# n_estimators=200: enough trees for stable importance estimates
# class_weight='balanced': imbalanced target (54.9% qty=1) handle karo
# max_depth=8: prevent overfitting in importance estimation
# =============================================================================

print("\n" + "="*65)
print("STEP 4: METHOD 2 — RANDOM FOREST IMPORTANCE + PERMUTATION IMPORTANCE")
print("="*65)

print(f"\n  Training Random Forest for feature importance...")
print(f"  (n_estimators=200, class_weight='balanced', random_state=42)")
print(f"  Note: This may take 1-2 minutes on 178K rows...")

rf_selector = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=50,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_selector.fit(X_mc, y)

# Built-in RF importance
rf_importance = pd.Series(
    rf_selector.feature_importances_,
    index=X_mc.columns
).sort_values(ascending=False)

# Selection threshold: top 60% by RF importance
rf_threshold_count = int(len(rf_importance) * 0.60)
rf_selected_builtin = set(rf_importance.head(rf_threshold_count).index.tolist())

print(f"\n  RF Built-in Importance — Top {rf_threshold_count} selected:")
for i, (col, score) in enumerate(rf_importance.items(), 1):
    status = "✓" if col in rf_selected_builtin else "✗"
    leaky  = " ⚠️ LEAKY" if col in LEAKY_FEATURES else ""
    print(f"    {i:2}. [{status}] {col:<45} RF_imp={score:.6f}{leaky}")

# Permutation Importance (on training data — fast version)
print(f"\n  Computing Permutation Importance...")
print(f"  (n_repeats=10, this confirms which features HURT the model)")

perm_result = permutation_importance(
    rf_selector, X_mc, y,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)
perm_importance = pd.Series(
    perm_result.importances_mean,
    index=X_mc.columns
).sort_values(ascending=False)

# Unconditional drop: negative permutation importance
neg_perm_features = set(perm_importance[perm_importance < 0].index.tolist())

print(f"\n  Permutation Importance Results:")
for col, score in perm_importance.items():
    status = "⚠️ UNCONDITIONAL DROP (negative)" if score < 0 else ("✓" if col in rf_selected_builtin else "✗")
    leaky  = " ⚠️ LEAKY" if col in LEAKY_FEATURES else ""
    print(f"    {col:<45} perm_imp={score:+.6f}  {status}{leaky}")

# Final Method 2 selected = builtin selected AND NOT negative permutation
rf_selected = rf_selected_builtin - neg_perm_features
print(f"\n  Negative permutation importance features (UNCONDITIONAL DROP): {len(neg_perm_features)}")
for f in neg_perm_features:
    print(f"    FORCE DROP: {f}  (perm_imp={perm_importance[f]:+.6f})")
print(f"\n  Method 2 Final Selected: {len(rf_selected)} features")

# ── Plot: RF + Permutation Importance ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(22, max(10, len(rf_importance)*0.3)))

# Plot 1: RF built-in importance
ax1 = axes[0]
colors1 = [palette['success'] if c in rf_selected_builtin else palette['accent']
           for c in rf_importance.index]
bars1 = ax1.barh(range(len(rf_importance)), rf_importance.values,
                 color=colors1, edgecolor='white', alpha=0.85)
ax1.set_yticks(range(len(rf_importance)))
ax1.set_yticklabels(rf_importance.index, fontsize=8)
ax1.invert_yaxis()
ax1.set_xlabel('RF Feature Importance (Gini)', fontsize=10)
ax1.set_title(f'RF Built-in Importance\n(Top {rf_threshold_count} selected = green)', fontsize=11, fontweight='bold')
ax1.grid(axis='x', linestyle='--', alpha=0.4)

# Plot 2: Permutation importance
ax2 = axes[1]
perm_sorted = perm_importance.sort_values(ascending=False)
colors2 = [palette['accent'] if c in neg_perm_features else palette['success']
           for c in perm_sorted.index]
bars2 = ax2.barh(range(len(perm_sorted)), perm_sorted.values,
                 color=colors2, edgecolor='white', alpha=0.85)
ax2.set_yticks(range(len(perm_sorted)))
ax2.set_yticklabels(perm_sorted.index, fontsize=8)
ax2.invert_yaxis()
ax2.axvline(x=0, color=palette['accent'], linestyle='-', linewidth=1.5)
ax2.set_xlabel('Permutation Importance (mean decrease in accuracy)', fontsize=10)
ax2.set_title('Permutation Importance\n(Red/Negative = UNCONDITIONAL DROP)', fontsize=11, fontweight='bold')
ax2.grid(axis='x', linestyle='--', alpha=0.4)

plt.suptitle('Method 2: Random Forest Feature Importance', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'fs_03_rf_importance.png'), dpi=300, bbox_inches='tight')
plt.close('all')
print(f"\n  Saved: {PLOTS_DIR}/fs_03_rf_importance.png")


# =============================================================================
# STEP 5: METHOD 3 — RFECV (RECURSIVE FEATURE ELIMINATION WITH CV)
# =============================================================================
# WHY RFECV?
# Yeh sabse rigorous wrapper method hai:
#   1. Sare features se start karo
#   2. Sabse weak feature remove karo (RF importance ke basis par)
#   3. Cross-validation se performance check karo
#   4. Performance improve hoti rahe toh remove karte raho
#   5. Performance girne lage → ruk jao — yahi optimal set hai
#
# Scoring = f1_macro: kyunki target imbalanced hai (54.9% qty=1)
# Macro F1 har class ko equal weight deta hai — is dataset ke liye sahi
#
# cv = StratifiedKFold(5): target distribution har fold mein maintain ho
#
# n_estimators=100: RFECV mein RF ko fast rakhne ke liye kam trees
# RFECV kaafi slow hota hai — n_estimators reduce karte hain slightly
# =============================================================================

print("\n" + "="*65)
print("STEP 5: METHOD 3 — RFECV")
print("="*65)

print(f"\n  Running RFECV with StratifiedKFold(5), scoring=f1_macro...")
print(f"  (This is the most computationally expensive step — 5-10 minutes)")

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rfecv_estimator = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    min_samples_leaf=50,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rfecv_selector = RFECV(
    estimator=rfecv_estimator,
    step=1,
    cv=cv_strategy,
    scoring='f1_macro',
    min_features_to_select=5,
    n_jobs=-1
)
rfecv_selector.fit(X_mc, y)

rfecv_selected = set(X_mc.columns[rfecv_selector.support_].tolist())
optimal_n      = rfecv_selector.n_features_

print(f"\n  RFECV Results:")
print(f"  Optimal number of features: {optimal_n}")
print(f"  Features selected by RFECV : {len(rfecv_selected)}")

print(f"\n  RFECV Feature Rankings (1 = best):")
rfecv_ranking = pd.Series(rfecv_selector.ranking_, index=X_mc.columns).sort_values()
for col, rank in rfecv_ranking.items():
    status = "✓ SELECT" if col in rfecv_selected else "✗ REJECT"
    leaky  = " ⚠️ LEAKY" if col in LEAKY_FEATURES else ""
    print(f"    [{status}] rank={rank:3d}  {col:<45}{leaky}")

# ── Plot: RFECV CV Scores ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
n_features_range = range(rfecv_selector.min_features_to_select,
                         rfecv_selector.min_features_to_select + len(rfecv_selector.cv_results_['mean_test_score']))
mean_scores = rfecv_selector.cv_results_['mean_test_score']
std_scores  = rfecv_selector.cv_results_['std_test_score']

ax.plot(n_features_range, mean_scores, color=palette['highlight'], linewidth=2, marker='o', markersize=4)
ax.fill_between(n_features_range,
                mean_scores - std_scores,
                mean_scores + std_scores,
                alpha=0.2, color=palette['highlight'])
ax.axvline(x=optimal_n, color=palette['accent'], linestyle='--', linewidth=2,
           label=f'Optimal: {optimal_n} features')
ax.set_xlabel('Number of Features', fontsize=11)
ax.set_ylabel('CV F1-Macro Score', fontsize=11)
ax.set_title('Method 3: RFECV — Cross-Validation Score vs Number of Features\n'
             '(Peak = optimal feature count)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'fs_04_rfecv_cv_scores.png'), dpi=300, bbox_inches='tight')
plt.close('all')
print(f"\n  Saved: {PLOTS_DIR}/fs_04_rfecv_cv_scores.png")


# =============================================================================
# STEP 6: CONSENSUS VOTING — 2-of-3 RULE
# =============================================================================
# WHY CONSENSUS?
# Koi ek method akela perfect nahi:
#   MI      → statistical filter, model-agnostic but ignores interactions
#   RF imp  → captures interactions but biased toward high-cardinality
#   RFECV   → most rigorous but computationally sensitive
#
# CONSENSUS RULE:
#   ≥ 2 methods selected → FINAL KEEP  ✓
#   < 2 methods selected → FINAL DROP  ✗
#   Negative perm importance → FORCE DROP regardless of votes
#
# Yeh approach individual method ki kisi bhi bias ko cancel out karta hai.
# =============================================================================

print("\n" + "="*65)
print("STEP 6: CONSENSUS VOTING — 2-of-3 RULE")
print("="*65)

all_features = list(X_mc.columns)
voting_results = []

for feat in all_features:
    vote_mi    = 1 if feat in mi_selected    else 0
    vote_rf    = 1 if feat in rf_selected    else 0
    vote_rfecv = 1 if feat in rfecv_selected else 0
    total_votes = vote_mi + vote_rf + vote_rfecv

    # Force drop if negative permutation importance
    force_drop = feat in neg_perm_features

    if force_drop:
        decision = 'FORCE_DROP'
    elif total_votes >= 2:
        decision = 'KEEP'
    else:
        decision = 'DROP'

    voting_results.append({
        'feature':      feat,
        'vote_MI':      vote_mi,
        'vote_RF':      vote_rf,
        'vote_RFECV':   vote_rfecv,
        'total_votes':  total_votes,
        'force_drop':   force_drop,
        'decision':     decision,
        'is_leaky':     feat in LEAKY_FEATURES,
        'target_corr':  round(X_mc[feat].corr(y), 4),
    })

voting_df = pd.DataFrame(voting_results).sort_values(
    ['decision', 'total_votes'], ascending=[True, False]
)

final_keep    = voting_df[voting_df['decision'] == 'KEEP']['feature'].tolist()
final_drop    = voting_df[voting_df['decision'] == 'DROP']['feature'].tolist()
final_forcedrop = voting_df[voting_df['decision'] == 'FORCE_DROP']['feature'].tolist()

print(f"\n  CONSENSUS VOTING RESULTS:")
print(f"  {'Feature':<45} {'MI':>4} {'RF':>4} {'RFECV':>5} {'Votes':>5} {'TargCorr':>8}  Decision")
print(f"  {'-'*100}")

for _, row in voting_df.iterrows():
    leaky_tag = " ⚠️" if row['is_leaky'] else ""
    decision_icon = "✓ KEEP" if row['decision'] == 'KEEP' else ("⛔ FORCE_DROP" if row['decision'] == 'FORCE_DROP' else "✗ DROP")
    print(f"  {row['feature']:<45} {row['vote_MI']:>4} {row['vote_RF']:>4} {row['vote_RFECV']:>5} "
          f"{row['total_votes']:>5} {row['target_corr']:>8.4f}  {decision_icon}{leaky_tag}")

print(f"\n  ─────────────────────────────────────────────────────────")
print(f"  KEPT       : {len(final_keep):3d} features")
print(f"  DROPPED    : {len(final_drop):3d} features (< 2/3 votes)")
print(f"  FORCE DROP : {len(final_forcedrop):3d} features (negative permutation importance)")
print(f"  TOTAL DROP : {len(final_drop) + len(final_forcedrop):3d} features")

# Note on leaky features that were kept
leaky_kept = [f for f in final_keep if f in LEAKY_FEATURES]
if leaky_kept:
    print(f"\n  ⚠️  Leaky features in KEEP list (will be fixed in Modeling Pipeline):")
    for f in leaky_kept:
        print(f"      {f}  → will be replaced by sklearn TargetEncoder in CV Pipeline")


# =============================================================================
# STEP 7: FINAL FEATURE SET — BUILD + SAVE
# =============================================================================

print("\n" + "="*65)
print("STEP 7: FINAL FEATURE SET — BUILD + SAVE")
print("="*65)

# Build final dataset
final_features = final_keep  # these are from X_mc
# Add back variance-dropped and multicoll-dropped columns that we need to track
# (they're already excluded — X_mc only has kept columns)

# Also add columns dropped earlier (before X_mc) to make sure we account for all
all_dropped = (
    low_var_cols +          # Variance threshold
    multicoll_drop +        # Multicollinearity
    final_drop +            # Consensus < 2 votes
    final_forcedrop         # Negative permutation importance
)

print(f"\n  Final feature set: {len(final_features)} features + 1 target")

# Build final dataframe: target + selected features
df_final = df[[TARGET] + [f for f in final_features if f in df.columns]].copy()

# Note: OHE columns are in X_mc (derived from df) — need to bring them back
# Re-check which final features exist in df vs were created as OHE
ohe_in_final = [f for f in final_features if f in ohe_cols_created]
non_ohe_in_final = [f for f in final_features if f not in ohe_cols_created]

# Rebuild df_final properly
df_final = pd.DataFrame()
df_final[TARGET] = df[TARGET].values

for feat in final_features:
    if feat in df.columns:
        df_final[feat] = df[feat].values
    elif feat in X_mc.columns:
        df_final[feat] = X_mc[feat].values

print(f"\n  Final dataset shape: {df_final.shape[0]:,} rows × {df_final.shape[1]} columns")
print(f"\n  Final columns (target + {len(final_features)} features):")
for i, col in enumerate(df_final.columns, 1):
    is_target = "← TARGET" if col == TARGET else ""
    leaky_tag = "⚠️ LEAKY (fix in modeling)" if col in LEAKY_FEATURES else ""
    corr_val  = df_final[col].corr(df_final[TARGET]) if col != TARGET else "—"
    corr_str  = f"corr={corr_val:+.4f}" if isinstance(corr_val, float) else corr_val
    print(f"    {i:2}. {col:<45} {corr_str}  {is_target} {leaky_tag}")

# Save
output_path = os.path.join(DATA_DIR, 'Retail_demand_FS.csv')
df_final.to_csv(output_path, index=False)
print(f"\n  ✓ Saved: {output_path}")
print(f"  Rows    : {df_final.shape[0]:,}")
print(f"  Columns : {df_final.shape[1]}")


# =============================================================================
# STEP 8: VISUALIZATIONS + FINAL SUMMARY REPORT
# =============================================================================

print("\n" + "="*65)
print("STEP 8: VISUALIZATIONS")
print("="*65)

# ── Plot 1: Voting Summary Heatmap ────────────────────────────────────────────
print(f"\n  Generating voting summary heatmap...")

voting_display = voting_df.set_index('feature')[['vote_MI','vote_RF','vote_RFECV','total_votes']].copy()
voting_display = voting_display.sort_values('total_votes', ascending=False)

fig, ax = plt.subplots(figsize=(10, max(8, len(voting_display)*0.32)))
cmap_votes = sns.color_palette(['#E74C3C', '#27AE60'], as_cmap=False)

# Color: 1=green (selected), 0=red (not selected)
vote_data = voting_display[['vote_MI','vote_RF','vote_RFECV']].values

im = ax.imshow(vote_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Mutual\nInfo', 'Random\nForest', 'RFECV'], fontsize=10)
ax.set_yticks(range(len(voting_display)))
ax.set_yticklabels(voting_display.index, fontsize=8)

# Annotate cells
for i in range(len(voting_display)):
    for j in range(3):
        val = vote_data[i, j]
        text = "✓" if val == 1 else "✗"
        ax.text(j, i, text, ha='center', va='center', fontsize=10, fontweight='bold',
                color='white' if val == 1 else 'white')

# Add total votes on right
for i, (feat, row) in enumerate(voting_display.iterrows()):
    decision = voting_df[voting_df['feature'] == feat]['decision'].values[0]
    color = palette['success'] if decision == 'KEEP' else palette['accent']
    ax.text(3.3, i, f"{int(row['total_votes'])}/3  {'KEEP' if decision=='KEEP' else 'DROP'}",
            va='center', fontsize=8, color=color, fontweight='bold')

ax.set_title('Feature Selection: Consensus Voting Summary\n'
             '(Green=Selected, Red=Rejected | Right: Total Votes & Decision)',
             fontsize=12, fontweight='bold', pad=15)
ax.set_xlim(-0.5, 4.0)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'fs_05_voting_summary.png'), dpi=300, bbox_inches='tight')
plt.close('all')
print(f"  Saved: {PLOTS_DIR}/fs_05_voting_summary.png")

# ── Plot 2: Final Feature Correlations with Target ────────────────────────────
print(f"\n  Generating final feature correlation chart...")

final_corrs = df_final.drop(columns=[TARGET]).corrwith(df_final[TARGET]).sort_values(key=abs, ascending=False)

fig, ax = plt.subplots(figsize=(12, max(7, len(final_corrs)*0.33)))
bar_colors = [palette['success'] if v > 0 else palette['accent'] for v in final_corrs.values]
bars = ax.barh(range(len(final_corrs)), final_corrs.values, color=bar_colors, edgecolor='white', alpha=0.85)
ax.set_yticks(range(len(final_corrs)))
ax.set_yticklabels(final_corrs.index, fontsize=9)
ax.invert_yaxis()
ax.axvline(x=0, color=palette['primary'], linewidth=1)
ax.set_xlabel('Pearson Correlation with order_item_quantity', fontsize=11)
ax.set_title('Final Selected Features — Correlation with Target\n'
             '(Green = positive correlation, Red = negative correlation)',
             fontsize=12, fontweight='bold')
for bar, val in zip(bars, final_corrs.values):
    x_pos = bar.get_width() + 0.01 if val >= 0 else bar.get_width() - 0.01
    ha = 'left' if val >= 0 else 'right'
    ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:+.4f}', va='center', fontsize=8, ha=ha)
ax.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'fs_06_final_feature_correlations.png'), dpi=300, bbox_inches='tight')
plt.close('all')
print(f"  Saved: {PLOTS_DIR}/fs_06_final_feature_correlations.png")

# ── Plot 3: Feature Count Funnel ──────────────────────────────────────────────
print(f"\n  Generating feature reduction funnel...")

stages = [
    ('FE Output\n(Raw Input)', X.shape[1]),
    ('After Variance\nThreshold', X_var.shape[1]),
    ('After Multicoll.\nRemoval', X_mc.shape[1]),
    ('Final Selected\n(Consensus)', len(final_features)),
]

fig, ax = plt.subplots(figsize=(10, 5))
stage_names = [s[0] for s in stages]
stage_counts = [s[1] for s in stages]
bar_colors2 = [palette['highlight'], palette['teal'], palette['success'], palette['accent']]
bars3 = ax.bar(stage_names, stage_counts, color=bar_colors2, edgecolor='white', alpha=0.85, width=0.5)
for bar, val in zip(bars3, stage_counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val}', ha='center', fontsize=12, fontweight='bold', color=palette['primary'])
ax.set_ylabel('Number of Features', fontsize=11)
ax.set_title('Feature Reduction Funnel — Phase 9 Feature Selection\n'
             f'({X.shape[1]} → {len(final_features)} features)',
             fontsize=12, fontweight='bold')
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.set_ylim(0, max(stage_counts) * 1.15)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'fs_07_feature_reduction_funnel.png'), dpi=300, bbox_inches='tight')
plt.close('all')
print(f"  Saved: {PLOTS_DIR}/fs_07_feature_reduction_funnel.png")


# =============================================================================
# FINAL SUMMARY REPORT
# =============================================================================

print("\n" + "="*70)
print("  FEATURE SELECTION COMPLETE — FINAL SUMMARY REPORT")
print("="*70)

print(f"""
INPUT DATASET   : Retail_demand_FE.csv     ({df.shape[0]:,} rows × {df.shape[1]} columns)
OUTPUT DATASET  : Retail_demand_FS.csv     ({df_final.shape[0]:,} rows × {df_final.shape[1]} columns)

PRE-SELECTION FIXES APPLIED:
  ✓ is_fulfilled        DROPPED  — temporal leakage (future order status)
  ✓ order_item_product_price_log  DROPPED  — exact duplicate of product_price_log
  ✓ order_item_discount DROPPED  — replaced by has_discount + discount_intensity
  ✓ product_price       DROPPED  — product_price_log + is_high_price already present
  ✓ OHE applied         → customer_segment ({3} dummies), market ({5} dummies),
                           order_region ({23} dummies), order_status ({9} dummies)

FEATURE REDUCTION FUNNEL:
  FE Output (after pre-fixes) : {X.shape[1]:3d} features
  After Variance Threshold    : {X_var.shape[1]:3d} features  (dropped {len(low_var_cols)})
  After Multicollinearity     : {X_mc.shape[1]:3d} features  (dropped {len(multicoll_drop)})
  After Consensus Voting      : {len(final_features):3d} features  (dropped {len(final_drop)+len(final_forcedrop)})

SELECTION METHODS USED:
  Method 1 — Mutual Information       : {len(mi_selected)} features selected
  Method 2 — RF + Permutation Imp.    : {len(rf_selected)} features selected
  Method 3 — RFECV (5-fold, F1-macro) : {len(rfecv_selected)} features selected
  CONSENSUS (≥ 2/3 votes)            : {len(final_keep)} features KEPT

FINAL SELECTED FEATURES ({len(final_features)}):
""")

for i, feat in enumerate(final_keep, 1):
    leaky_tag = "  ⚠️ LEAKY — fix in Modeling via TargetEncoder Pipeline" if feat in LEAKY_FEATURES else ""
    row = voting_df[voting_df['feature'] == feat].iloc[0]
    print(f"  {i:2}. {feat:<45} [{int(row['vote_MI'])}MI + {int(row['vote_RF'])}RF + {int(row['vote_RFECV'])}RFECV = {int(row['total_votes'])}/3]{leaky_tag}")

print(f"""
DROPPED FEATURES SUMMARY:
  Pre-selection fixes  : 4 + OHE_internal
  Variance Threshold   : {len(low_var_cols)} features  → {low_var_cols if low_var_cols else 'None'}
  Multicollinearity    : {len(multicoll_drop)} features  → {multicoll_drop}
  Consensus < 2 votes  : {len(final_drop)} features
  Force Drop (neg perm): {len(final_forcedrop)} features  → {list(final_forcedrop) if final_forcedrop else 'None'}

⚠️  LEAKY FEATURES IN FINAL SET (TO FIX IN MODELING PHASE):
  These features were computed on full dataset → data leakage.
  In Modeling Phase: wrap in sklearn Pipeline with TargetEncoder.
  Cross-validation ke andar encoding hogi — leakage automatically prevent hogi.
""")
for lf in leaky_kept:
    print(f"  → {lf}")

print(f"""
MODELING RECOMMENDATIONS (Next Phase):
  • Models    : XGBoost, LightGBM, Random Forest (tree-based — non-normal data confirmed)
  • CV        : StratifiedKFold(5) — target imbalance handle karo
  • Metric    : F1-Macro (primary) — imbalanced target (54.9% qty=1)
  • Encoding  : sklearn Pipeline + TargetEncoder for leaky features
  • Imbalance : class_weight='balanced' ya SMOTE
  • Tuning    : RandomizedSearchCV (efficient) ya Optuna (advanced)
  • Baseline  : DummyClassifier F1-Macro ≈ 0.12 (beat karna zaroori hai)
  • Target    : F1-Macro > 0.50 (minimum) | > 0.65 (ideal)

PLOTS SAVED ({PLOTS_DIR}/):
  fs_01_correlation_heatmap.png
  fs_02_mutual_information.png
  fs_03_rf_importance.png
  fs_04_rfecv_cv_scores.png
  fs_05_voting_summary.png
  fs_06_final_feature_correlations.png
  fs_07_feature_reduction_funnel.png
""")

print("="*70)
print("FEATURE SELECTION COMPLETE ✓  |  Next Phase: Modeling")
print("="*70)
