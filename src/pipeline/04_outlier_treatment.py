# =============================================================================
# 🔧 Outliers Treatment — Retail Demand Forecasting
#
# | Column                       | Method                                  |
# |------------------------------|-----------------------------------------|
# | `benefit_per_order`          | Winsorization (1st – 99th percentile)   |
# | `order_item_profit_ratio`    | Winsorization (1st – 99th percentile)   |
# | `order_item_discount`        | Winsorization (1st – 99th percentile)   |  ← NEW
# | `product_price`              | Log Transformation (`log1p`)            |
# | `order_item_product_price`   | Log Transformation (`log1p`)            |
# | `sales`                      | Log Transformation (`log1p`)            |  ← NEW
# | `order_item_total`           | Log Transformation (`log1p`)            |  ← NEW
# | `sales_per_customer`         | Log Transformation (`log1p`)            |  ← NEW
# | `latitude`                   | Keep as-is (no treatment)               |
# | `longitude`                  | Keep as-is (no treatment)               |
#
# WHY THESE TREATMENTS:
#
# Winsorization → jab column mein negative/zero values meaningful hain aur
#                 ek natural business upper/lower bound exist karta hai.
#                 Extreme tails cap hoti hain, rows delete nahi hoti.
#
# log1p        → jab column strictly positive ho, right-skewed ho (skew > 1),
#                 aur high values real business events hain (not errors).
#                 log(1+x) zero-safe hai aur skewness compress karta hai.
#
# No treatment → jab outliers statistically flagged hon lekin domain mein
#                 valid data ho (e.g. geographic coordinates, real discounts).
# =============================================================================


# ── Step 1: Imports ──────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless-safe; no blocking plt.show() windows
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(ROOT_DIR, 'data', 'processed')
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# OUTPUT FOLDER SETUP
# =============================================================================
PLOTS_DIR = os.path.join(ROOT_DIR, 'assets', 'plots', 'outlier_treatment')
os.makedirs(PLOTS_DIR, exist_ok=True)
print(f"Plots folder ready: '{PLOTS_DIR}/'")

# ── Plot Style ──────────────────────────────────────────────────────────────
palette = {
    'primary':   '#2C3E50',
    'accent':    '#E74C3C',
    'highlight': '#3498DB',
    'success':   '#27AE60',
    'warn':      '#F39C12',
    'bg':        '#FAFAFA',
    'grid':      '#ECEFF1'
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
    'axes.titlesize':   11
})

print('✅ Libraries loaded successfully.')


# ── Step 2: Load Dataset ─────────────────────────────────────────────────────

df = pd.read_csv(os.path.join(DATA_DIR, 'Retail_demand_forecasting_CLEANED.csv'), encoding='latin-1')

print(f'✅ Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns')

# Sab treatment columns — original + 4 new
all_treatment_cols = [
    'benefit_per_order',
    'order_item_profit_ratio',
    'order_item_discount',
    'product_price',
    'order_item_product_price',
    'sales',
    'order_item_total',
    'sales_per_customer',
]

print(f'\nAll target columns for treatment:')
print(df[all_treatment_cols].describe().round(4))


# ── Step 3: Utility Functions ─────────────────────────────────────────────────

def iqr_outlier_count(series):
    """IQR-based outlier count utility."""
    Q1  = series.quantile(0.25)
    Q3  = series.quantile(0.75)
    IQR = Q3 - Q1
    return int(((series < Q1 - 1.5*IQR) | (series > Q3 + 1.5*IQR)).sum())

def winsorize_column(df, col, lower_pct=0.01, upper_pct=0.99):
    """
    Winsorize a column by capping at given percentiles.
    Returns the capped series and the cap values used.
    """
    lower_cap = df[col].quantile(lower_pct)
    upper_cap = df[col].quantile(upper_pct)
    capped    = df[col].clip(lower=lower_cap, upper=upper_cap)
    return capped, lower_cap, upper_cap


# ── Step 4: Before-Treatment Snapshot ────────────────────────────────────────
# Baseline statistics — taake baad mein compare kar sakein.

print('\n' + '=' * 75)
print('BEFORE TREATMENT — BASELINE SNAPSHOT')
print('=' * 75)
print(f'{"COLUMN":<30} {"MIN":>10} {"MAX":>10} {"SKEW":>8} {"OUTLIERS":>10}')
print('-' * 75)
for col in all_treatment_cols:
    s = df[col]
    print(f'{col:<30} {s.min():>10.2f} {s.max():>10.2f} '
          f'{s.skew():>8.3f} {iqr_outlier_count(s):>10,}')
print('=' * 75)


# ── Before distribution plots (ALL 8 columns) ────────────────────────────────
print("\n[1/5] Generating Before Treatment plots (all 8 columns)...")

n_cols_plot = 4
n_rows_plot = 2  # 8 columns → 2 rows of 4

fig, axes = plt.subplots(n_rows_plot * 2, n_cols_plot, figsize=(22, 18))
fig.suptitle('BEFORE Treatment — Distributions & Box Plots (All 8 Columns)',
             fontsize=15, fontweight='bold', color=palette['primary'], y=0.995)

for i, col in enumerate(all_treatment_cols):
    row_block = i // n_cols_plot        # 0 for first 4, 1 for last 4
    col_idx   = i  % n_cols_plot        # 0-3

    hist_row  = row_block * 2
    box_row   = row_block * 2 + 1

    data = df[col].dropna()

    # Histogram
    ax_h = axes[hist_row][col_idx]
    ax_h.hist(data, bins=60, color=palette['highlight'], alpha=0.65,
              edgecolor='white', linewidth=0.3, density=True)
    ax_h.set_title(f'{col}\nSkew: {data.skew():.3f}  |  Outliers: {iqr_outlier_count(data):,}',
                   fontsize=9, pad=8)
    ax_h.set_xlabel('Value', fontsize=8)
    ax_h.set_ylabel('Density', fontsize=8)
    ax_h.grid(linestyle='--', alpha=0.4)
    ax_h.tick_params(labelsize=8)

    # Box plot
    ax_b = axes[box_row][col_idx]
    ax_b.boxplot(data, patch_artist=True, vert=True,
                 flierprops=dict(marker='o', color=palette['accent'],
                                markersize=2, alpha=0.4,
                                markerfacecolor=palette['accent']),
                 medianprops=dict(color=palette['success'], linewidth=2),
                 boxprops=dict(facecolor='#AED6F1', color=palette['highlight']),
                 whiskerprops=dict(color=palette['primary']),
                 capprops=dict(color=palette['primary']))
    ax_b.set_title(f'IQR Outliers: {iqr_outlier_count(data):,}',
                   fontsize=9, color=palette['accent'], pad=8)
    ax_b.grid(axis='y', linestyle='--', alpha=0.4)
    ax_b.tick_params(labelsize=8)

fig.tight_layout(rect=[0, 0.01, 1, 0.95])
save_path = os.path.join(PLOTS_DIR, '01_before_treatment_all8.png')
fig.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'   Saved: {save_path}')


# =============================================================================
# ── Step 5: WINSORIZATION
#    Columns: benefit_per_order, order_item_profit_ratio, order_item_discount
#
# benefit_per_order:
#   - 18.71% negative values = real business losses → delete nahi kar sakte
#   - Extreme tails (bulk returns, fraud) cap karo
#
# order_item_profit_ratio:
#   - Ratio column — mathematically -1 se neeche jaana business logic se bahar
#   - Valid range enforce karna zaroori hai
#
# order_item_discount:
#   - Zero discount wale orders bhi hain (zero = "no discount" meaningful signal)
#   - Business mein discount ki natural upper bound hoti hai
#   - Extraordinary extreme values (data entry errors / one-off events) cap karo
#   - log1p yahan better nahi kyunki zero ka business meaning preserve karna hai
# =============================================================================

print('\n' + '=' * 60)
print('STEP 5: WINSORIZATION (3 columns)')
print('=' * 60)

winsor_cols = ['benefit_per_order', 'order_item_profit_ratio', 'order_item_discount']
winsors     = {}

for col in winsor_cols:
    capped, lo, hi = winsorize_column(df, col)
    winsors[col]   = {'lower_cap': lo, 'upper_cap': hi}
    df[col]        = capped

    print(f'\n✅ {col}')
    print(f'   Reason     →  ', end='')
    if col == 'benefit_per_order':
        print('Negative values = real losses | extreme tails capped')
    elif col == 'order_item_profit_ratio':
        print('Ratio must stay in business-valid range')
    elif col == 'order_item_discount':
        print('Zero = meaningful (no discount) | business ceiling enforced')
    print(f'   Capped at  →  Lower: {lo:.4f}  |  Upper: {hi:.4f}')
    print(f'   New range  →  Min:   {df[col].min():.4f}  |  Max:   {df[col].max():.4f}')
    print(f'   New skew   →  {df[col].skew():.4f}')
    print(f'   Outliers   →  {iqr_outlier_count(df[col]):,} (after winsorization)')


# =============================================================================
# ── Step 6: LOG TRANSFORMATION (log1p)
#    Columns: product_price, order_item_product_price,
#             sales, order_item_total, sales_per_customer
#
# product_price / order_item_product_price:
#   - Product catalog prices — $1,999 ek real high-end product hai
#   - Right-skewed (skew ≈ 3.2) — ML models ke liye problematic
#
# sales:
#   - sales = price × quantity — high-value orders real transactions hain
#   - Skew ≈ 2.92, strictly positive — log1p perfect fit
#
# order_item_total:
#   - Per-line-item total, sales se r=0.99 correlation
#   - Same distribution issue — consistency ke liye same transform
#
# sales_per_customer:
#   - High-value customers real hain (premium/bulk buyers)
#   - Skew ≈ 2.92, strictly positive — log1p compress karta hai without loss
#
# NOTE: Original columns preserved — _log versions ML training ke liye use karo
# =============================================================================

print('\n' + '=' * 60)
print('STEP 6: LOG TRANSFORMATION — log1p (5 columns)')
print('=' * 60)

log_cols = [
    'product_price',
    'order_item_product_price',
    'sales',
    'order_item_total',
    'sales_per_customer',
]

for col in log_cols:
    new_col = col + '_log'

    # Safety check — log undefined for negative values
    min_val = df[col].min()
    assert min_val >= 0, f'❌ {col} has negative values ({min_val:.4f}) — log1p unsafe!'

    df[new_col] = np.log1p(df[col])

    print(f'\n✅ {col}  →  {new_col}')
    print(f'   Reason     →  ', end='')
    if col == 'product_price':
        print('Catalog price right-skewed — real high-end products exist')
    elif col == 'order_item_product_price':
        print('Per-item price right-skewed — same catalog price logic')
    elif col == 'sales':
        print('price × qty — high-value bulk orders are real transactions')
    elif col == 'order_item_total':
        print('qty-linked total, r=0.99 with sales — consistency with sales transform')
    elif col == 'sales_per_customer':
        print('Premium/bulk buyers naturally exist — high values are real')
    print(f'   Original   →  Skew: {df[col].skew():.4f}  '
          f'| Min: {df[col].min():.2f}  | Max: {df[col].max():.2f}')
    print(f'   After log  →  Skew: {df[new_col].skew():.4f}  '
          f'| Min: {df[new_col].min():.4f}  | Max: {df[new_col].max():.4f}')
    print(f'   Outliers   →  {iqr_outlier_count(df[new_col]):,} (after log transform)')

print('\nℹ️  Original columns preserved. Use _log versions for ML training.')


# ── Step 7: Geographic Columns — No Treatment ────────────────────────────────
# Valid geographic coordinates hain — multi-region global dataset.
# Statistical methods ne minority-region coords ko outlier flag kiya
# sirf isliye ke majority orders ek region se hain.
# Treat karna = data corruption.

print('\n' + '=' * 60)
print('STEP 7: GEOGRAPHIC COLUMNS — NO TREATMENT')
print('=' * 60)
print('ℹ️  latitude  & longitude — No treatment applied.')
print(f'   latitude   →  Min: {df["latitude"].min():.4f}  | Max: {df["latitude"].max():.4f}')
print(f'   longitude  →  Min: {df["longitude"].min():.4f}  | Max: {df["longitude"].max():.4f}')
print('   Reason     →  Valid geographic coordinates (multi-region global dataset)')


# ── Step 8: After-Treatment Snapshot ─────────────────────────────────────────

# After snapshot columns (winsorized in-place + log versions)
after_cols = [
    'benefit_per_order',            # winsorized in-place
    'order_item_profit_ratio',      # winsorized in-place
    'order_item_discount',          # winsorized in-place  ← NEW
    'product_price_log',
    'order_item_product_price_log',
    'sales_log',                    # ← NEW
    'order_item_total_log',         # ← NEW
    'sales_per_customer_log',       # ← NEW
]

print('\n' + '=' * 75)
print('AFTER TREATMENT — FINAL SNAPSHOT')
print('=' * 75)
print(f'{"COLUMN":<36} {"MIN":>8} {"MAX":>8} {"SKEW":>8} {"OUTLIERS":>10}')
print('-' * 75)
for col in after_cols:
    s = df[col]
    print(f'{col:<36} {s.min():>8.3f} {s.max():>8.3f} '
          f'{s.skew():>8.3f} {iqr_outlier_count(s):>10,}')
print('=' * 75)


# ── After distribution plots (ALL 8 treated versions) ────────────────────────
print("\n[2/5] Generating After Treatment plots (all 8 treated columns)...")

fig, axes = plt.subplots(n_rows_plot * 2, n_cols_plot, figsize=(22, 18))
fig.suptitle('AFTER Treatment — Distributions & Box Plots (All 8 Columns)',
             fontsize=15, fontweight='bold', color=palette['success'], y=0.995)

for i, col in enumerate(after_cols):
    row_block = i // n_cols_plot
    col_idx   = i  % n_cols_plot
    hist_row  = row_block * 2
    box_row   = row_block * 2 + 1

    data = df[col].dropna()

    ax_h = axes[hist_row][col_idx]
    ax_h.hist(data, bins=60, color=palette['success'], alpha=0.65,
              edgecolor='white', linewidth=0.3, density=True)
    ax_h.set_title(f'{col}\nSkew: {data.skew():.3f}  |  Outliers: {iqr_outlier_count(data):,}',
                   fontsize=9, pad=8)
    ax_h.set_xlabel('Value', fontsize=8)
    ax_h.set_ylabel('Density', fontsize=8)
    ax_h.grid(linestyle='--', alpha=0.4)
    ax_h.tick_params(labelsize=8)

    ax_b = axes[box_row][col_idx]
    ax_b.boxplot(data, patch_artist=True, vert=True,
                 flierprops=dict(marker='o', color=palette['warn'],
                                markersize=2, alpha=0.4,
                                markerfacecolor=palette['warn']),
                 medianprops=dict(color=palette['accent'], linewidth=2),
                 boxprops=dict(facecolor='#A9DFBF', color=palette['success']),
                 whiskerprops=dict(color=palette['primary']),
                 capprops=dict(color=palette['primary']))
    ax_b.set_title(f'IQR Outliers: {iqr_outlier_count(data):,}',
                   fontsize=9, color=palette['success'], pad=8)
    ax_b.grid(axis='y', linestyle='--', alpha=0.4)
    ax_b.tick_params(labelsize=8)

fig.tight_layout(rect=[0, 0.01, 1, 0.95])
save_path = os.path.join(PLOTS_DIR, '02_after_treatment_all8.png')
fig.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'   Saved: {save_path}')


# ── Step 9: Before vs After Comparison — Winsorized Columns ──────────────────
print("\n[3/5] Generating Before vs After — Winsorization columns...")

df_orig = pd.read_csv(os.path.join(DATA_DIR, 'Retail_demand_forecasting_CLEANED.csv'), encoding='latin-1')

winsor_compare = [
    ('benefit_per_order',        'benefit_per_order',        'Winsorization'),
    ('order_item_profit_ratio',  'order_item_profit_ratio',  'Winsorization'),
    ('order_item_discount',      'order_item_discount',      'Winsorization'),
]

fig, axes = plt.subplots(3, 2, figsize=(15, 14))
fig.suptitle('Before vs After — Winsorized Columns (3 columns)',
             fontsize=15, fontweight='bold', color=palette['primary'], y=0.995)

for row_i, (orig_col, treated_col, method) in enumerate(winsor_compare):
    before = df_orig[orig_col].dropna()
    after  = df[treated_col].dropna()

    ax1 = axes[row_i][0]
    ax1.hist(before, bins=80, color=palette['accent'], alpha=0.65,
             edgecolor='white', linewidth=0.2, density=True)
    ax1.set_title(f'BEFORE: {orig_col}\nSkew: {before.skew():.3f}  '
                  f'| Outliers: {iqr_outlier_count(before):,}',
                  fontsize=10, color=palette['accent'], pad=8)
    ax1.grid(linestyle='--', alpha=0.4)
    ax1.tick_params(labelsize=9)

    ax2 = axes[row_i][1]
    ax2.hist(after, bins=80, color=palette['success'], alpha=0.65,
             edgecolor='white', linewidth=0.2, density=True)
    ax2.set_title(f'AFTER ({method}): {treated_col}\nSkew: {after.skew():.3f}  '
                  f'| Outliers: {iqr_outlier_count(after):,}',
                  fontsize=10, color=palette['success'], pad=8)
    ax2.grid(linestyle='--', alpha=0.4)
    ax2.tick_params(labelsize=9)

fig.tight_layout(rect=[0, 0.01, 1, 0.94])
save_path = os.path.join(PLOTS_DIR, '03_before_vs_after_winsorized.png')
fig.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'   Saved: {save_path}')


# ── Step 10: Before vs After Comparison — Log Transformed Columns ─────────────
print("\n[4/5] Generating Before vs After — Log Transform columns...")

log_compare = [
    ('product_price',            'product_price_log',            'log1p'),
    ('order_item_product_price', 'order_item_product_price_log', 'log1p'),
    ('sales',                    'sales_log',                    'log1p'),
    ('order_item_total',         'order_item_total_log',         'log1p'),
    ('sales_per_customer',       'sales_per_customer_log',       'log1p'),
]

fig, axes = plt.subplots(5, 2, figsize=(15, 20))
fig.suptitle('Before vs After — Log Transformed Columns (5 columns)',
             fontsize=15, fontweight='bold', color=palette['primary'], y=0.995)

for row_i, (orig_col, treated_col, method) in enumerate(log_compare):
    before = df_orig[orig_col].dropna()
    after  = df[treated_col].dropna()

    ax1 = axes[row_i][0]
    ax1.hist(before, bins=80, color=palette['accent'], alpha=0.65,
             edgecolor='white', linewidth=0.2, density=True)
    ax1.set_title(f'BEFORE: {orig_col}\nSkew: {before.skew():.3f}  '
                  f'| Outliers: {iqr_outlier_count(before):,}',
                  fontsize=10, color=palette['accent'], pad=8)
    ax1.grid(linestyle='--', alpha=0.4)
    ax1.tick_params(labelsize=9)

    ax2 = axes[row_i][1]
    ax2.hist(after, bins=80, color=palette['success'], alpha=0.65,
             edgecolor='white', linewidth=0.2, density=True)
    ax2.set_title(f'AFTER ({method}): {treated_col}\nSkew: {after.skew():.3f}  '
                  f'| Outliers: {iqr_outlier_count(after):,}',
                  fontsize=10, color=palette['success'], pad=8)
    ax2.grid(linestyle='--', alpha=0.4)
    ax2.tick_params(labelsize=9)

fig.tight_layout(rect=[0, 0.01, 1, 0.96])
save_path = os.path.join(PLOTS_DIR, '04_before_vs_after_log_transformed.png')
fig.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'   Saved: {save_path}')


# ── Step 11: Save Treated Dataset ────────────────────────────────────────────
print("\n[5/5] Saving treated dataset...")

output_file = os.path.join(DATA_DIR, 'Retail_demand_outliers_TREATED.csv')
df.to_csv(output_file, index=False)

# New columns added count
new_log_cols   = [c + '_log' for c in log_cols]
total_new_cols = len(new_log_cols)   # 5 new _log columns

print('\n' + '=' * 62)
print('  ✅ TREATMENT COMPLETE — FINAL REPORT')
print('=' * 62)
print(f'  Saved as  :  {output_file}')
print(f'  Rows      :  {df.shape[0]:,}')
print(f'  Columns   :  {df.shape[1]}  '
      f'(original {df_orig.shape[1]} + {total_new_cols} new _log columns)')
print()
print('  COLUMNS TREATED:')
print('  ┌─────────────────────────────────────────┬──────────────────────────┐')
print('  │ Column                                  │ Method                   │')
print('  ├─────────────────────────────────────────┼──────────────────────────┤')
print('  │ benefit_per_order                       │ Winsorization (1–99%)    │')
print('  │ order_item_profit_ratio                 │ Winsorization (1–99%)    │')
print('  │ order_item_discount              ← NEW  │ Winsorization (1–99%)    │')
print('  ├─────────────────────────────────────────┼──────────────────────────┤')
print('  │ product_price            → _log         │ log1p Transform          │')
print('  │ order_item_product_price → _log         │ log1p Transform          │')
print('  │ sales                    → _log  ← NEW  │ log1p Transform          │')
print('  │ order_item_total         → _log  ← NEW  │ log1p Transform          │')
print('  │ sales_per_customer       → _log  ← NEW  │ log1p Transform          │')
print('  ├─────────────────────────────────────────┼──────────────────────────┤')
print('  │ latitude                                │ No treatment             │')
print('  │ longitude                               │ No treatment             │')
print('  └─────────────────────────────────────────┴──────────────────────────┘')
print()
print('  📌 NOTES:')
print('     1. Original price/sales columns preserved in dataset.')
print('        Use _log versions for ML model training.')
print('     2. sales, order_item_total, sales_per_customer — potential leakage')
print('        columns. Evaluate carefully before including in final model.')
print('     3. Winsorized columns modified in-place (no new column created).')
print('=' * 62)


# =============================================================================
# FINAL PLOTS SUMMARY
# =============================================================================
saved_files = sorted(os.listdir(PLOTS_DIR))
print(f"\n{'='*60}")
print(f"ALL PLOTS SAVED  →  folder: '{PLOTS_DIR}/'")
print(f"{'='*60}")
for f in saved_files:
    print(f"   ✅  {f}")
print(f"\nTotal plot files: {len(saved_files)}")
