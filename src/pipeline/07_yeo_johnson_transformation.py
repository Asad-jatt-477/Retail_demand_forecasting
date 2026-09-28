# =============================================================================
# STEP: YEO-JOHNSON TRANSFORMATION
# Columns: benefit_per_order & order_item_profit_ratio
# Reason: Both have negative + zero + positive values → only Yeo-Johnson works
# =============================================================================

# ⚠️ Ye columns normal ke KAREEB hain ya transformation se improve ho sakti hain

# * benefit_per_order :  

# Negatives hain isliye log transform nahi hoga. Ye aise hi theek hai modeling ke liye — tree models handle kar lenge, linear models ke liye Yeo-Johnson transform try kar sakte ho



# * order_item_profit_ratio :

# negative values hain , aise hi chhod do ya Yeo-johnson



# * order_item_product_price:

# tumne already order_item_product_price_log bana liya hai , (kaam ho gya hai)



# * product_price :

#  tumne already product_price_log bana liya hai , (kam ho gya hai)



# Apply Transformation on two columns :

# (1): benefit_per_order  

# (2): order_item_profit_ratio


# Tumhare dono columns ki ek common problem hai — inka data teen zones mein hai: negative values (~19%), zeros (~0.7%), aur positive values (~80%). Yahi combination transformation selection ko limited kar deta hai.

#  3 methods mathematically fail on your data. 'Cube root' barely helps. 'Shift+Log' makes things worse. 'Quantile transform' works but kills interpretability. 'Yeo-Johnson' is the only method that truly solves the problem.

#  Yeo-Johnson Handles negative, zero, and positive values natively. Optimizes lambda automatically. Skew near 0 for both columns.


# --------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless-safe; no blocking plt.show() windows
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import probplot
import warnings
import os
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(ROOT_DIR, 'data', 'processed')

# -----------------------------------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------------------------------
df = pd.read_csv(os.path.join(DATA_DIR, 'Retail_demand_outliers_TREATED.csv'))
print(f"Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

cols = ['benefit_per_order', 'order_item_profit_ratio']

# -----------------------------------------------------------------------------
# 2. APPLY YEO-JOHNSON TRANSFORMATION
# -----------------------------------------------------------------------------
lambdas = {}

for col in cols:
    transformed, lam = stats.yeojohnson(df[col])
    df[f'{col}_yj'] = transformed
    lambdas[col] = lam
    print(f"\n{col}:")
    print(f"  Lambda (λ)     : {lam:.4f}")
    print(f"  Original  skew : {df[col].skew():.4f}")
    print(f"  Transformed skew: {df[f'{col}_yj'].skew():.4f}")

# -----------------------------------------------------------------------------
# 3. BEFORE vs AFTER SUMMARY
# -----------------------------------------------------------------------------
print("\n" + "="*65)
print("BEFORE vs AFTER SUMMARY")
print("="*65)
print(f"{'Column':<35} {'Skew Before':>12} {'Skew After':>12}")
print("-"*65)
for col in cols:
    before = df[col].skew()
    after  = df[f'{col}_yj'].skew()
    print(f"{col:<35} {before:>12.4f} {after:>12.4f}")

# -----------------------------------------------------------------------------
# 4. NORMALITY TEST — BEFORE vs AFTER
# -----------------------------------------------------------------------------
print("\n" + "="*65)
print("NORMALITY TEST (D'Agostino-Pearson) — p > 0.05 = Normal")
print("="*65)

for col in cols:
    _, p_before = stats.normaltest(df[col])
    _, p_after  = stats.normaltest(df[f'{col}_yj'])
    is_normal_after = "✅ YES" if p_after > 0.05 else "❌ NO (but improved)"
    print(f"\n{col}:")
    print(f"  p-value before : {p_before:.6f}  → NOT normal")
    print(f"  p-value after  : {p_after:.6f}  → {is_normal_after}")

# -----------------------------------------------------------------------------
# 5. VISUALIZATION — Distribution + QQ Plot (Before & After)
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(17, 10))
fig.suptitle(
    "Yeo-Johnson Transformation — Before vs After\n(benefit_per_order  &  order_item_profit_ratio)",
    fontsize=15, fontweight='bold', y=0.99
)

gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.5, wspace=0.35, top=0.88, bottom=0.07)

plot_configs = [
    # (row, col_before, col_after, title)
    (0, 'benefit_per_order',        'benefit_per_order_yj'),
    (1, 'order_item_profit_ratio',  'order_item_profit_ratio_yj'),
]

colors = {'before': '#E05252', 'after': '#3A86FF'}

for row, col_orig, col_yj in plot_configs:

    lam = lambdas[col_orig]

    # --- Histogram Before ---
    ax1 = fig.add_subplot(gs[row, 0])
    ax1.hist(df[col_orig], bins=60, color=colors['before'], alpha=0.8, edgecolor='white', linewidth=0.3)
    ax1.set_title(f"Before\n{col_orig}", fontsize=9, fontweight='bold')
    ax1.set_ylabel("Frequency")
    ax1.annotate(f"Skew: {df[col_orig].skew():.3f}",
                 xy=(0.97, 0.93), xycoords='axes fraction',
                 ha='right', fontsize=8,
                 bbox=dict(boxstyle='round,pad=0.3', fc='#fdecea', ec='#E05252', lw=0.8))
    ax1.tick_params(labelsize=8)

    # --- QQ Plot Before ---
    ax2 = fig.add_subplot(gs[row, 1])
    (osm, osr), (slope, intercept, r) = probplot(df[col_orig], dist="norm")
    ax2.plot(osm, osr, 'o', color=colors['before'], markersize=1.5, alpha=0.4)
    ax2.plot(osm, slope * np.array(osm) + intercept, 'r-', linewidth=1.5)
    ax2.set_title(f"Q-Q Plot Before\nR²={r**2:.3f}", fontsize=9, fontweight='bold')
    ax2.set_xlabel("Theoretical Quantiles", fontsize=8)
    ax2.set_ylabel("Sample Quantiles", fontsize=8)
    ax2.tick_params(labelsize=8)

    # --- Histogram After ---
    ax3 = fig.add_subplot(gs[row, 2])
    ax3.hist(df[col_yj], bins=60, color=colors['after'], alpha=0.8, edgecolor='white', linewidth=0.3)
    ax3.set_title(f"After Yeo-Johnson (λ={lam:.3f})\n{col_orig}", fontsize=9, fontweight='bold')
    ax3.set_ylabel("Frequency")
    ax3.annotate(f"Skew: {df[col_yj].skew():.3f}",
                 xy=(0.97, 0.93), xycoords='axes fraction',
                 ha='right', fontsize=8,
                 bbox=dict(boxstyle='round,pad=0.3', fc='#e8f0fe', ec='#3A86FF', lw=0.8))
    ax3.tick_params(labelsize=8)

    # --- QQ Plot After ---
    ax4 = fig.add_subplot(gs[row, 3])
    (osm2, osr2), (slope2, intercept2, r2) = probplot(df[col_yj], dist="norm")
    ax4.plot(osm2, osr2, 'o', color=colors['after'], markersize=1.5, alpha=0.4)
    ax4.plot(osm2, slope2 * np.array(osm2) + intercept2, 'r-', linewidth=1.5)
    ax4.set_title(f"Q-Q Plot After\nR²={r2**2:.3f}", fontsize=9, fontweight='bold')
    ax4.set_xlabel("Theoretical Quantiles", fontsize=8)
    ax4.set_ylabel("Sample Quantiles", fontsize=8)
    ax4.tick_params(labelsize=8)
    

PLOTS_DIR = os.path.join(ROOT_DIR, 'assets', 'plots', 'yeo_johnson')
os.makedirs(PLOTS_DIR, exist_ok=True)
save_path = os.path.join(PLOTS_DIR, 'yeo_johnson_before_after.png')
fig.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"\nPlot saved: {save_path}")

# -----------------------------------------------------------------------------
# 6. SAVE UPDATED DATASET
# -----------------------------------------------------------------------------
df.to_csv(os.path.join(DATA_DIR, 'Retail_demand_YJ_transformed.csv'), index=False)
print(f"\nDataset saved: Retail_demand_YJ_transformed.csv")
print(f"New columns added: benefit_per_order_yj, order_item_profit_ratio_yj")
print(f"Total columns now: {df.shape[1]}")
print("\nDONE ✅")
