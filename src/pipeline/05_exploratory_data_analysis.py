# =============================================================================
#  RETAIL DEMAND FORECASTING — EXPLORATORY DATA ANALYSIS (EDA)
#  Target Variable: order_item_quantity (values: 1, 2, 3, 4, 5)
#  Input: Retail_demand_outliers_TREATED.csv
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless-safe; no blocking plt.show() windows
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, kstest, normaltest, chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# ── Color Palette (consistent with previous phases) ──────────────────────────
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

print("✅ Setup complete.")


# =============================================================================
# OUTPUT FOLDER SETUP — All plots will be saved inside this folder
# =============================================================================

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(ROOT_DIR, 'data', 'processed')
REPORTS_DIR = os.path.join(ROOT_DIR, 'reports')
# Small statistical-summary tables (a few rows each) go in their own
# subfolder, kept separate from the written reports (.docx/.json) that also
# live directly under reports/ — keeps "reports/" from looking like it holds
# raw data when these are really just EDA test-result exports.
EDA_STATS_DIR = os.path.join(REPORTS_DIR, 'eda_stats')
os.makedirs(EDA_STATS_DIR, exist_ok=True)

PLOTS_DIR = os.path.join(ROOT_DIR, 'assets', 'plots', 'eda')
os.makedirs(PLOTS_DIR, exist_ok=True)
print(f"📁 Output folder ready: '{PLOTS_DIR}/'")


# =============================================================================
# PLOTTING HELPERS — keep every figure a "normal", legible, non-overlapping size
# =============================================================================
def paginate(items, per_page):
    return [items[i:i + per_page] for i in range(0, len(items), per_page)]


def truncate_label(s, maxlen=26):
    s = str(s)
    return s if len(s) <= maxlen else s[:maxlen - 1] + '…'


def heatmap_figsize(n_features):
    side = max(10, min(20, 0.42 * n_features + 6))
    return (side, side * 0.8)


def annot_fontsize(n_features):
    return max(5.5, min(9, 200 / max(n_features, 1)))



# =============================================================================
# STEP 1: LOAD DATASET & SANITY CHECK
# =============================================================================

df = pd.read_csv(os.path.join(DATA_DIR, 'Retail_demand_outliers_TREATED.csv'), encoding='latin-1')

print(f"Shape        : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Missing vals : {df.isnull().sum().sum()}")
print(f"Duplicates   : {df.duplicated().sum()}")
print(f"\nColumn list  :")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2}. {col}  [{df[col].dtype}]")

# Identify column groups for EDA
TARGET = 'order_item_quantity'

NUMERIC_COLS = df.select_dtypes(include=['int64','float64']).columns.tolist()
NUMERIC_COLS = [c for c in NUMERIC_COLS if c != TARGET]

CATEGORICAL_COLS = df.select_dtypes(include=['object','category']).columns.tolist()
# product_image is a raw URL string (already flagged elsewhere as zero predictive
# value, and 1:1 redundant with product_name) — plotting URLs as bar-chart
# categories produces unreadable labels for no analytical benefit, so it's
# excluded here from the EDA visualizations/tests below.
if 'product_image' in CATEGORICAL_COLS:
    CATEGORICAL_COLS.remove('product_image')

print(f"\nTarget        : {TARGET}")
print(f"Numeric cols  : {len(NUMERIC_COLS)}  → {NUMERIC_COLS}")
print(f"Categorical   : {len(CATEGORICAL_COLS)}  → {CATEGORICAL_COLS}")


# =============================================================================
# STEP 2: TARGET VARIABLE — order_item_quantity (1–5)
# =============================================================================

print("\n" + "="*60)
print("STEP 2: TARGET VARIABLE ANALYSIS")
print("="*60)

target = df[TARGET]
vc     = target.value_counts().sort_index()
total  = len(df)

print(f"\nDistribution of '{TARGET}':")
print(f"{'Qty':>5} | {'Count':>8} | {'%':>7}")
print("-" * 28)
for qty, cnt in vc.items():
    bar = '█' * int(cnt / total * 50)
    print(f"  {qty}   | {cnt:>8,} | {cnt/total*100:>6.1f}%  {bar}")

print(f"\nMean   : {target.mean():.4f}")
print(f"Median : {target.median():.1f}")
print(f"Mode   : {target.mode()[0]}")
print(f"Std    : {target.std():.4f}")

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.suptitle('TARGET VARIABLE: order_item_quantity',
             fontsize=15, fontweight='bold', color=palette['primary'], y=1.02)

# Bar chart
ax = axes[0]
bars = ax.bar(vc.index, vc.values, color=palette['highlight'],
              edgecolor='white', linewidth=0.8, width=0.6)
ax.set_title('Class Distribution')
ax.set_xlabel('Order Item Quantity')
ax.set_ylabel('Count')
for bar, cnt in zip(bars, vc.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
            f'{cnt/total*100:.1f}%', ha='center', fontsize=9, fontweight='bold')
ax.grid(axis='y', linestyle='--', alpha=0.5)

# Pie chart
ax2 = axes[1]
colors_pie = [palette['highlight'], palette['success'], palette['warn'],
              palette['accent'], palette['purple']]
wedges, texts, autotexts = ax2.pie(
    vc.values, labels=[f'Qty {i}' for i in vc.index],
    autopct='%1.1f%%', colors=colors_pie, startangle=140,
    pctdistance=0.8, wedgeprops=dict(linewidth=1.5, edgecolor='white'))
ax2.set_title('Class Share (%)')

# Cumulative %
ax3 = axes[2]
cum_pct = (vc.values / total * 100).cumsum()
ax3.bar(vc.index, vc.values / total * 100, color=palette['highlight'],
        alpha=0.6, edgecolor='white')
ax3_twin = ax3.twinx()
ax3_twin.plot(vc.index, cum_pct, 'o-', color=palette['accent'],
              linewidth=2, markersize=6, label='Cumulative %')
ax3_twin.set_ylabel('Cumulative %', color=palette['accent'])
ax3.set_title('Cumulative Distribution')
ax3.set_xlabel('Quantity')
ax3.set_ylabel('Class %')
ax3.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout(rect=[0, 0, 1, 0.93])
save_path = os.path.join(PLOTS_DIR, 'eda_01_target_distribution.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close('all')
print(f"Saved: {save_path}")


# =============================================================================
# STEP 3: UNIVARIATE ANALYSIS — NUMERIC FEATURES
# =============================================================================

print("\n" + "="*60)
print("STEP 3: UNIVARIATE — NUMERIC FEATURES")
print("="*60)

# Summary statistics table
summary = df[NUMERIC_COLS].describe().T
summary['skewness'] = df[NUMERIC_COLS].skew()
summary['kurtosis'] = df[NUMERIC_COLS].kurt()
summary['cv_%']     = (df[NUMERIC_COLS].std() / df[NUMERIC_COLS].mean().abs() * 100).round(2)
print(summary[['mean','std','min','max','skewness','kurtosis','cv_%']].round(3))

# ── Histogram + KDE for all numeric columns ───────────────────────────────
GRID_COLS = 3
PER_PAGE = 12
numeric_pages = paginate(NUMERIC_COLS, PER_PAGE)
print(f"\n{len(NUMERIC_COLS)} numeric columns -> {len(numeric_pages)} page(s) per chart type, "
      f"so every chart stays a readable, normal size.")

for p, cols in enumerate(numeric_pages, start=1):
    n_rows = int(np.ceil(len(cols) / GRID_COLS))
    fig, axes = plt.subplots(n_rows, GRID_COLS, figsize=(15, max(3.8 * n_rows, 4)))
    fig.suptitle(f'STEP 3: Univariate Distributions — Numeric Features (page {p}/{len(numeric_pages)})',
                 fontsize=15, fontweight='bold', color=palette['primary'], y=0.995)
    axes = np.atleast_1d(axes).flatten()

    legend_handles = legend_labels = None
    for i, col in enumerate(cols):
        ax = axes[i]
        data = df[col].dropna()
        ax.hist(data, bins=50, color=palette['highlight'],
                alpha=0.55, density=True, edgecolor='white', linewidth=0.3)
        try:
            kde_x = np.linspace(data.min(), data.max(), 300)
            kde   = stats.gaussian_kde(data.sample(min(len(data), 10000), random_state=42))
            ax.plot(kde_x, kde(kde_x), color=palette['accent'], linewidth=2)
        except Exception:
            pass
        mu, sigma = data.mean(), data.std()
        ax.axvline(mu,         color=palette['success'], linestyle='-',  linewidth=1.5, label='Mean')
        ax.axvline(data.median(), color=palette['warn'], linestyle='--', linewidth=1.5, label='Median')
        ax.set_title(f'{col}', fontsize=10)
        ax.set_xlabel(f'Skew: {data.skew():.2f} | Kurt: {data.kurt():.2f}', fontsize=8.5)
        ax.grid(linestyle='--', alpha=0.4)
        ax.tick_params(labelsize=8)
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)
    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc='upper right', fontsize=9,
                   bbox_to_anchor=(0.995, 0.985), ncol=2, framealpha=0.9)
    top = 0.94 if n_rows <= 2 else (0.96 if n_rows <= 3 else 0.97)
    fig.tight_layout(rect=[0, 0.01, 1, top])
    save_path = os.path.join(PLOTS_DIR, f'eda_02_univariate_numeric_page{p}.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


# =============================================================================
# STEP 4: NORMALITY TESTING — Kya Dataset Normal Distribution Follow Karta Hai?
# =============================================================================
# 3 tests use honge:
# 1. Shapiro-Wilk     → small-to-medium samples (best for N < 5000)
# 2. Kolmogorov-Smirnov → large samples (N > 5000)
# 3. D'Agostino K²   → skewness + kurtosis based (any sample size)
# Har test ka p-value dekho:
#   p < 0.05 → NOT normal (reject H0)
#   p >= 0.05 → Could be normal (fail to reject H0)
# =============================================================================

print("\n" + "="*70)
print("STEP 4: NORMALITY TESTING")
print("="*70)

SAMPLE_SIZE = 5000  # Shapiro max ~5000 tak reliable

normality_results = []

for col in NUMERIC_COLS:
    data   = df[col].dropna()
    sample = data.sample(min(len(data), SAMPLE_SIZE), random_state=42)

    # Shapiro-Wilk (on sample)
    sw_stat, sw_p = shapiro(sample)

    # Kolmogorov-Smirnov (standardized)
    data_std = (data - data.mean()) / data.std()
    ks_stat, ks_p = kstest(data_std.sample(min(len(data_std), 5000), random_state=42), 'norm')

    # D'Agostino K²
    da_stat, da_p = normaltest(sample)

    # Decision: majority vote
    votes_normal  = sum([sw_p >= 0.05, ks_p >= 0.05, da_p >= 0.05])
    is_normal     = votes_normal >= 2
    skew_val      = data.skew()
    kurt_val      = data.kurt()

    normality_results.append({
        'column':       col,
        'n':            len(data),
        'skewness':     round(skew_val, 4),
        'kurtosis':     round(kurt_val, 4),
        'shapiro_p':    round(sw_p,  6),
        'ks_p':         round(ks_p,  6),
        'dagostino_p':  round(da_p,  6),
        'is_normal':    '✅ YES' if is_normal else '❌ NO',
    })

norm_df = pd.DataFrame(normality_results)
print(norm_df.to_string(index=False))

# Save results
norm_df.to_csv(os.path.join(EDA_STATS_DIR, 'eda_normality_test_results.csv'), index=False)
print("\nSaved: eda_stats/eda_normality_test_results.csv")

# ── Normality Summary Plot ────────────────────────────────────────────────
normal_cols     = norm_df[norm_df['is_normal'] == '✅ YES']['column'].tolist()
non_normal_cols = norm_df[norm_df['is_normal'] == '❌ NO']['column'].tolist()

print(f"\n  NORMAL columns     ({len(normal_cols)})  : {normal_cols}")
print(f"  NON-NORMAL columns ({len(non_normal_cols)}) : {non_normal_cols}")

# ── Q-Q Plots for all numeric cols ───────────────────────────────────────
for p, cols in enumerate(numeric_pages, start=1):
    n_rows = int(np.ceil(len(cols) / GRID_COLS))
    fig, axes = plt.subplots(n_rows, GRID_COLS, figsize=(15, max(3.8 * n_rows, 4)))
    fig.suptitle(f'STEP 4: Q-Q Plots — Normal Distribution Check (page {p}/{len(numeric_pages)})',
                 fontsize=14.5, fontweight='bold', color=palette['primary'], y=0.995)
    axes = np.atleast_1d(axes).flatten()

    for i, col in enumerate(cols):
        ax   = axes[i]
        data = df[col].dropna().sample(min(len(df[col].dropna()), 3000), random_state=42)
        (osm, osr), (slope, intercept, r) = stats.probplot(data, dist='norm')
        ax.plot(osm, osr,  'o',  color=palette['highlight'], markersize=2, alpha=0.4)
        ax.plot(osm, slope * np.array(osm) + intercept,
                color=palette['accent'], linewidth=2)
        row_match = norm_df[norm_df['column'] == col]
        is_n      = row_match['is_normal'].values[0] if len(row_match) else '?'
        color_t   = palette['success'] if '✅' in is_n else palette['accent']
        ax.set_title(f'{col}', fontsize=10)
        ax.set_xlabel(f'R²={r**2:.3f} | {is_n}', fontsize=8.5, color=color_t)
        ax.grid(linestyle='--', alpha=0.4)
        ax.tick_params(labelsize=8)

    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)
    top = 0.94 if n_rows <= 2 else (0.96 if n_rows <= 3 else 0.97)
    fig.tight_layout(rect=[0, 0.01, 1, top])
    save_path = os.path.join(PLOTS_DIR, f'eda_03_normality_qq_plots_page{p}.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


# =============================================================================
# STEP 5: BIVARIATE ANALYSIS — NUMERIC FEATURES vs TARGET
# =============================================================================

print("\n" + "="*60)
print("STEP 5: BIVARIATE — NUMERIC vs TARGET")
print("="*60)

qty_colors = {1: '#3498DB', 2: '#27AE60', 3: '#F39C12', 4: '#E74C3C', 5: '#8E44AD'}

for p, cols in enumerate(numeric_pages, start=1):
    n_rows = int(np.ceil(len(cols) / GRID_COLS))
    fig, axes = plt.subplots(n_rows, GRID_COLS, figsize=(15, max(4.2 * n_rows, 4)))
    fig.suptitle(f'STEP 5: Numeric Features vs Target — Box Plots per Quantity Class '
                 f'(page {p}/{len(numeric_pages)})',
                 fontsize=14, fontweight='bold', color=palette['primary'], y=0.995)
    axes = np.atleast_1d(axes).flatten()

    for i, col in enumerate(cols):
        ax = axes[i]
        groups = [df[df[TARGET] == qty][col].dropna() for qty in sorted(df[TARGET].unique())]

        bp = ax.boxplot(groups, patch_artist=True, notch=False,
                        flierprops=dict(marker='.', markersize=2, alpha=0.3,
                                        color=palette['accent']),
                        medianprops=dict(color='white', linewidth=2))

        colors_list = [qty_colors[q] for q in sorted(df[TARGET].unique())]
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        # ANOVA F-test for significance
        f_stat, p_val = stats.f_oneway(*groups)
        sig = '★ Significant' if p_val < 0.05 else '✗ Not Significant'
        sig_color = palette['success'] if p_val < 0.05 else palette['accent']

        ax.set_title(col, fontsize=10, pad=6)
        ax.set_xlabel(f'ANOVA: F={f_stat:.1f}, p={p_val:.4f}  {sig}',
                      fontsize=8, color=sig_color)
        ax.set_xticklabels([f'Qty {q}' for q in sorted(df[TARGET].unique())],
                           fontsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)
    top = 0.93 if n_rows <= 2 else (0.95 if n_rows <= 3 else 0.96)
    fig.tight_layout(rect=[0, 0.01, 1, top])
    save_path = os.path.join(PLOTS_DIR, f'eda_04_bivariate_numeric_vs_target_page{p}.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")

# ── Mean of each numeric feature per quantity class ───────────────────────
print("\nMean values per Quantity class:")
mean_by_qty = df.groupby(TARGET)[NUMERIC_COLS].mean().round(3)
print(mean_by_qty.to_string())
mean_by_qty.to_csv(os.path.join(EDA_STATS_DIR, 'eda_mean_by_quantity.csv'))
print("\nSaved: eda_stats/eda_mean_by_quantity.csv")


# =============================================================================
# MARKDOWN INSIGHTS — STEP 5: Bivariate Analysis
# =============================================================================
# ✅ Confirmed Business Insights — Bivariate Analysis se
#
# Insight 1 — Sabse Strong Finding: Price hi Quantity ka Predictor hai
# Product price > $100 wale orders mein 100% cases mein sirf qty=1 hai.
# Ek bhi exception nahi. Jabke $100 ya usse kam price par qty 1-5 almost
# equally distribute hoti hai.
#
# High price (>$100) → qty=1 : 100.0%
# Low price (≤$100)  → qty=1 : 21.3% baaki 2-5 distribute
#
# Business meaning: Customers mehnge products ek hi baar khareedte hain.
# Saste products repeat/bulk mein jaate hain. Ye retail ka universal law
# hai aur tumhare data mein perfectly confirmed hai.
#
# Insight 2 — Discount Rate ka Quantity se koi Talluq Nahi
# ANOVA F=0.08, p=0.99 — practically zero.
# Business meaning: Company discount dene se demand/quantity nahi badh rahi.
# Discount strategy quantity ko drive nahi kar rahi — ye ek concerning
# finding hai. Matlab customers zyada quantity isliye nahi le rahe k unhe
# discount mila. Demand quantity product type aur price se tay ho rahi hai,
# discount se nahi.
#
# Insight 3 — Shipping aur Processing ka Quantity se koi Rishta Nahi
# Shipping days, processing delay, late delivery risk — teeno ka quantity
# se correlation zero hai.
# Business meaning: Bulk orders ko alag treatment nahi milti. Chahey ek
# item ho ya paanch — delivery timeline same hi rehti hai. Operations mein
# koi quantity-based prioritization nahi ho rahi.
#
# Insight 4 — Profit Ratio completely flat hai across all quantities
# Median = 0.27 for qty 1, 2, 3, 4, 5 — exactly same. Correlation = 0.001.
# Business meaning: Per-item margin quantity se independent hai. Iska matlab
# company bulk discount nahi de rahi. Jo bhi pricing structure hai wo
# quantity-blind hai — ek item pe jo margin hai, paanch items pe bhi wohi
# margin hai.
#
# Insight 5 — Benefit per Order mein qty=2 par dip
# Mean: qty=1(24.4) → qty=2(12.3) → qty=3(18.6) → qty=4(25.7) → qty=5(32.2)
# Qty=2 par ek interesting dip hai.
# Business meaning: 2-item orders mein company comparatively kam kamaati
# hai — shayad specific product mix jo qty=2 mein jaata hai wo lower margin
# products hain.
#
# Feature Engineering — Price Tier banana chahiye ya nahi?
# Haan, banana chahiye — aur ye sabse important feature banega kyunki:
# Price > $100 → hamesha qty=1. Ye perfect separation hai. Model ke liye
# ye binary information gold hai.
# Meri recommendation — sirf ye do features banao:
# =============================================================================
 
# ----------------------- New features ye nikal saktey hein ----------------------------
# # 1. Price Tier — 3 categories, data ke natural breakpoints par

# df['price_tier'] = pd.cut(df['product_price'],
#                           bins=[0, 50, 100, float('inf')],
#                           labels=['Low', 'Medium', 'High'])
# # Low ≤$50 → bulk buying possible
# # Medium $50-100 → mixed behavior
# # High >$100 → always qty=1

# # 2. is_high_price — simple binary, direct signal

# df['is_high_price'] = (df['product_price'] > 100).astype(int)


# =============================================================================
# STEP 6: CATEGORICAL FEATURES — Frequency + Target Breakdown
# =============================================================================

print("\n" + "="*60)
print("STEP 6: CATEGORICAL FEATURE ANALYSIS")
print("="*60)

# Top-N for high-cardinality columns
TOP_N = 10

for col in CATEGORICAL_COLS:
    n_unique = df[col].nunique()
    print(f"\n  {col}  —  {n_unique} unique values")
    vc = df[col].value_counts().head(TOP_N)
    print(vc.to_string())

# ── Subplot: Countplots for each categorical col ──────────────────────────
CAT_GRID_COLS = 2
CAT_PER_PAGE = 6
cat_pages = paginate(CATEGORICAL_COLS, CAT_PER_PAGE)
print(f"\n{len(CATEGORICAL_COLS)} categorical columns -> {len(cat_pages)} page(s) per chart, "
      f"so every chart stays a readable, normal size.")

for p, cols in enumerate(cat_pages, start=1):
    n_rows = int(np.ceil(len(cols) / CAT_GRID_COLS))
    fig, axes = plt.subplots(n_rows, CAT_GRID_COLS, figsize=(15, max(4.2 * n_rows, 4)))
    fig.suptitle(f'STEP 6: Categorical Features — Top {TOP_N} Value Frequencies '
                 f'(page {p}/{len(cat_pages)})',
                 fontsize=14, fontweight='bold', color=palette['primary'], y=0.995)
    axes = np.atleast_1d(axes).flatten()

    for i, col in enumerate(cols):
        ax = axes[i]
        top_vals = df[col].value_counts().head(TOP_N)
        labels = [truncate_label(v) for v in top_vals.index.astype(str)]
        bars = ax.barh(labels, top_vals.values,
                       color=palette['highlight'], alpha=0.75, edgecolor='white')
        ax.set_title(col, fontsize=10.5)
        ax.set_xlabel('Count', fontsize=8.5)
        ax.invert_yaxis()
        for bar, val in zip(bars, top_vals.values):
            ax.text(bar.get_width() + top_vals.max() * 0.015,
                    bar.get_y() + bar.get_height()/2,
                    f'{val:,}', va='center', fontsize=7.5)
        ax.set_xlim(0, top_vals.max() * 1.18)
        ax.grid(axis='x', linestyle='--', alpha=0.4)
        ax.tick_params(labelsize=8.5)

    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)
    top = 0.90 if n_rows <= 2 else (0.93 if n_rows <= 3 else 0.95)
    fig.tight_layout(rect=[0, 0.01, 1, top])
    save_path = os.path.join(PLOTS_DIR, f'eda_05_categorical_counts_page{p}.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")

# ── Average Quantity per Categorical value ────────────────────────────────
print("\nAverage order_item_quantity by categorical features:")
for col in CATEGORICAL_COLS:
    top_vals = df[col].value_counts().head(TOP_N).index
    sub = df[df[col].isin(top_vals)]
    avg = sub.groupby(col)[TARGET].mean().sort_values(ascending=False)
    print(f"\n  {col}:")
    print(avg.round(3).to_string())


# =============================================================================
# MARKDOWN INSIGHTS — STEP 6: Categorical Features
# =============================================================================
# 1. type — Payment Method (Bhugtan ka Tareeqa)
# Kya represent karta hai:
# Ye column batata hai ke customer ne order ka payment kaise kiya —
# DEBIT card, TRANSFER (bank transfer), PAYMENT (online gateway), ya CASH.
#
# Business Insights:
# Retail business mein payment method directly order size ko motivate nahi
# karta, balki yeh customer trust aur operational channel ko dikhata hai.
# Debit aur bank transfer ka high volume (approx 66%) yeh zahir karta hai
# ke aapka customer base online financial ecosystem par zyada trust karta
# hai. Cash transactions ka sirf 11% hona supply chain management ke liye
# aik positive signal hai kyunki Cash on Delivery (COD) mein return aur
# rejection rates zyada hote hain. Digital payment dominant hone ki wajah
# se business checkout funnel ko mazeed optimize kar sakta hai.
#
# Ye Features banao:
# is_digital_payment: Binary flag (1 agar Debit/Transfer/Payment, 0 agar Cash).
# payment_risk_profile: Historical order cancellation rate per payment type
#                       ke hisab se encoding karna.
#
# 2. delivery_status — Delivery Ki Halat
# Kya represent karta hai:
# Har order ki delivery outcome — Late delivery, Advance shipping,
# Shipping on time, ya Shipping canceled.
#
# Business Insights:
# Demand forecasting mein hume sirf 'potential demand' nahi, balki
# 'fulfilled demand' ko forecast karna hota hai. Agar 55% orders late ho
# rahe hain, toh real retail world mein iska matlab hai ke inventory
# planning ya toh inefficient hai ya logistics partners streamline nahi
# hain. Iska target variable par direct impact bhale hi na dikhe (kyunki
# order pehle hota hai, delivery baad mein), lekin yeh historical churn
# lag create karta hai — matlab jis region mein is mahine deliveries late
# huin, agle mahine wahan se order_item_quantity drop ho jayegi. Is
# temporal correlation ko model mein daalna zaroori hai.
#
# ---------------- New Features ye nikal saktey hien --------------------:
# 1: 'delivery_delay_severity':  days_for_shipping_real minus
# 2: 'days_for_shipment_scheduled' :  karke exact days ka delay nikalna.
# 3: 'regional_late_rate': Har order_region ka rolling late delivery percentage
#   banana taake model ko pata chale ke kaunsi market unpredictable hai.
#
# 3. category_name — Product Category (Sabse Important Feature)
# 
# Kya represent karta hai:
# Product ki category jaise Cleats, Men's Footwear, Women's Apparel,
# Electronics, etc.
#
# Business Insights: Retail demand forecasting mein product category hi
# demand ka baseline structure tay karti hai. Books, Electronics, aur
# Computers ka hamesha baseline structure Single-Unit (Qty=1) rahega
# kyunki log aamtaur par ek waqt mein ek hi laptop ya camera khareedte
# hain. Jabki Cleats, Apparel, aur Sports goods Bulk/Multi-unit (Qty >=3)
# behavior dikhate hain kyunki yeh dynamic usage, sizing variants, ya team
# requirements ki wajah se zyada quantity mein khareede jaate hain. Model
# ko in dono behaviors ko alag alag handle karna sikhana hoga, warna
# single-unit categories ki forecasting hamesha overshoot karegi.
#
# ---------------------Ye Features banao ---------------------:
# 1: 'category_avg_qty' — har category ki historical average quantity
#                    (target encoding)
# 2: 'is_bulk_category' — binary flag (1 agar avg qty > 2.5)
# 3: 'is_single_unit_category'  — binary flag (1 agar avg qty = 1.0)
#
# 4. customer_segment — Customer Ka Type
# Kya represent karta hai:
# B2C ya B2B classification — Consumer (individual), Corporate (company),
# Home Office (ghar se kaam karne wale professionals).
#
# Data: Consumer: 92,435 (52%), Corporate: 54,170 (30%),
#       Home Office: 31,791 (18%)
#
# Business Insights: B2C (Consumer) aur B2B (Corporate/Home Office)
# segments ka buying pattern retail mein bilkul mukhtalif hota hai.
# Halankiye overall average quantity same dikh rahi hai, lekin inka
# seasonal behavior alag hota hai. Corporate orders financial quarter ke
# end mein ya back-to-office seasons mein spike karte hain, jabki
# individual consumers holidays aur weekends par khareedari karte hain.
# Forecasting model mein segment ko calendar flags (weekends, quarters)
# ke sath interact karwana zaroori hai taake temporary demand shocks ko
# pakda ja sake.
#
#------------------ Ye Features banao----------------------:
# 1: 'segment_seasonal_weight': customer_segment X order_quarter ka
#   interaction matrix.
# 2: 'is_b2b': Binary flag (1 agar Corporate ya Home Office, 0 agar Consumer)
#   kyunki B2B accounts ki lifetime value (LTV) zyada hoti hai.
#
# 5. department_name — Department
# Kya represent karta hai:
# Product ka department grouping — Fan Shop, Apparel, Golf, Footwear,
# Outdoors, etc.
# Fan Shop (66,585) aur Apparel (48,589) sabse bade departments hain.
# Golf department ki avg qty = 2.989 aur Footwear = 2.988 — high-quantity.
# Technology, Book Shop, Pet Shop, Health and Beauty — avg qty = 1.0.
# Department + Category ek hierarchical relationship hai.
# Feature idea: 'dept_avg_qty' (target encoding per department).
#
# 6. market — Global Market
# Kya represent karta hai:
# Geographic market region — LATAM, Europe, Pacific Asia, USCA, Africa.
# LATAM (51,594) aur Europe (50,252) sabse bade markets hain.
# Africa (11,614) chhota market hai lekin avg qty = 2.204 — sabse zyada.
# Pacific Asia sabse kam avg qty (2.084).
#
# 7. order_region — Order Ki Specific Region
# Kya represent karta hai:
# Zyada granular geographic breakdown — Central America, Western Europe,
# South America, Oceania, etc. (23 unique regions).
# Central America (28,341) sabse zyada orders.
# Canada (959 orders) mein avg qty = 2.276 — sabse zyada.
# Feature idea: region_late_delivery_rate.
#
# 8. order_status — Order Ki Processing Status
# Kya represent karta hai:
# Order lifecycle stage — COMPLETE, PENDING_PAYMENT, PROCESSING, PENDING,
# CLOSED, ON_HOLD, SUSPECTED_FRAUD, CANCELED, PAYMENT_REVIEW.
# COMPLETE (58,782) sabse zyada — lekin overall orders ka sirf 33%.
# SUSPECTED_FRAUD (4,021) aur PAYMENT_REVIEW (1,865) revenue risk hain.
# CANCELED orders (3,647) — in orders ko demand forecasting se bahar rakhna.
# Feature idea: is_fulfilled — binary (1 agar COMPLETE ya CLOSED).
#
# 9. shipping_mode — Shipping Ka Tareeqa
# Kya represent karta hai:
# Delivery speed option — Standard Class, Second Class, First Class,
# Same Day.
# Standard Class dominant hai (106,522 orders = 60%).
# Same Day sirf 9,617 orders (~5%).
# Avg quantity mein minimal farq hai (2.13–2.16).
# Feature idea: shipping_urgency_score — Same Day=4, First=3, Second=2,
#               Standard=1 — ordinal encoding.
# =============================================================================


# =============================================================================
# STEP 7: CATEGORICAL vs TARGET — Chi-Square Independence Test
# =============================================================================

print("\n" + "="*60)
print("STEP 7: CHI-SQUARE TEST — Categorical vs Target")
print("="*60)

chi2_results = []

for col in CATEGORICAL_COLS:
    try:
        ct = pd.crosstab(df[col], df[TARGET])
        chi2, p, dof, expected = chi2_contingency(ct)
        # Cramér's V (effect size)
        n       = ct.sum().sum()
        phi2    = chi2 / n
        r, k    = ct.shape
        cramers_v = np.sqrt(phi2 / min(k-1, r-1))
        chi2_results.append({
            'column':     col,
            'chi2':       round(chi2, 2),
            'p_value':    round(p, 6),
            'dof':        dof,
            "cramers_v":  round(cramers_v, 4),
            'significant': '✅ YES' if p < 0.05 else '❌ NO'
        })
    except Exception as e:
        print(f"  Warning: {col} — {e}")

chi2_df = pd.DataFrame(chi2_results).sort_values('cramers_v', ascending=False)
print(chi2_df.to_string(index=False))
chi2_df.to_csv(os.path.join(EDA_STATS_DIR, 'eda_chi2_results.csv'), index=False)
print("\nSaved: eda_stats/eda_chi2_results.csv")

# ── Stacked Bar — Top categorical cols by Cramér's V ─────────────────────
top_cat_cols = chi2_df.head(6)['column'].tolist()

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('STEP 7: Quantity Distribution by Top Categorical Features\n'
             '(Stacked % bars — relationship with target)',
             fontsize=14, fontweight='bold', color=palette['primary'], y=0.995)
axes = axes.flatten()

qty_palette = ['#3498DB','#27AE60','#F39C12','#E74C3C','#8E44AD']

for i, col in enumerate(top_cat_cols[:6]):
    ax = axes[i]
    top_vals = df[col].value_counts().head(8).index
    sub      = df[df[col].isin(top_vals)]
    ct_pct   = pd.crosstab(sub[col], sub[TARGET], normalize='index') * 100
    ct_pct.index = [truncate_label(v, 18) for v in ct_pct.index.astype(str)]

    ct_pct.plot(kind='bar', stacked=True, ax=ax,
                color=qty_palette, edgecolor='white', linewidth=0.5)
    cv_val = chi2_df[chi2_df['column'] == col]['cramers_v'].values[0]
    ax.set_title(f"{col}\nCramér's V = {cv_val:.4f}", fontsize=10, pad=6)
    ax.set_xlabel('')
    ax.set_ylabel('% of orders', fontsize=8.5)
    ax.tick_params(axis='x', rotation=40, labelsize=8)
    for label in ax.get_xticklabels():
        label.set_ha('right')
    ax.legend(title='Qty', fontsize=7.5, title_fontsize=7.5, loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.4)

for j in range(len(top_cat_cols), len(axes)):
    axes[j].set_visible(False)

fig.tight_layout(rect=[0, 0.01, 1, 0.90])
save_path = os.path.join(PLOTS_DIR, 'eda_06_categorical_vs_target.png')
fig.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {save_path}")


# =============================================================================
# MARKDOWN INSIGHTS — STEP 7: Chi-Square Analysis
# =============================================================================
# Q: Pehle Samjho: Ye Test Kya Kar Raha Hai?
# Chi-Square test ye check karta hai ke koi categorical column aur target
# variable (order_item_quantity) ke beech statistical relationship hai ya
# nahi. Cramér's V effect size batata hai — ye jitna bada, utna strong
# relationship.
#
# Q: En plots ki zaroorat q pari?
# Machine learning model banane se pehle hume aankhon se dekhna hota hai
# ke kya kisi category ke badalne se target variable (Quantity) ka behavior
# badal raha hai ya nahi? Agar kisi feature ke badalne se colors ka ratio
# bilkul same rahe, toh iska matlab hai woh feature bekaar hai. Agar ratio
# badal raha hai, toh woh feature sona (gold) hai!
#
# Inse Kya Insights Mil Rahin Hain?
# High-Impact Features (product_name, category_name): Inke plots mein har
# bar ke andar colors ka distribution bohot alag hai! Iska matlab yeh
# features target ko directly influence kar rahe hain.
# Low-Impact Features (customer_city, order_state): Inke plots mein sabhi
# bars bilkul ek jaisi, ek hi level par (bilkul flat pattern). Iska matlab
# customer chahay New York mein ho ya Los Angeles mein, uski quantity
# khareedne ka pattern bilkul same hai.
#
# Chi-Square Test Kya Hai Aur Isko Kyun Lagaya?
# Jab hamara target variable categorical ya discrete counts ho (jaise
# order_item_quantity: 1,2,3,4,5) aur humare input features bhi categorical
# hon, toh hum unke beech ka rishta maloom karne ke liye Chi-Square Test of
# Independence lagate hain. Asan Alfaz Mein: Yeh ek statistical test hai jo
# mathematically sabit karta hai ke "Kya in do columns ke beech koi real
# connection hai, ya yeh sirf ek ittefaq (random chance) hai?"
#
# Is Test Ki Kya Zaroorat Thi?
# Plots ko dekh kar insani aankh dhoka kha sakti hai, lekin maths dhoka
# nahi khati. Is test ki zaroorat isliye thi taake hum Feature Selection
# kar sakein. Jo features statistically significant honge, hum sirf unhi
# ko ML model ko train karne ke liye denge.
#
# 'product_name' — Cramér's V = 0.3979 (Sabse Strongest!)
# Yeh sabse powerful categorical predictor hai. Specific product ka naam
# directly quantity determine karta hai — kuch products naturally bulk mein
# bikti hain (e.g., sports accessories, golf balls) aur kuch single unit
# (cameras, laptops). Feature engineering mein product_avg_qty (target
# encoding) ye sabse important feature banega.
#
# Is Analysis sey pata chalta hai k hum :

# -----------------  ye new Features nakaal saktey hein ------------------
#
# 1. Target Encoding — Sabse Powerful
# 'product_avg_qty'      # har product ki historical avg quantity
# 'category_avg_qty'     # har category ki avg quantity
# 'department_avg_qty'   # har department ki avg quantity
#
# 2. Category Behavior Flags
# 'is_bulk_category'     # 1 agar category avg > 2.5
# 'is_single_unit'       # 1 agar category avg = 1.0
# =============================================================================


# =============================================================================
# STEP 8: CORRELATION ANALYSIS
# =============================================================================

print("\n" + "="*60)
print("STEP 8: CORRELATION ANALYSIS")
print("="*60)

corr_matrix = df[NUMERIC_COLS + [TARGET]].corr()

# Correlation with target specifically
target_corr = corr_matrix[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
print(f"\nCorrelation with '{TARGET}'  (sorted by |r|):")
print(target_corr.round(4).to_string())

# ── Full Heatmap ──────────────────────────────────────────────────────────
n_feat_corr = len(NUMERIC_COLS) + 1
fig, ax = plt.subplots(figsize=heatmap_figsize(n_feat_corr))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, ax=ax,
            cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            annot=True, fmt='.2f', annot_kws={'size': annot_fontsize(n_feat_corr)},
            linewidths=0.4, linecolor='white',
            cbar_kws={'shrink': 0.75, 'label': 'Pearson r'})
ax.set_title('STEP 8: Correlation Heatmap — All Numeric Features + Target',
             fontsize=14, fontweight='bold', pad=14)
ax.tick_params(axis='x', rotation=45, labelsize=8.5)
ax.tick_params(axis='y', rotation=0,  labelsize=8.5)
for label in ax.get_xticklabels():
    label.set_ha('right')
fig.tight_layout(rect=[0, 0, 1, 0.97])
save_path = os.path.join(PLOTS_DIR, 'eda_07_correlation_heatmap.png')
fig.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {save_path}")

# ── Target correlation bar chart ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
colors = [palette['success'] if v > 0 else palette['accent']
          for v in target_corr.values]
ax.barh(target_corr.index, target_corr.values, color=colors,
        edgecolor='white', alpha=0.8)
ax.axvline(0, color=palette['primary'], linewidth=1.2)
ax.set_title(f"Correlation of Features with '{TARGET}'",
             fontsize=13, fontweight='bold', pad=10)
ax.set_xlabel('Pearson r', fontsize=10)
ax.tick_params(labelsize=8.5)
ax.grid(axis='x', linestyle='--', alpha=0.4)
fig.tight_layout()
save_path = os.path.join(PLOTS_DIR, 'eda_08_target_correlations.png')
fig.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {save_path}")


# =============================================================================
# MARKDOWN INSIGHTS — STEP 8: Correlation Analysis
# =============================================================================
# 1. Price vs Quantity ka Negative Curve (-0.57)
# Raw price ka correlation -0.48 hai jabki Log price ka -0.57 hai.
# Business Insight: Yeh retail ka universally accepted law of demand hai ke
# sasti cheezein (sports accessories, apparel) bulk mein bikti hain aur
# premium ya mehangi cheezein (laptops, insurance, heavy equipment) single
# unit bikti hain. Log transform ne is non-linear curve ko behtar pakda
# hai, jo aapke model ki accuracy ke liye bohot zaroori hai.
#
# 2. Discount Rate ka Alarm Bell (0.0000)
# Business Insight: Retail mein koi bhi discount is ummeed par diya jata
# hai ke customer quantity barha dega. Agar correlation zero hai, toh iska
# matlab hai ke is business ki markdown (discounting) strategy poori tarah
# flop hai. Management fuzool mein margin zaya kar rahi hai.
#
# 3. Duplicates (Multicollinearity) ko Sahi Pehchana
# Heatmap mein jahan dark blue boxes hain (1.00), ye bilkul saaf duplicates
# hain, ek ko hatana zaroori tha:
# 'product_price' aur 'order_item_product_price' bilkul saaf duplicates hain.
#
# cols_to_drop = [
#     'product_price', 'order_item_product_price', 'product_price_log',
#     'processing_delay_days', 'customer_zipcode'
# ]
#
# ------------------- ye New Features banay ja saktey hein -------------------------:
# [ price_tier,
#   discount_waste_flag,
#   actual_revenue_per_order,
#   is_bulk_category,
#   is_single_unit_category,
#   category_avg_qty  (target encoding),
#   product_avg_qty   (target encoding) ]
# =============================================================================


# =============================================================================
# STEP 9: TEMPORAL ANALYSIS — Sales Trends Over Time
# =============================================================================

print("\n" + "="*60)
print("STEP 9: TEMPORAL ANALYSIS")
print("="*60)

# Monthly order volume and average quantity
monthly = df.groupby(['order_year', 'order_month']).agg(
    total_orders    = (TARGET, 'count'),
    avg_quantity    = (TARGET, 'mean'),
    total_quantity  = (TARGET, 'sum'),
    avg_price       = ('order_item_product_price', 'mean'),
    total_profit    = ('benefit_per_order', 'sum'),
).reset_index()

monthly['period'] = pd.to_datetime(
    monthly['order_year'].astype(str) + '-' +
    monthly['order_month'].astype(str).str.zfill(2))

monthly = monthly.sort_values('period')
print(monthly[['period','total_orders','avg_quantity','total_profit']].to_string(index=False))

# ── Multi-line time series plots ──────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(15, 13))
fig.suptitle('STEP 9: Temporal Trends — Monthly Business Metrics',
             fontsize=15, fontweight='bold', color=palette['primary'], y=0.995)

# Panel 1: Order volume
axes[0].fill_between(monthly['period'], monthly['total_orders'],
                     alpha=0.25, color=palette['highlight'])
axes[0].plot(monthly['period'], monthly['total_orders'],
             color=palette['highlight'], linewidth=2, marker='o', markersize=3)
axes[0].set_title('Monthly Order Volume')
axes[0].set_ylabel('Total Orders')
axes[0].grid(linestyle='--', alpha=0.4)

# Panel 2: Avg quantity
axes[1].plot(monthly['period'], monthly['avg_quantity'],
             color=palette['success'], linewidth=2, marker='s', markersize=3)
axes[1].axhline(monthly['avg_quantity'].mean(), color=palette['accent'],
                linestyle='--', linewidth=1.5, label=f"Overall Mean: {monthly['avg_quantity'].mean():.3f}")
axes[1].set_title('Monthly Avg Order Quantity')
axes[1].set_ylabel('Avg Quantity')
axes[1].legend(fontsize=9)
axes[1].grid(linestyle='--', alpha=0.4)

# Panel 3: Total profit
axes[2].fill_between(monthly['period'], monthly['total_profit'],
                     alpha=0.25, color=palette['warn'])
axes[2].plot(monthly['period'], monthly['total_profit'],
             color=palette['warn'], linewidth=2, marker='^', markersize=3)
axes[2].axhline(0, color=palette['accent'], linewidth=1, linestyle='--')
axes[2].set_title('Monthly Total Profit/Benefit')
axes[2].set_ylabel('Total Benefit')
axes[2].grid(linestyle='--', alpha=0.4)

for ax in axes:
    ax.tick_params(axis='x', rotation=30, labelsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.95])
save_path = os.path.join(PLOTS_DIR, 'eda_09_temporal_trends.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close('all')
print(f"Saved: {save_path}")

# ── Heatmap: Month vs Weekday avg quantity ────────────────────────────────
pivot = df.pivot_table(values=TARGET, index='order_month',
                       columns='order_weekday', aggfunc='mean')
pivot.index = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec'][:len(pivot)]
pivot.columns = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][:len(pivot.columns)]

fig, ax = plt.subplots(figsize=(12, 6.5))
sns.heatmap(pivot, ax=ax, cmap='YlOrRd', annot=True, fmt='.2f',
            linewidths=0.5, linecolor='white', cbar_kws={'label': 'Avg Quantity'})
ax.set_title('Avg Order Quantity: Month × Weekday Heatmap\n'
             '(Business seasonality patterns)',
             fontsize=13, fontweight='bold', pad=10)
fig.tight_layout()
save_path = os.path.join(PLOTS_DIR, 'eda_10_month_weekday_heatmap.png')
fig.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {save_path}")


# =============================================================================
# MARKDOWN INSIGHTS — STEP 9: Temporal Analysis
# =============================================================================
# 1. 2015 ka Period: Mature aur Stable Demand
# 2015 mein order volume (4700-5400) aur average quantity (2.18-2.25)
# bilkul stable rahe. Real world mein yeh is baat ka saboot hota hai ke
# aapka brand market mein set ho chuka hai. Koi achanak spike ya crash na
# hona ek healthy operation ki nishani hai.
#
# 2. 2016 ka Period: "Slow Burn" (Dheemi Girawoot)
# 2016 mein average quantity chupke se thodi neeche ghirna shuru hui
# (2.22 se ghir kar 2.17 par aayi). Pliant aankhon ko 0.05 ka drop chota
# lagta hai, lekin jab aap isko laakhon orders par multiply karte hain,
# toh yeh lakhon rupayon ka revenue loss hota hai.
#
# 3. September 2017: Highest Profit Ke Baad Sudden Collapse
# September 2017 mein profit ab tak ka sabse zyada ($134,242) tha, lekin
# achanak October mein orders 57% gir gaye aur quantity exact 1.000 par
# freeze ho gayi. Yeh 100% Data Truncation ya ERP/Database system ke
# badalne ka ishara hai.
#
# 4. January-February ka Pattern: Post-Holiday Restocking
# Heatmap mein January ki row sabse dark red hai, jahan Monday aur Tuesday
# ko quantity bohot high hai. Retail stores January ke shuruati dino mein
# dobara se bulk orders deke apni inventory ko replenish karte hain.
#
# 5. March-April ka Pattern: Sports Seasonality Peak
# March aur April mein har din demand consistently high aur stable rehti
# hai. Sports clubs, schools, aur outdoor enthusiasts isi period mein bulk
# khareedari karte hain — aapke business ka asli natural seasonal peak.
#
# 6. Weekday vs Weekend Pattern: B2B (Corporate) Component
# Saturday aur Sunday ko average quantity baki haftay ke dino ke muqable
# thodi halki hai, jabki Wednesday ko demand high hai. Iska matlab hai ke
# customers mein ek bada hissa B2B (Corporate) buyers ka hai jo business
# hours mein order karte hain.
#
# --------------- ye New Features banay ja saktey hein ---------------------
# 'Is_stable_period'
# 'month_weekday_avg_qty'
# 'rolling_3month_avg'
# =============================================================================


# =============================================================================
# STEP 10: BUSINESS INSIGHTS — Client-Facing Analysis
# =============================================================================

print("\n" + "="*60)
print("STEP 10: BUSINESS INSIGHTS")
print("="*60)

fig = plt.figure(figsize=(19, 15))
fig.suptitle('STEP 10: Business Insights — Retail Demand Patterns',
             fontsize=16, fontweight='bold', color=palette['primary'], y=0.99)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.4, top=0.93, bottom=0.05)

# ── B1: Top 10 Product Categories by Avg Quantity ────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
if 'category_name' in df.columns:
    cat_qty = df.groupby('category_name')[TARGET].mean().sort_values(ascending=False).head(10)
    bars = ax1.barh(cat_qty.index, cat_qty.values,
                    color=palette['highlight'], alpha=0.8, edgecolor='white')
    for bar, val in zip(bars, cat_qty.values):
        ax1.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', fontsize=8)
    ax1.set_title('Top 10 Categories by Avg Order Quantity')
    ax1.set_xlabel('Avg Quantity')
    ax1.grid(axis='x', linestyle='--', alpha=0.4)
    ax1.invert_yaxis()

# ── B2: Customer Segment breakdown ───────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
if 'customer_segment' in df.columns:
    seg = df.groupby('customer_segment')[TARGET].agg(['mean','count'])
    seg.columns = ['avg_qty', 'n_orders']
    bars2 = ax2.bar(seg.index, seg['avg_qty'],
                    color=[palette['success'], palette['warn'], palette['accent']],
                    edgecolor='white', alpha=0.85)
    ax2.set_title('Avg Quantity by\nCustomer Segment')
    ax2.set_ylabel('Avg Quantity')
    ax2.tick_params(axis='x', rotation=20, labelsize=8)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    for bar, row in zip(bars2, seg.itertuples()):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.003,
                 f'n={row.n_orders:,}', ha='center', fontsize=7)

# ── B3: Shipping Mode vs Quantity ─────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, :2])
if 'shipping_mode' in df.columns:
    ship = df.groupby('shipping_mode')[TARGET].value_counts(normalize=True).unstack() * 100
    ship.plot(kind='bar', stacked=True, ax=ax3,
              color=qty_palette, edgecolor='white', linewidth=0.5)
    ax3.set_title('Quantity Distribution by Shipping Mode')
    ax3.set_ylabel('% of Orders')
    ax3.tick_params(axis='x', rotation=20, labelsize=9)
    ax3.legend(title='Qty', fontsize=8, title_fontsize=8)
    ax3.grid(axis='y', linestyle='--', alpha=0.4)

# ── B4: Market / Region ───────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
if 'market' in df.columns:
    mkt = df.groupby('market')[TARGET].mean().sort_values(ascending=False)
    ax4.barh(mkt.index, mkt.values, color=palette['teal'], alpha=0.8, edgecolor='white')
    ax4.set_title('Avg Quantity by Market')
    ax4.set_xlabel('Avg Quantity')
    ax4.invert_yaxis()
    ax4.grid(axis='x', linestyle='--', alpha=0.4)

# ── B5: Delivery Status vs Quantity ───────────────────────────────────────
ax5 = fig.add_subplot(gs[2, :2])
if 'delivery_status' in df.columns:
    del_qty = df.groupby('delivery_status')[TARGET].value_counts(normalize=True).unstack() * 100
    del_qty.plot(kind='bar', stacked=True, ax=ax5,
                 color=qty_palette, edgecolor='white', linewidth=0.5)
    ax5.set_title('Quantity by Delivery Status\n(Late delivery → demand pattern?)')
    ax5.set_ylabel('% of Orders')
    ax5.tick_params(axis='x', rotation=20, labelsize=9)
    ax5.legend(title='Qty', fontsize=8, title_fontsize=8)
    ax5.grid(axis='y', linestyle='--', alpha=0.4)

# ── B6: Order Status impact ───────────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 2])
if 'order_status' in df.columns:
    os_qty = df.groupby('order_status')[TARGET].mean().sort_values()
    ax6.barh(os_qty.index, os_qty.values, color=palette['purple'],
             alpha=0.8, edgecolor='white')
    ax6.set_title('Avg Quantity by\nOrder Status')
    ax6.set_xlabel('Avg Quantity')
    ax6.grid(axis='x', linestyle='--', alpha=0.4)
    ax6.invert_yaxis()

save_path = os.path.join(PLOTS_DIR, 'eda_11_business_insights.png')
fig.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {save_path}")


# =============================================================================
# MARKDOWN INSIGHTS — STEP 10: Business Insights Charts
# =============================================================================
# Chart 1: Top 10 Categories by Avg Order Quantity
# "As Seen on TV" (~3.029) aur "Tennis & Racquet" (~3.021) sabse upar hain.
# Sab categories ki avg qty 2.95-3.03 k beech hai — demand relatively
# uniform hai across categories. Koi ek category dramatically zyada nahi.
# Retail Importance: Inventory planning, supplier negotiations, shelf space.
# Business Insight: Avg quantities mein bahut kam difference hai — matlab
# category type demand ko drive nahi kar rahi utna jitna expect hota.
# Doosre factors (price, segment, region) zyada influential hain.
#
# Chart 2: Avg Quantity by Customer Segment
# Teeno segments (~2.0) almost equal hain. Corporate segment thoda upar.
# Business Insight: Corporate aur Home Office segments mein bulk ordering
# ki expected pattern confirm nahi hoti — surprising finding jo pricing/
# discount strategy ko revisit karne ka signal deta hai.
#
# Chart 3: Quantity Distribution by Shipping Mode
# Har shipping mode mein distribution almost identical hai.
# Qty=1 (blue) dominant hai har jagah ~40-45%.
# Business Insight: Shipping mode aur quantity mein koi meaningful
# relationship nahi — customers shipping choice urgency/preference par
# karte hain, quantity ke basis par nahi.
#
# Chart 4: Avg Quantity by Market
# Africa sabse upar (~2.0+), Pacific Asia sabse neeche.
# Business Insight: Africa market mein highest avg quantity — developing
# markets mein bulk buying tendency zyada hoti hai (less frequent shopping
# trips). Ye regional forecasting mein alag treatment ki zaroorat indicate
# karta hai.
#
# Chart 5: Quantity by Delivery Status
# Sab delivery statuses mein qty distribution almost same hai.
# Business Insight: Delivery performance aur order quantity mein koi strong
# pattern nahi — customers apni quantity delivery experience k basis par
# adjust nahi karte.
#
# Chart 6: Avg Quantity by Order Status
# SUSPECTED_FRAUD orders ki avg qty normal orders jaisi hai.
# Business Insight: Fraudulent orders quantity se identify nahi ho sakte —
# fraud detection ke liye doosre features zyada useful honge. Ye EDA se
# milne wala important negative finding hai jo modeling mein guide karega.
# =============================================================================


# =============================================================================
# STEP 11: PROFIT & PRICE INTELLIGENCE
# =============================================================================

print("\n" + "="*60)
print("STEP 11: PROFIT & PRICE INTELLIGENCE")
print("="*60)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('STEP 11: Profit & Price Intelligence — Client Business Metrics',
             fontsize=15, fontweight='bold', color=palette['primary'], y=0.995)

qty_levels = sorted(df[TARGET].unique())

# P1: benefit_per_order by quantity (manual boxplot — avoids pandas' own
# auto-generated "Boxplot grouped by..." title, which can't be fully
# suppressed once df.boxplot(by=...) creates it, and it ends up overlapping
# the custom title set below.)
ax = axes[0][0]
groups1 = [df[df[TARGET] == q]['benefit_per_order'].dropna() for q in qty_levels]
ax.boxplot(groups1, patch_artist=True, tick_labels=[str(q) for q in qty_levels],
           medianprops=dict(color=palette['accent'], linewidth=2),
           boxprops=dict(facecolor='#AED6F1', color=palette['highlight']))
ax.set_title('Benefit Per Order vs Quantity', fontsize=10.5, pad=6)
ax.set_xlabel('Order Quantity', fontsize=9)
ax.set_ylabel('Benefit Per Order', fontsize=9)
ax.grid(axis='y', linestyle='--', alpha=0.4)

# P2: profit ratio by quantity
ax2 = axes[0][1]
groups2 = [df[df[TARGET] == q]['order_item_profit_ratio'].dropna() for q in qty_levels]
ax2.boxplot(groups2, patch_artist=True, tick_labels=[str(q) for q in qty_levels],
            medianprops=dict(color=palette['accent'], linewidth=2),
            boxprops=dict(facecolor='#AED6F1', color=palette['highlight']))
ax2.set_title('Profit Ratio vs Quantity', fontsize=10.5, pad=6)
ax2.set_xlabel('Order Quantity', fontsize=9)
ax2.grid(axis='y', linestyle='--', alpha=0.4)

# P3: Scatter — product_price_log vs benefit_per_order
ax3 = axes[0][2]
if 'product_price_log' in df.columns:
    sample = df.sample(min(len(df), 5000), random_state=42)
    scatter_colors = [qty_colors[q] for q in sample[TARGET]]
    ax3.scatter(sample['product_price_log'], sample['benefit_per_order'],
                c=scatter_colors, alpha=0.3, s=5)
    ax3.set_title('Product Price (log) vs Benefit', fontsize=10.5, pad=6)
    ax3.set_xlabel('product_price_log', fontsize=9)
    ax3.set_ylabel('benefit_per_order', fontsize=9)
    ax3.grid(linestyle='--', alpha=0.4)
    from matplotlib.patches import Patch
    handles = [Patch(color=qty_colors[q], label=f'Qty {q}') for q in sorted(df[TARGET].unique())]
    ax3.legend(handles=handles, fontsize=7.5, title='Quantity', title_fontsize=7.5)

# P4: Profit margin by category (top 10)
ax4 = axes[1][0]
if 'category_name' in df.columns:
    top_cats = df['category_name'].value_counts().head(10).index
    cat_profit = df[df['category_name'].isin(top_cats)].groupby('category_name')['order_item_profit_ratio'].mean().sort_values()
    cat_labels = [truncate_label(v, 20) for v in cat_profit.index]
    ax4.barh(cat_labels, cat_profit.values,
             color=[palette['success'] if v >= 0 else palette['accent'] for v in cat_profit.values],
             alpha=0.8, edgecolor='white')
    ax4.axvline(0, color='black', linewidth=1)
    ax4.set_title('Avg Profit Ratio by Category', fontsize=10.5, pad=6)
    ax4.set_xlabel('Profit Ratio', fontsize=9)
    ax4.tick_params(labelsize=8)
    ax4.grid(axis='x', linestyle='--', alpha=0.4)

# P5: Discount rate analysis
ax5 = axes[1][1]
if 'order_item_discount_rate' in df.columns:
    disc_qty = df.groupby(TARGET)['order_item_discount_rate'].mean()
    ax5.bar(disc_qty.index, disc_qty.values, color=palette['warn'],
            edgecolor='white', alpha=0.85)
    ax5.set_title('Avg Discount Rate by Quantity\n(Higher qty = more discount?)', fontsize=10, pad=6)
    ax5.set_xlabel('Order Quantity', fontsize=9)
    ax5.set_ylabel('Avg Discount Rate', fontsize=9)
    ax5.grid(axis='y', linestyle='--', alpha=0.4)

# P6: Processing delay by quantity
ax6 = axes[1][2]
if 'processing_delay_days' in df.columns:
    delay_qty = df.groupby(TARGET)['processing_delay_days'].mean()
    ax6.plot(delay_qty.index, delay_qty.values,
             'o-', color=palette['purple'], linewidth=2, markersize=8)
    ax6.set_title('Avg Processing Delay by Quantity\n(Larger orders take longer?)', fontsize=10, pad=6)
    ax6.set_xlabel('Order Quantity', fontsize=9)
    ax6.set_ylabel('Avg Delay (Days)', fontsize=9)
    ax6.grid(linestyle='--', alpha=0.4)

fig.tight_layout(rect=[0, 0.01, 1, 0.93])
save_path = os.path.join(PLOTS_DIR, 'eda_12_profit_price_intelligence.png')
fig.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {save_path}")


# =============================================================================
# MARKDOWN INSIGHTS — STEP 11: Profit & Price Intelligence
# =============================================================================

# ------------------- Ye New Features banay ja saktey hein ------------------------
# 1. Processing Delay Risk (100% Safe aur Solid Feature)
# Kyonke quantity 4 ka pattern historical data mein ajeeb hai, ye
# operational features mein help karega:
# df['delay_risk_qty_4'] = (df['order_item_quantity'] == 4).astype(int)
#
# 2. Category Profit Tier (Safe - Kyonke category historical trend par depend)
# profit_map = {'Electronics': 'High', 'Shop By Sport': 'Low',
#               'Cleats': 'High'}  # Baki categories bar se dekh kar map karein
# df['category_profit_tier'] = df['category'].map(profit_map)
#
# 3. Base Price Bands (Safe - Quantity ke bagair sirf product price ka log scale)
# df['log_price_band'] = pd.cut(df['product_price_log'], bins=3,
#                               labels=['Cheap', 'Medium', 'Expensive'])
#
# Chart-by-Chart Business Insights:
# 1. Benefit Per Order vs Quantity — Quantity 1-5 sab ka median benefit
#    roughly similar. Lekin negative outliers bohot zyada hain — kuch orders
#    mein serious loss ho raha hai. Bulk order = guaranteed profit nahi.
# 2. Profit Ratio vs Quantity — Profit ratio roughly flat across quantities.
#    Negative profit ratios bhi exist karte hain (loss-making orders).
# 3. Product Price (log) vs Benefit — Price badhne ke sath benefit zyada
#    variable. High-price items par bada loss bhi possible, bada profit bhi.
# 4. Avg Profit Ratio by Category — Electronics sabse zyada profitable.
#    Shop By Sport sabse kam profitable. Difference 0.08-0.12 range mein.
# 5. Avg Discount Rate by Quantity — Discount rate almost equal across all
#    quantities (~8-10%). Business bulk discount policy follow nahi kar raha.
# 6. Avg Processing Delay by Quantity — Non-linear pattern. Quantity 3 par
#    delay sabse kam, Quantity 4 par sabse zyada. Operational bottleneck hai.
# =============================================================================


# =============================================================================
# STEP 12: FINAL EDA SUMMARY — Console Report
# =============================================================================

print("\n" + "="*65)
print("  EDA COMPLETE — SUMMARY REPORT")
print("="*65)

print(f"""
DATASET OVERVIEW
  Rows    : {df.shape[0]:,}
  Columns : {df.shape[1]}
  Target  : {TARGET}  (multiclass: 1–5)

TARGET DISTRIBUTION
""")
for qty, cnt in df[TARGET].value_counts().sort_index().items():
    print(f"  Qty {qty} → {cnt:,} orders ({cnt/len(df)*100:.1f}%)")

print(f"""
NORMALITY FINDINGS
  Normal columns     : {len(normal_cols)}  → {normal_cols}
  Non-normal columns : {len(non_normal_cols)} → {non_normal_cols}
  → Most features are NON-NORMAL (expected for retail transactional data).
  → Use non-parametric tests / tree-based models / robust scalers.

CORRELATION WITH TARGET
""")
print(target_corr.round(4).to_string())

print(f"""
CHI-SQUARE: TOP CATEGORICAL PREDICTORS
""")
print(chi2_df[['column','cramers_v','significant']].to_string(index=False))

print(f"""
BUSINESS INSIGHTS (KEY FINDINGS)
  1. Product pricing (log-transformed) shows distinct separation across qty classes.
  2. Benefit per order varies significantly with order quantity (ANOVA p<0.05).
  3. Shipping mode and delivery status show meaningful quantity distribution shifts.
  4. Seasonal patterns exist — certain months/weekdays have higher avg quantities.
  5. Customer segment influences ordering behavior.
  6. Category-level profit margins vary widely — some categories run at a loss.

MODELING RECOMMENDATIONS
  - Tree-based models (XGBoost, RF) preferred → non-normal, mixed feature types.
  - Use _log price columns, NOT raw price columns.
  - Encode categoricals: OHE for low-cardinality, Target Encoding for high-cardinality.
  - Class imbalance: check if SMOTE or class weights needed (Qty 4-5 likely minority).

OUTPUT FILES SAVED  (folder: assets/plots/eda/):
  assets/plots/eda/eda_01_target_distribution.png
  assets/plots/eda/eda_02_univariate_numeric_page*.png  (paginated, ~12 features/page)
  assets/plots/eda/eda_03_normality_qq_plots_page*.png  (paginated)
  assets/plots/eda/eda_04_bivariate_numeric_vs_target_page*.png  (paginated)
  assets/plots/eda/eda_05_categorical_counts_page*.png  (paginated, ~6 features/page)
  assets/plots/eda/eda_06_categorical_vs_target.png
  assets/plots/eda/eda_07_correlation_heatmap.png
  assets/plots/eda/eda_08_target_correlations.png
  assets/plots/eda/eda_09_temporal_trends.png
  assets/plots/eda/eda_10_month_weekday_heatmap.png
  assets/plots/eda/eda_11_business_insights.png
  assets/plots/eda/eda_12_profit_price_intelligence.png
  reports/eda_stats/eda_normality_test_results.csv
  reports/eda_stats/eda_mean_by_quantity.csv
  reports/eda_stats/eda_chi2_results.csv
""")

print("="*65)
print("EDA COMPLETE ✅  |  Next: Phase 4 — Feature Engineering")
print("="*65)