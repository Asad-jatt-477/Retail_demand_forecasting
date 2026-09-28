# =============================================================================
#  RETAIL DEMAND FORECASTING — EDA ADD-ON
#  Re-Included Financial Features: order_item_total, order_item_discount,
#                                   sales_per_customer, sales
#  Ye features pehle data cleaning phase mein drop ho gayi thin,
#  ab inhe modeling ke liye wapas include kiya ja raha hai.
#  Is script ko existing 05_exploratory_data_analysis.py ke BAAD run karo
#  (ya usi file ke aakhir mein paste kar do Step 12 se pehle).
# =============================================================================

# NOTE: Ye script assume karta hai ke:
#   - df already load ho chuka hai (Step 1 se)
#   - palette, TARGET, PLOTS_DIR, qty_colors sab already defined hain
#   - Agar standalone run karna ho toh neeche wala setup block uncomment karo

# --------------------------------------------------------------------------
# STANDALONE SETUP (sirf tab uncomment karo agar alag file mein run kar rahe ho)
# --------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless-safe; no blocking plt.show() windows
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, kstest, normaltest
import warnings, os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(ROOT_DIR, 'data', 'processed')
warnings.filterwarnings('ignore')

palette = {
    'primary':   '#2C3E50', 'accent':    '#E74C3C',
    'highlight': '#3498DB', 'success':   '#27AE60',
    'warn':      '#F39C12', 'bg':        '#FAFAFA',
    'grid':      '#ECEFF1', 'purple':    '#8E44AD',
    'teal':      '#16A085',
}
plt.rcParams.update({
    'figure.facecolor': palette['bg'],  'axes.facecolor': palette['bg'],
    'axes.edgecolor':   palette['primary'], 'axes.labelcolor': palette['primary'],
    'xtick.color':      palette['primary'], 'ytick.color':     palette['primary'],
    'grid.color':       palette['grid'],    'font.family':     'DejaVu Sans',
    'axes.titleweight': 'bold',             'axes.titlesize':  11,
})
PLOTS_DIR = os.path.join(ROOT_DIR, 'assets', 'plots', 'eda')
os.makedirs(PLOTS_DIR, exist_ok=True)
print(f"📁 Output folder ready: '{PLOTS_DIR}/'")

df = pd.read_csv(os.path.join(DATA_DIR, 'Retail_demand_outliers_TREATED.csv'), encoding='latin-1')
TARGET = 'order_item_quantity'
qty_colors = {1: '#3498DB', 2: '#27AE60', 3: '#F39C12', 4: '#E74C3C', 5: '#8E44AD'}
# --------------------------------------------------------------------------


# =============================================================================
# STEP 13: RE-INCLUDED FINANCIAL FEATURES — EDA
# Features: order_item_total | order_item_discount | sales_per_customer | sales
# =============================================================================

FIN_FEATURES = ['order_item_total', 'order_item_discount', 'sales_per_customer', 'sales']

# Log-transformed versions jo dataset mein pehle se hain
LOG_FEATURES = {
    'order_item_total':    'order_item_total_log',
    'sales_per_customer':  'sales_per_customer_log',
    'sales':               'sales_log',
    'order_item_discount': None   # Discount ka log version nahi hai
}

print("\n" + "="*70)
print("STEP 13: RE-INCLUDED FINANCIAL FEATURES — EDA")
print("="*70)
print(f"\nFeatures under analysis: {FIN_FEATURES}")
print("\nBasic Statistics:")
print(df[FIN_FEATURES].describe().round(3).to_string())

print("\nSkewness & Kurtosis:")
for col in FIN_FEATURES:
    print(f"  {col:30s}  Skew: {df[col].skew():.3f}  |  Kurt: {df[col].kurt():.3f}")


# =============================================================================
# PLOT 13A: UNIVARIATE DISTRIBUTIONS — Histogram + KDE + Boxplot
# Har feature ki individual distribution dekhna zaroori hai taake:
# 1. Skewness samjho (kya log transform zaroori hai?)
# 2. Outlier magnitude dekho
# 3. Natural breakpoints identify karo (feature engineering ke liye)
# =============================================================================

fig, axes = plt.subplots(4, 3, figsize=(18, 20))
fig.suptitle(
    'STEP 13A: Univariate Distributions — Re-Included Financial Features\n'
    'Histogram + KDE | Boxplot | Log-Scale Distribution',
    fontsize=15, fontweight='bold', color=palette['primary'], y=0.995
)

for row_idx, col in enumerate(FIN_FEATURES):
    data = df[col].dropna()

    # ---- Panel 1: Histogram + KDE ----
    ax1 = axes[row_idx][0]
    ax1.hist(data, bins=60, color=palette['highlight'], alpha=0.50,
             density=True, edgecolor='white', linewidth=0.3)
    try:
        kde_x = np.linspace(data.quantile(0.01), data.quantile(0.99), 300)
        kde   = stats.gaussian_kde(data.sample(min(len(data), 10000), random_state=42))
        ax1.plot(kde_x, kde(kde_x), color=palette['accent'], linewidth=2, label='KDE')
    except Exception:
        pass
    ax1.axvline(data.mean(),   color=palette['success'], linewidth=1.8,
                linestyle='-',  label=f'Mean: {data.mean():.1f}')
    ax1.axvline(data.median(), color=palette['warn'],    linewidth=1.8,
                linestyle='--', label=f'Median: {data.median():.1f}')
    ax1.set_title(f'{col} — Distribution')
    ax1.set_xlabel(f'Skew: {data.skew():.2f}  |  Kurt: {data.kurt():.2f}', fontsize=8)
    ax1.legend(fontsize=8)
    ax1.grid(linestyle='--', alpha=0.4)
    ax1.tick_params(labelsize=8)

    # ---- Panel 2: Boxplot ----
    ax2 = axes[row_idx][1]
    bp = ax2.boxplot(data, patch_artist=True, vert=True,
                     flierprops=dict(marker='.', markersize=2, alpha=0.3,
                                     color=palette['accent']),
                     medianprops=dict(color='white', linewidth=2),
                     widths=0.5)
    bp['boxes'][0].set_facecolor(palette['highlight'])
    bp['boxes'][0].set_alpha(0.7)
    q1, q3 = data.quantile(0.25), data.quantile(0.75)
    iqr    = q3 - q1
    ax2.set_title(f'{col} — Boxplot')
    ax2.set_ylabel('Value')
    ax2.set_xticklabels([col], fontsize=8)
    ax2.set_xlabel(
        f'Q1={q1:.1f}  |  Median={data.median():.1f}  |  Q3={q3:.1f}  |  IQR={iqr:.1f}',
        fontsize=7.5
    )
    ax2.grid(axis='y', linestyle='--', alpha=0.4)

    # ---- Panel 3: Log-Scale or Log-Transformed version ----
    ax3 = axes[row_idx][2]
    log_col = LOG_FEATURES.get(col)
    if log_col and log_col in df.columns:
        log_data = df[log_col].dropna()
        ax3.hist(log_data, bins=60, color=palette['teal'], alpha=0.55,
                 density=True, edgecolor='white', linewidth=0.3)
        try:
            kde_x2 = np.linspace(log_data.min(), log_data.max(), 300)
            kde2   = stats.gaussian_kde(
                log_data.sample(min(len(log_data), 10000), random_state=42))
            ax3.plot(kde_x2, kde2(kde_x2), color=palette['primary'],
                     linewidth=2, label='KDE')
        except Exception:
            pass
        ax3.axvline(log_data.mean(),   color=palette['success'], linewidth=1.5,
                    linestyle='-',  label=f'Mean: {log_data.mean():.2f}')
        ax3.axvline(log_data.median(), color=palette['warn'],    linewidth=1.5,
                    linestyle='--', label=f'Median: {log_data.median():.2f}')
        ax3.set_title(f'{log_col} — (Log Transformed)')
        ax3.set_xlabel(f'Skew: {log_data.skew():.2f}  |  Kurt: {log_data.kurt():.2f}',
                       fontsize=8)
        ax3.legend(fontsize=8)
    else:
        # Discount: log nahi — positive-only values par log1p apply karke dikhao
        pos_data = data[data > 0]
        if len(pos_data) > 0:
            log_manual = np.log1p(pos_data)
            ax3.hist(log_manual, bins=60, color=palette['purple'], alpha=0.55,
                     density=True, edgecolor='white', linewidth=0.3)
            ax3.set_title(f'log1p({col}) — Positive orders only\n'
                          f'({len(pos_data):,} of {len(data):,} orders)')
            ax3.set_xlabel(f'Skew after log1p: {log_manual.skew():.2f}', fontsize=8)
            zero_pct = (data == 0).sum() / len(data) * 100
            ax3.text(0.05, 0.90, f'Zero-discount orders: {zero_pct:.1f}%',
                     transform=ax3.transAxes, fontsize=8, color=palette['accent'],
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        ax3.grid(linestyle='--', alpha=0.4)

    ax3.grid(linestyle='--', alpha=0.4)
    ax3.tick_params(labelsize=8)

plt.tight_layout(rect=[0, 0.01, 1, 0.93])
plt.savefig(os.path.join(PLOTS_DIR, 'eda_13a_financial_univariate.png'),
            dpi=300, bbox_inches='tight')
plt.close('all')
print(f"Saved: {PLOTS_DIR}/eda_13a_financial_univariate.png")


# =============================================================================
# PLOT 13B: BIVARIATE ANALYSIS — Financial Features vs Target (Boxplots)
# Har feature ko quantity class (1-5) ke against compare karo
# ANOVA F-test se statistical significance confirm karo
# =============================================================================

print("\n" + "="*60)
print("STEP 13B: Financial Features vs Target — Bivariate Analysis")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    'STEP 13B: Financial Features vs order_item_quantity\n'
    'Box plots per Quantity Class + ANOVA Significance',
    fontsize=15, fontweight='bold', color=palette['primary'], y=0.995
)
axes = axes.flatten()

for i, col in enumerate(FIN_FEATURES):
    ax = axes[i]
    groups = [df[df[TARGET] == qty][col].dropna()
              for qty in sorted(df[TARGET].unique())]

    bp = ax.boxplot(groups, patch_artist=True, notch=False,
                    flierprops=dict(marker='.', markersize=2, alpha=0.25,
                                    color=palette['accent']),
                    medianprops=dict(color='white', linewidth=2))

    qty_list = sorted(df[TARGET].unique())
    for patch, qty in zip(bp['boxes'], qty_list):
        patch.set_facecolor(qty_colors[qty])
        patch.set_alpha(0.75)

    # ANOVA
    f_stat, p_val = stats.f_oneway(*groups)
    sig       = '✅ Significant' if p_val < 0.05 else '❌ Not Significant'
    sig_color = palette['success'] if p_val < 0.05 else palette['accent']

    # Group means line overlay
    means = [g.mean() for g in groups]
    ax.plot(range(1, len(means)+1), means, 'D--', color=palette['primary'],
            markersize=6, linewidth=1.5, alpha=0.8, label='Group Mean')

    ax.set_title(col, fontsize=10, fontweight='bold')
    ax.set_xlabel(
        f'ANOVA: F={f_stat:.2f},  p={p_val:.4e}\n{sig}',
        fontsize=8, color=sig_color
    )
    ax.set_ylabel('Value', fontsize=8)
    ax.set_xticklabels([f'Qty {q}' for q in qty_list], fontsize=8, rotation=20)
    ax.legend(fontsize=7)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    print(f"\n  {col}:")
    print(f"    ANOVA  F={f_stat:.2f},  p={p_val:.4e}  → {sig}")
    for qty, grp in zip(qty_list, groups):
        print(f"    Qty {qty}: mean={grp.mean():.2f}  |  median={grp.median():.2f}")

plt.tight_layout(rect=[0, 0.01, 1, 0.93])
plt.savefig(os.path.join(PLOTS_DIR, 'eda_13b_financial_vs_target_boxplot.png'),
            dpi=300, bbox_inches='tight')
plt.close('all')
print(f"\nSaved: {PLOTS_DIR}/eda_13b_financial_vs_target_boxplot.png")


# =============================================================================
# PLOT 13C: MEAN TREND LINES — Quantity class ke sath financial metrics kaise change hoti hain
# Line plot: har quantity class par mean value clearly dikhana
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(
    'STEP 13C: Mean Financial Metric by Quantity Class\n'
    'Trend lines — Kya quantity badhne par value bhi barhti hai?',
    fontsize=15, fontweight='bold', color=palette['primary'], y=0.995
)
axes = axes.flatten()

qty_vals = sorted(df[TARGET].unique())

for i, col in enumerate(FIN_FEATURES):
    ax = axes[i]
    means  = df.groupby(TARGET)[col].mean()
    stds   = df.groupby(TARGET)[col].std()

    ax.fill_between(qty_vals,
                    [means[q] - stds[q] for q in qty_vals],
                    [means[q] + stds[q] for q in qty_vals],
                    alpha=0.15, color=palette['highlight'], label='±1 Std Dev')
    ax.plot(qty_vals, [means[q] for q in qty_vals],
            'o-', color=palette['highlight'], linewidth=2.5,
            markersize=9, markeredgecolor='white', markeredgewidth=1.5)

    for qty in qty_vals:
        ax.annotate(f'{means[qty]:.1f}',
                    xy=(qty, means[qty]),
                    xytext=(0, 12), textcoords='offset points',
                    ha='center', fontsize=8.5, fontweight='bold',
                    color=palette['primary'])

    # Trend direction
    slope = np.polyfit(qty_vals, [means[q] for q in qty_vals], 1)[0]
    direction = '↑ Increasing' if slope > 0.5 else ('↓ Decreasing' if slope < -0.5 else '→ Flat')
    ax.set_title(f'{col}\nTrend: {direction}  (slope={slope:.2f})',
                 fontsize=9, fontweight='bold')
    ax.set_xlabel('Order Item Quantity', fontsize=8)
    ax.set_ylabel(f'Mean {col}', fontsize=8)
    ax.set_xticks(qty_vals)
    ax.legend(fontsize=8)
    ax.grid(linestyle='--', alpha=0.4)
    ax.tick_params(labelsize=8)

plt.tight_layout(rect=[0, 0.01, 1, 0.93])
plt.savefig(os.path.join(PLOTS_DIR, 'eda_13c_financial_mean_trends.png'),
            dpi=300, bbox_inches='tight')
plt.close('all')
print(f"Saved: {PLOTS_DIR}/eda_13c_financial_mean_trends.png")


# =============================================================================
# PLOT 13D: CORRELATION — Financial Features ke beech aur Target ke sath
# Heatmap + Target correlation bar chart
# Multicollinearity identify karo → modeling mein ek hi rakhna hoga
# =============================================================================

fin_corr_cols = FIN_FEATURES + [TARGET]
# Log versions bhi shamil karo
log_cols_present = [v for v in LOG_FEATURES.values() if v and v in df.columns]
fin_corr_cols_extended = FIN_FEATURES + log_cols_present + [TARGET]
fin_corr_cols_extended = list(dict.fromkeys(fin_corr_cols_extended))  # deduplicate

corr_fin = df[fin_corr_cols_extended].corr()

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle(
    'STEP 13D: Correlation Analysis — Financial Features\n'
    'Heatmap (with log versions) + Target Correlation Bar',
    fontsize=15, fontweight='bold', color=palette['primary'], y=0.995
)

# Panel 1: Heatmap
ax1 = axes[0]
mask = np.triu(np.ones_like(corr_fin, dtype=bool))
sns.heatmap(corr_fin, mask=mask, ax=ax1,
            cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            annot=True, fmt='.2f', annot_kws={'size': 8},
            linewidths=0.5, linecolor='white',
            cbar_kws={'shrink': 0.75, 'label': 'Pearson r'})
ax1.set_title('Correlation Heatmap\n(Raw + Log versions)', fontsize=10)
ax1.tick_params(axis='x', rotation=45, labelsize=8)
ax1.tick_params(axis='y', rotation=0,  labelsize=8)

# Panel 2: Target correlation bar
ax2 = axes[1]
target_corr_fin = corr_fin[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
colors_bar = [palette['success'] if v > 0 else palette['accent']
              for v in target_corr_fin.values]
bars = ax2.barh(target_corr_fin.index, target_corr_fin.values,
                color=colors_bar, alpha=0.80, edgecolor='white')
ax2.axvline(0, color=palette['primary'], linewidth=1.2)
for bar, val in zip(bars, target_corr_fin.values):
    ax2.text(val + (0.005 if val >= 0 else -0.005),
             bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center', ha='left' if val >= 0 else 'right',
             fontsize=8.5, fontweight='bold')
ax2.set_title(f"Correlation with '{TARGET}'\n(Sorted by |r|)", fontsize=10)
ax2.set_xlabel('Pearson r', fontsize=9)
ax2.grid(axis='x', linestyle='--', alpha=0.4)

plt.tight_layout(rect=[0, 0.01, 1, 0.93])
plt.savefig(os.path.join(PLOTS_DIR, 'eda_13d_financial_correlation.png'),
            dpi=300, bbox_inches='tight')
plt.close('all')
print(f"Saved: {PLOTS_DIR}/eda_13d_financial_correlation.png")

print("\nCorrelation with target (financial features):")
print(target_corr_fin.round(4).to_string())


# =============================================================================
# PLOT 13E: SCATTER PLOTS — Financial Features vs Target + Quantity Color
# Sample 5000 points, color by quantity class
# Trend pattern aur cluster behavior dekhna
# =============================================================================

sample = df.sample(min(len(df), 5000), random_state=42)
scatter_colors = [qty_colors[q] for q in sample[TARGET]]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    'STEP 13E: Scatter Plots — Financial Features vs order_item_quantity\n'
    'Color = Quantity Class  |  Sample = 5,000 orders',
    fontsize=15, fontweight='bold', color=palette['primary'], y=0.995
)
axes = axes.flatten()

scatter_pairs = [
    ('order_item_total',    TARGET,    'order_item_total vs Quantity'),
    ('order_item_discount', TARGET,    'order_item_discount vs Quantity'),
    ('sales_per_customer',  TARGET,    'sales_per_customer vs Quantity'),
    ('sales',               TARGET,    'sales vs Quantity'),
]

for i, (x_col, y_col, title) in enumerate(scatter_pairs):
    ax = axes[i]
    ax.scatter(sample[x_col], sample[y_col],
               c=scatter_colors, alpha=0.30, s=8)

    # Per-class mean markers
    for qty in sorted(df[TARGET].unique()):
        grp = df[df[TARGET] == qty]
        ax.scatter(grp[x_col].mean(), qty,
                   color=qty_colors[qty], s=120, marker='D',
                   edgecolor='white', linewidth=1.2, zorder=5,
                   label=f'Qty {qty} mean')

    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_xlabel(x_col, fontsize=8)
    ax.set_ylabel('order_item_quantity', fontsize=8)
    ax.set_yticks(sorted(df[TARGET].unique()))
    ax.legend(fontsize=6.5, ncol=2, title='Class Mean ◆')
    ax.grid(linestyle='--', alpha=0.35)
    ax.tick_params(labelsize=8)

plt.tight_layout(rect=[0, 0.01, 1, 0.93])
plt.savefig(os.path.join(PLOTS_DIR, 'eda_13e_financial_scatter.png'),
            dpi=300, bbox_inches='tight')
plt.close('all')
print(f"Saved: {PLOTS_DIR}/eda_13e_financial_scatter.png")


# =============================================================================
# PLOT 13F: DISCOUNT DEEP DIVE
# order_item_discount ka special analysis — ye feature unique hai:
# - Bahut saare orders mein discount = 0
# - Zero-inflated distribution hai
# - Discount aur quantity ka relationship retail ka core question hai
# =============================================================================

print("\n" + "="*60)
print("STEP 13F: DISCOUNT FEATURE — DEEP DIVE")
print("="*60)

disc = df['order_item_discount']
zero_disc   = (disc == 0).sum()
nonzero_pct = (disc > 0).sum() / len(disc) * 100
print(f"\n  Total orders       : {len(disc):,}")
print(f"  Zero-discount      : {zero_disc:,}  ({zero_disc/len(disc)*100:.1f}%)")
print(f"  Non-zero discount  : {(disc > 0).sum():,}  ({nonzero_pct:.1f}%)")
print(f"  Max discount       : {disc.max():.2f}")
print(f"  Mean (non-zero)    : {disc[disc > 0].mean():.2f}")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(
    'STEP 13F: order_item_discount — Deep Dive Analysis\n'
    'Zero-Inflated Feature | Discount Behavior by Quantity Class',
    fontsize=15, fontweight='bold', color=palette['primary'], y=0.995
)

# F1: Zero vs Non-Zero pie
ax = axes[0][0]
ax.pie([zero_disc, len(disc) - zero_disc],
       labels=['No Discount (=0)', 'Has Discount (>0)'],
       autopct='%1.1f%%',
       colors=[palette['accent'], palette['success']],
       startangle=90,
       wedgeprops=dict(linewidth=1.5, edgecolor='white'))
ax.set_title('Discount Presence\n(Zero vs Non-Zero Orders)', fontsize=9)

# F2: Non-zero discount distribution
ax2 = axes[0][1]
nonzero_disc = disc[disc > 0]
ax2.hist(nonzero_disc, bins=50, color=palette['highlight'],
         alpha=0.60, edgecolor='white', density=True)
try:
    kde_x = np.linspace(nonzero_disc.min(), nonzero_disc.max(), 300)
    kde   = stats.gaussian_kde(nonzero_disc.sample(min(len(nonzero_disc), 10000), random_state=42))
    ax2.plot(kde_x, kde(kde_x), color=palette['accent'], linewidth=2)
except Exception:
    pass
ax2.axvline(nonzero_disc.mean(),   color=palette['success'], linewidth=1.5,
            linestyle='-',  label=f'Mean: {nonzero_disc.mean():.2f}')
ax2.axvline(nonzero_disc.median(), color=palette['warn'], linewidth=1.5,
            linestyle='--', label=f'Median: {nonzero_disc.median():.2f}')
ax2.set_title('Non-Zero Discount Distribution\n(Orders jahan discount tha)')
ax2.set_xlabel('Discount Amount ($)', fontsize=8)
ax2.legend(fontsize=8)
ax2.grid(linestyle='--', alpha=0.4)

# F3: Avg discount per quantity class
ax3 = axes[0][2]
avg_disc_all = df.groupby(TARGET)['order_item_discount'].mean()
avg_disc_nz  = df[df['order_item_discount'] > 0].groupby(TARGET)['order_item_discount'].mean()

x = np.array(sorted(df[TARGET].unique()))
width = 0.35
ax3.bar(x - width/2, avg_disc_all.values, width, color=palette['highlight'],
        alpha=0.75, edgecolor='white', label='All orders (incl. zero)')
ax3.bar(x + width/2, avg_disc_nz.values,  width, color=palette['success'],
        alpha=0.75, edgecolor='white', label='Non-zero discount only')
ax3.set_title('Avg Discount by Quantity Class\n(All vs Non-Zero Orders)', fontsize=9)
ax3.set_xlabel('Order Quantity', fontsize=8)
ax3.set_ylabel('Avg Discount ($)', fontsize=8)
ax3.set_xticks(x)
ax3.set_xticklabels([f'Qty {q}' for q in x], fontsize=8)
ax3.legend(fontsize=8)
ax3.grid(axis='y', linestyle='--', alpha=0.4)

# F4: % of orders with discount per quantity class
ax4 = axes[1][0]
disc_pct_per_qty = df.groupby(TARGET).apply(
    lambda g: (g['order_item_discount'] > 0).sum() / len(g) * 100
)
bars = ax4.bar(disc_pct_per_qty.index, disc_pct_per_qty.values,
               color=[qty_colors[q] for q in disc_pct_per_qty.index],
               alpha=0.80, edgecolor='white')
for bar, val in zip(bars, disc_pct_per_qty.values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')
ax4.set_title('% Orders WITH Discount\nby Quantity Class', fontsize=9)
ax4.set_xlabel('Order Quantity', fontsize=8)
ax4.set_ylabel('% Orders with Discount > 0', fontsize=8)
ax4.set_xticks(sorted(df[TARGET].unique()))
ax4.set_xticklabels([f'Qty {q}' for q in sorted(df[TARGET].unique())], fontsize=8)
ax4.grid(axis='y', linestyle='--', alpha=0.4)
ax4.set_ylim(0, disc_pct_per_qty.max() * 1.15)

# F5: Discount vs order_item_total (scatter — kya discount badhne par total value bhi barhti hai?)
ax5 = axes[1][1]
sub = df[df['order_item_discount'] > 0].sample(min(5000, (df['order_item_discount'] > 0).sum()), random_state=42)
sc_colors = [qty_colors[q] for q in sub[TARGET]]
ax5.scatter(sub['order_item_discount'], sub['order_item_total'],
            c=sc_colors, alpha=0.25, s=6)
# Trend line
z = np.polyfit(sub['order_item_discount'], sub['order_item_total'], 1)
p = np.poly1d(z)
x_line = np.linspace(sub['order_item_discount'].min(), sub['order_item_discount'].max(), 200)
ax5.plot(x_line, p(x_line), color=palette['primary'], linewidth=2, linestyle='--',
         label=f'Trend (slope={z[0]:.2f})')
ax5.set_title('Discount vs order_item_total\n(Non-zero discount orders only)', fontsize=9)
ax5.set_xlabel('order_item_discount ($)', fontsize=8)
ax5.set_ylabel('order_item_total ($)', fontsize=8)
from matplotlib.patches import Patch
handles = [Patch(color=qty_colors[q], label=f'Qty {q}') for q in sorted(df[TARGET].unique())]
ax5.legend(handles=handles + [plt.Line2D([0],[0], color=palette['primary'],
           linestyle='--', label='Trend')], fontsize=7, title='Quantity')
ax5.grid(linestyle='--', alpha=0.35)

# F6: discount_rate vs order_item_discount — consistency check
ax6 = axes[1][2]
if 'order_item_discount_rate' in df.columns:
    sub2 = df.sample(min(5000, len(df)), random_state=42)
    ax6.scatter(sub2['order_item_discount_rate'], sub2['order_item_discount'],
                c=[qty_colors[q] for q in sub2[TARGET]], alpha=0.25, s=6)
    ax6.set_title('Discount Rate vs Discount Amount\n(Sanity check — proportional?)', fontsize=9)
    ax6.set_xlabel('order_item_discount_rate', fontsize=8)
    ax6.set_ylabel('order_item_discount ($)', fontsize=8)
    ax6.grid(linestyle='--', alpha=0.35)
    handles2 = [Patch(color=qty_colors[q], label=f'Qty {q}')
                for q in sorted(df[TARGET].unique())]
    ax6.legend(handles=handles2, fontsize=7, title='Quantity')

plt.tight_layout(rect=[0, 0.01, 1, 0.93])
plt.savefig(os.path.join(PLOTS_DIR, 'eda_13f_discount_deepdive.png'),
            dpi=300, bbox_inches='tight')
plt.close('all')
print(f"Saved: {PLOTS_DIR}/eda_13f_discount_deepdive.png")


# =============================================================================
# PLOT 13G: SALES & REVENUE ECOSYSTEM
# sales, sales_per_customer, order_item_total ke beech business relationship
# In teeno ka combined analysis retail revenue pattern reveal karta hai
# =============================================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(
    'STEP 13G: Sales & Revenue Ecosystem\n'
    'sales | sales_per_customer | order_item_total — Inter-relationships',
    fontsize=15, fontweight='bold', color=palette['primary'], y=0.995
)

# G1: sales vs order_item_total scatter
ax = axes[0][0]
sample2 = df.sample(min(5000, len(df)), random_state=42)
sc = ax.scatter(sample2['order_item_total'], sample2['sales'],
                c=[qty_colors[q] for q in sample2[TARGET]],
                alpha=0.25, s=6)
z  = np.polyfit(df['order_item_total'], df['sales'], 1)
x_vals = np.linspace(df['order_item_total'].quantile(0.01),
                      df['order_item_total'].quantile(0.99), 200)
ax.plot(x_vals, np.poly1d(z)(x_vals), color=palette['primary'],
        linewidth=2, linestyle='--', label=f'r={df[["order_item_total","sales"]].corr().iloc[0,1]:.3f}')
ax.set_title('order_item_total vs sales\n(Kya ye almost same hain?)', fontsize=9)
ax.set_xlabel('order_item_total ($)', fontsize=8)
ax.set_ylabel('sales ($)', fontsize=8)
ax.legend(fontsize=8)
ax.grid(linestyle='--', alpha=0.35)

# G2: sales_per_customer by quantity class (violin)
ax2 = axes[0][1]
spc_data = [df[df[TARGET] == q]['sales_per_customer'].dropna() for q in sorted(df[TARGET].unique())]
parts = ax2.violinplot(spc_data, positions=sorted(df[TARGET].unique()),
                       showmedians=True, showextrema=True)
for pc, qty in zip(parts['bodies'], sorted(df[TARGET].unique())):
    pc.set_facecolor(qty_colors[qty])
    pc.set_alpha(0.65)
parts['cmedians'].set_color('white')
parts['cmedians'].set_linewidth(2)
ax2.set_title('sales_per_customer Distribution\nby Quantity Class (Violin)', fontsize=9)
ax2.set_xlabel('Order Quantity', fontsize=8)
ax2.set_ylabel('sales_per_customer ($)', fontsize=8)
ax2.set_xticks(sorted(df[TARGET].unique()))
ax2.set_xticklabels([f'Qty {q}' for q in sorted(df[TARGET].unique())], fontsize=8)
ax2.grid(axis='y', linestyle='--', alpha=0.4)

# G3: order_item_total distribution by quantity (violin)
ax3 = axes[0][2]
tot_data = [df[df[TARGET] == q]['order_item_total'].dropna() for q in sorted(df[TARGET].unique())]
parts3   = ax3.violinplot(tot_data, positions=sorted(df[TARGET].unique()),
                          showmedians=True, showextrema=True)
for pc, qty in zip(parts3['bodies'], sorted(df[TARGET].unique())):
    pc.set_facecolor(qty_colors[qty])
    pc.set_alpha(0.65)
parts3['cmedians'].set_color('white')
parts3['cmedians'].set_linewidth(2)
ax3.set_title('order_item_total Distribution\nby Quantity Class (Violin)', fontsize=9)
ax3.set_xlabel('Order Quantity', fontsize=8)
ax3.set_ylabel('order_item_total ($)', fontsize=8)
ax3.set_xticks(sorted(df[TARGET].unique()))
ax3.set_xticklabels([f'Qty {q}' for q in sorted(df[TARGET].unique())], fontsize=8)
ax3.grid(axis='y', linestyle='--', alpha=0.4)

# G4: Cumulative Revenue share by quantity class
ax4 = axes[1][0]
rev_by_qty = df.groupby(TARGET)['sales'].sum().sort_index()
total_rev  = rev_by_qty.sum()
pct_rev    = rev_by_qty / total_rev * 100
order_cnt  = df[TARGET].value_counts().sort_index()
pct_orders = order_cnt / order_cnt.sum() * 100

x   = np.array(sorted(df[TARGET].unique()))
w   = 0.35
ax4.bar(x - w/2, pct_orders.values, w, color=palette['highlight'],
        alpha=0.75, edgecolor='white', label='% Orders')
ax4.bar(x + w/2, pct_rev.values,    w, color=palette['success'],
        alpha=0.75, edgecolor='white', label='% Revenue')
for xi, (o, r) in zip(x, zip(pct_orders.values, pct_rev.values)):
    ax4.text(xi - w/2, o + 0.3, f'{o:.1f}%', ha='center', fontsize=7.5, fontweight='bold')
    ax4.text(xi + w/2, r + 0.3, f'{r:.1f}%', ha='center', fontsize=7.5,
             fontweight='bold', color=palette['success'])
ax4.set_title('% Orders vs % Revenue\nby Quantity Class', fontsize=9)
ax4.set_xlabel('Order Quantity', fontsize=8)
ax4.set_ylabel('Percentage (%)', fontsize=8)
ax4.set_xticks(x)
ax4.set_xticklabels([f'Qty {q}' for q in x], fontsize=8)
ax4.legend(fontsize=8)
ax4.grid(axis='y', linestyle='--', alpha=0.4)

# G5: sales_per_customer vs sales scatter
ax5 = axes[1][1]
samp3 = df.sample(min(5000, len(df)), random_state=42)
ax5.scatter(samp3['sales_per_customer'], samp3['sales'],
            c=[qty_colors[q] for q in samp3[TARGET]], alpha=0.25, s=6)
r_val = df[['sales_per_customer', 'sales']].corr().iloc[0, 1]
ax5.set_title(f'sales_per_customer vs sales\nr = {r_val:.3f}', fontsize=9)
ax5.set_xlabel('sales_per_customer ($)', fontsize=8)
ax5.set_ylabel('sales ($)', fontsize=8)
handles = [Patch(color=qty_colors[q], label=f'Qty {q}') for q in sorted(df[TARGET].unique())]
ax5.legend(handles=handles, fontsize=7, title='Quantity')
ax5.grid(linestyle='--', alpha=0.35)

# G6: Per-unit revenue (order_item_total / quantity) — actual value per unit
ax6 = axes[1][2]
df['revenue_per_unit'] = df['order_item_total'] / df[TARGET]
rpu_means = df.groupby(TARGET)['revenue_per_unit'].mean()
bars6 = ax6.bar(rpu_means.index, rpu_means.values,
                color=[qty_colors[q] for q in rpu_means.index],
                alpha=0.80, edgecolor='white')
for bar, val in zip(bars6, rpu_means.values):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'${val:.1f}', ha='center', fontsize=9, fontweight='bold')
ax6.set_title('Avg Revenue Per Unit\n(order_item_total ÷ quantity)', fontsize=9)
ax6.set_xlabel('Order Quantity', fontsize=8)
ax6.set_ylabel('Revenue Per Unit ($)', fontsize=8)
ax6.set_xticks(sorted(df[TARGET].unique()))
ax6.set_xticklabels([f'Qty {q}' for q in sorted(df[TARGET].unique())], fontsize=8)
ax6.grid(axis='y', linestyle='--', alpha=0.4)

# Cleanup temp column
df.drop(columns=['revenue_per_unit'], inplace=True)

plt.tight_layout(rect=[0, 0.01, 1, 0.93])
plt.savefig(os.path.join(PLOTS_DIR, 'eda_13g_sales_revenue_ecosystem.png'),
            dpi=300, bbox_inches='tight')
plt.close('all')
print(f"Saved: {PLOTS_DIR}/eda_13g_sales_revenue_ecosystem.png")


# =============================================================================
# PLOT 13H: LOG-TRANSFORMED VERSIONS vs TARGET
# Log versions pehle se dataset mein hain — inki bivariate analysis
# (order_item_total_log, sales_log, sales_per_customer_log)
# =============================================================================

log_cols_available = [c for c in ['order_item_total_log', 'sales_log', 'sales_per_customer_log']
                      if c in df.columns]

if log_cols_available:
    n_log = len(log_cols_available)
    fig, axes = plt.subplots(2, n_log, figsize=(6 * n_log, 11))
    if n_log == 1:
        axes = axes.reshape(2, 1)
    fig.suptitle(
        'STEP 13H: Log-Transformed Financial Features vs Target\n'
        'Boxplot + Mean trend — Log versions better predictors hain?',
        fontsize=15, fontweight='bold', color=palette['primary'], y=0.995
    )

    for i, lcol in enumerate(log_cols_available):
        # Row 0: Boxplot
        ax_top = axes[0][i]
        groups = [df[df[TARGET] == q][lcol].dropna() for q in sorted(df[TARGET].unique())]
        bp = ax_top.boxplot(groups, patch_artist=True,
                            flierprops=dict(marker='.', markersize=2, alpha=0.2,
                                            color=palette['accent']),
                            medianprops=dict(color='white', linewidth=2))
        for patch, qty in zip(bp['boxes'], sorted(df[TARGET].unique())):
            patch.set_facecolor(qty_colors[qty])
            patch.set_alpha(0.72)
        f_stat, p_val = stats.f_oneway(*groups)
        sig = '✅ Sig' if p_val < 0.05 else '❌ Not Sig'
        ax_top.set_title(f'{lcol}\nANOVA F={f_stat:.1f}, p={p_val:.2e} {sig}', fontsize=9)
        ax_top.set_xticklabels([f'Qty {q}' for q in sorted(df[TARGET].unique())],
                               fontsize=8, rotation=20)
        ax_top.grid(axis='y', linestyle='--', alpha=0.4)

        # Row 1: Mean trend
        ax_bot = axes[1][i]
        means  = [df[df[TARGET] == q][lcol].mean() for q in sorted(df[TARGET].unique())]
        ax_bot.plot(sorted(df[TARGET].unique()), means,
                    'o-', color=palette['teal'], linewidth=2.5, markersize=9,
                    markeredgecolor='white', markeredgewidth=1.5)
        for qty, m in zip(sorted(df[TARGET].unique()), means):
            ax_bot.annotate(f'{m:.2f}', xy=(qty, m),
                            xytext=(0, 10), textcoords='offset points',
                            ha='center', fontsize=8.5, fontweight='bold')
        raw_col = lcol.replace('_log', '')
        raw_r   = df[[raw_col, TARGET]].corr().iloc[0, 1] if raw_col in df.columns else None
        log_r   = df[[lcol, TARGET]].corr().iloc[0, 1]
        note    = f'Log r={log_r:.3f}'
        if raw_r is not None:
            note += f'  |  Raw r={raw_r:.3f}'
        ax_bot.set_title(f'Mean {lcol} per Qty\n{note}', fontsize=9)
        ax_bot.set_xlabel('Order Quantity', fontsize=8)
        ax_bot.set_ylabel(lcol, fontsize=8)
        ax_bot.set_xticks(sorted(df[TARGET].unique()))
        ax_bot.grid(linestyle='--', alpha=0.4)

    plt.tight_layout(rect=[0, 0.01, 1, 0.93])
    plt.savefig(os.path.join(PLOTS_DIR, 'eda_13h_log_features_vs_target.png'),
                dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f"Saved: {PLOTS_DIR}/eda_13h_log_features_vs_target.png")


# =============================================================================
# MARKDOWN INSIGHTS — STEP 13: Re-Included Financial Features
# =============================================================================
#
# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 1: order_item_total ($)
# Kya Hai: Ek order item ka total value = product price × quantity − discount
# ─────────────────────────────────────────────────────────────────────────────
#
# Distribution:
#   Right-skewed hai (skew > 1.5). Chote orders (< $100) ka volume bohot
#   zyada hai. Log transform ke baad distribution significantly improve hoti
#   hai aur ML models ke liye zyada suitable ban jati hai.
#
# Bivariate Insight (vs Quantity):
#   order_item_total INCREASES with quantity — ye expected hai kyunki
#   total = price × qty. Lekin increase linear nahi hoti kyunki higher
#   quantity wale orders mein per-unit discount apply hota hai. Quantity 5
#   wale orders ka total highest hai lekin per-unit revenue kam hai.
#   ANOVA: Statistically significant (p << 0.05).
#
# Multicollinearity Risk:
#   order_item_total aur sales ke beech r ≈ 0.99 — ye almost identical hain.
#   Dono ek saath model mein mat daalo. Log version prefer karo.
#
# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 2: order_item_discount ($)
# Kya Hai: Har order item par jo discount dollar amount apply hui
# ─────────────────────────────────────────────────────────────────────────────
#
# Distribution (Zero-Inflated):
#   Bahut zyada orders mein discount = $0. Ye ek "zero-inflated" feature hai.
#   Normal distribution assume karna yahan galat hoga. Models ko ye bata dena
#   zaroori hai ke kya koi discount tha ya nahi — binary flag zaroori hai.
#
# Bivariate Insight (vs Quantity):
#   Higher quantity wale orders mein average discount BARHTA HAI — ye retail
#   ki bulk discount policy ko confirm karta hai. Lekin ye relationship weak
#   hai (ANOVA p < 0.05 lekin effect size chhota hai). Discount presence
#   (binary) quantity se zyada correlated hai compared to discount amount.
#
# Business Critical Insight:
#   order_item_discount aur order_item_discount_rate dono dataset mein hain.
#   Rate (%) zyada interpretable hai aur scale-invariant hai. Amount ($) zyada
#   sensitive to product price hai. Dono rakhna theek hai lekin ek choose
#   karna ho toh rate prefer karo (already existing feature hai).
#
# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 3: sales_per_customer ($)
# Kya Hai: Ek customer ne is order session mein total kitna kharcha kiya
# ─────────────────────────────────────────────────────────────────────────────
#
# Distribution:
#   Bimodal distribution possible hai — kuch customers single item khareed
#   rahe hain, kuch multiple items. Right tail long hai (high-value customers).
#   Log version (sales_per_customer_log) zyada symmetric hai.
#
# Bivariate Insight (vs Quantity):
#   sales_per_customer aur order_item_quantity ke beech POSITIVE relationship
#   hai — jo customer zyada kharcha karta hai, woh zyada quantity bhi order
#   karta hai. Ye Customer Lifetime Value (CLV) ka ek proxy measure hai.
#   Ye feature alag perspective deta hai: per-order view nahi, per-customer
#   view.
#
# Multicollinearity Risk:
#   sales_per_customer ≈ sales aur order_item_total ke beech high correlation
#   expect karo. Modeling phase mein VIF check zaroori hoga.
#
# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 4: sales ($)
# Kya Hai: Actual realized sales amount (after discount adjustment)
# ─────────────────────────────────────────────────────────────────────────────
#
# Distribution:
#   order_item_total se almost identical distribution. r ≈ 0.99 between
#   sales aur order_item_total — ye almost duplicate features hain. Difference
#   sirf rounding ya minor accounting adjustments se aata hai.
#
# Bivariate Insight (vs Quantity):
#   order_item_total jaisi hi pattern. Quantity badhne ke sath sales barhti
#   hai — trivially expected kyunki quantity × price = total.
#
# Key Decision:
#   sales aur order_item_total DONO simultaneously model mein daalna
#   HARMFUL hoga (perfect multicollinearity). Sirf LOG version rakho —
#   sales_log ya order_item_total_log — ek baar feature engineering mein
#   VIF analysis karke decide karo kaun drop karna hai.
#
# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING SUGGESTIONS (In 4 Features Se)
# ─────────────────────────────────────────────────────────────────────────────
#
# 1. has_discount (Binary Flag) — RECOMMENDED
#    df['has_discount'] = (df['order_item_discount'] > 0).astype(int)
#    Reason: Zero-inflated feature ko binary mein convert karna zyada
#    effective hai. Model ko 'koi discount tha ya nahi' samajhna zyada
#    relevant hai than exact dollar amount.
#
# 2. discount_intensity (Categorized Discount)
#    df['discount_intensity'] = pd.cut(
#        df['order_item_discount'],
#        bins=[-0.01, 0, 5, 15, float('inf')],
#        labels=['No Discount', 'Low ($1-5)', 'Medium ($6-15)', 'High (>$15)']
#    )
#    Reason: Continuous discount mein noise hai. 4 buckets model ko zyada
#    meaningful signal dete hain.
#
# 3. Keep: sales_log OR order_item_total_log (sirf ek — VIF decide karega)
#    Reason: Dono almost identical hain. Log version skewness handle karta
#    hai. Ek ko feature engineering phase mein drop kar dena hoga.
#
# 4. Keep: sales_per_customer_log
#    Reason: Customer-level perspective deta hai jo order-level features
#    se alag hai. CLV proxy — modeling mein useful signal.
#
# 5. DROP from final model: raw sales, raw order_item_total (log use karo)
#    ALSO DROP: order_item_discount (has_discount + discount_intensity use karo)
#
# ─────────────────────────────────────────────────────────────────────────────
# PLOTS GENERATED (Step 13):
#   EDA_Plots/eda_13a_financial_univariate.png
#   EDA_Plots/eda_13b_financial_vs_target_boxplot.png
#   EDA_Plots/eda_13c_financial_mean_trends.png
#   EDA_Plots/eda_13d_financial_correlation.png
#   EDA_Plots/eda_13e_financial_scatter.png
#   EDA_Plots/eda_13f_discount_deepdive.png
#   EDA_Plots/eda_13g_sales_revenue_ecosystem.png
#   EDA_Plots/eda_13h_log_features_vs_target.png
# =============================================================================

print("\n" + "="*65)
print("  STEP 13 COMPLETE — Financial Features EDA Done")
print("="*65)
print("""
PLOTS GENERATED:
  EDA_Plots/eda_13a_financial_univariate.png
  EDA_Plots/eda_13b_financial_vs_target_boxplot.png
  EDA_Plots/eda_13c_financial_mean_trends.png
  EDA_Plots/eda_13d_financial_correlation.png
  EDA_Plots/eda_13e_financial_scatter.png
  EDA_Plots/eda_13f_discount_deepdive.png
  EDA_Plots/eda_13g_sales_revenue_ecosystem.png
  EDA_Plots/eda_13h_log_features_vs_target.png

KEY DECISIONS FOR MODELING PHASE:
  ✅ sales_log OR order_item_total_log  → sirf ek rakhna (VIF decide karega)
  ❌ sales_per_customer_log             → drop karo (Sales = Quantity x Price → target leakage, Phase 9 FIX 6)
  ✅ has_discount (binary)              → banana (zero-inflated ko handle karo)
  ❌ discount_intensity (4 buckets)     → NAHI banana (dollar discount = qty x price x rate → target leakage)
  ❌ raw sales, raw order_item_total    → drop karo (log version use karo)
  ❌ order_item_discount (raw)          → drop karo (has_discount use karo)
""")
