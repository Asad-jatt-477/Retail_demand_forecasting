import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os
import matplotlib
matplotlib.use('Agg')  # headless-safe; no blocking plt.show() windows

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(ROOT_DIR, 'data', 'processed')

df = pd.read_csv(os.path.join(DATA_DIR, 'Retail_demand_forecasting_CLEANED.csv'), encoding='latin-1')

# =============================================================================
# OUTPUT FOLDER SETUP
# =============================================================================
PLOTS_DIR = os.path.join(ROOT_DIR, 'assets', 'plots', 'outlier_visual')
os.makedirs(PLOTS_DIR, exist_ok=True)
print(f"Plots folder ready: '{PLOTS_DIR}/'")

# Outliers detection using all visual methods
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
print(f"Total numeric columns found: {len(numeric_cols)}")
print(numeric_cols)

# ─── COLOR PALETTE ────────────────────────────────────────────────────────────
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

df_num = df[numeric_cols].copy()

# =============================================================================
# PLOTTING HELPERS — keep every figure a "normal", legible, non-overlapping size
# =============================================================================
PER_PAGE = 12           # max subplots per figure/page (3 cols x 4 rows)
GRID_COLS = 3


def paginate(items, per_page=PER_PAGE):
    return [items[i:i + per_page] for i in range(0, len(items), per_page)]


def new_grid(n_items, title, row_height=3.6):
    """A 3-column grid sized for n_items (<= PER_PAGE), with room reserved for
    the suptitle so it never overlaps the top row of subplot titles."""
    n_rows = int(np.ceil(n_items / GRID_COLS))
    fig, axes = plt.subplots(n_rows, GRID_COLS, figsize=(15, max(row_height * n_rows, 4)))
    axes = np.atleast_1d(axes).flatten()
    fig.suptitle(title, fontsize=15, fontweight='bold', color=palette['primary'], y=0.995)
    return fig, axes, n_rows


def finish_grid(fig, axes, used, save_path, n_rows):
    for j in range(used, len(axes)):
        axes[j].set_visible(False)
    top = 0.94 if n_rows <= 2 else (0.96 if n_rows <= 3 else 0.97)
    fig.tight_layout(rect=[0, 0.02, 1, top])
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved: {save_path}")


def heatmap_figsize(n_features):
    side = max(10, min(20, 0.42 * n_features + 6))
    return (side, side * 0.8)


def annot_fontsize(n_features):
    return max(5.5, min(9, 200 / max(n_features, 1)))


pages = paginate(numeric_cols)
print(f"\n{len(numeric_cols)} numeric columns -> {len(pages)} page(s) of up to {PER_PAGE} per chart, "
      f"so every chart stays a readable, normal size.")

# =============================================================================
# METHOD 1: BOX PLOTS — IQR-based outlier boundaries
# =============================================================================
print("\n[1/6] Generating Box Plots...")

for p, cols in enumerate(pages, start=1):
    fig, axes, n_rows = new_grid(len(cols), f'METHOD 1: Box Plots — IQR-Based Outlier Detection '
                                             f'(page {p}/{len(pages)})')
    for i, col in enumerate(cols):
        ax = axes[i]
        data = df_num[col].dropna()
        ax.boxplot(data, patch_artist=True, notch=False, vert=True,
                   flierprops=dict(marker='o', color=palette['accent'], markersize=3, alpha=0.5,
                                    markerfacecolor=palette['accent']),
                   medianprops=dict(color=palette['success'], linewidth=2),
                   boxprops=dict(facecolor='#AED6F1', color=palette['highlight']),
                   whiskerprops=dict(color=palette['primary'], linewidth=1.5),
                   capprops=dict(color=palette['primary'], linewidth=2))
        Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
        IQR = Q3 - Q1
        lb, ub = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        n_out = ((data < lb) | (data > ub)).sum()
        pct = 100 * n_out / len(data)
        ax.set_title(col, fontsize=10, pad=6)
        ax.set_xlabel(f'Outliers: {n_out:,} ({pct:.1f}%)', fontsize=8.5, color=palette['accent'])
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.tick_params(labelsize=8)
    finish_grid(fig, axes, len(cols), os.path.join(PLOTS_DIR, f'01_boxplots_page{p}.png'), n_rows)

# =============================================================================
# METHOD 2: HISTOGRAMS + KDE — Distribution Shape & Tails
# =============================================================================
print("\n[2/6] Generating Histograms + KDE...")

for p, cols in enumerate(pages, start=1):
    fig, axes, n_rows = new_grid(len(cols), f'METHOD 2: Histograms + KDE — Distribution Tails & Skewness '
                                             f'(page {p}/{len(pages)})')
    legend_handles = None
    for i, col in enumerate(cols):
        ax = axes[i]
        data = df_num[col].dropna()
        ax.hist(data, bins=50, color=palette['highlight'], alpha=0.55, density=True,
                edgecolor='white', linewidth=0.3, label='Distribution')
        try:
            kde_x = np.linspace(data.min(), data.max(), 300)
            kde = stats.gaussian_kde(data.sample(min(len(data), 10000), random_state=42))
            ax.plot(kde_x, kde(kde_x), color=palette['accent'], linewidth=2, label='KDE')
        except Exception:
            pass
        mu, sigma = data.mean(), data.std()
        ax.axvline(mu - 3 * sigma, color=palette['warn'], linestyle='--', linewidth=1.2, label='±3σ')
        ax.axvline(mu + 3 * sigma, color=palette['warn'], linestyle='--', linewidth=1.2)
        skew = data.skew()
        ax.set_title(col, fontsize=10)
        ax.set_xlabel(f'Skewness: {skew:.2f}', fontsize=8.5, color=palette['primary'])
        ax.grid(linestyle='--', alpha=0.4)
        ax.tick_params(labelsize=8)
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc='upper right', fontsize=9,
                   bbox_to_anchor=(0.99, 0.985), ncol=3, framealpha=0.9)
    finish_grid(fig, axes, len(cols), os.path.join(PLOTS_DIR, f'02_histograms_kde_page{p}.png'), n_rows)

# =============================================================================
# METHOD 3: VIOLIN PLOTS — Distribution Density + Range
# =============================================================================
print("\n[3/6] Generating Violin Plots...")

for p, cols in enumerate(pages, start=1):
    fig, axes, n_rows = new_grid(len(cols), f'METHOD 3: Violin Plots — Density Distribution & Spread '
                                             f'(page {p}/{len(pages)})')
    for i, col in enumerate(cols):
        ax = axes[i]
        data = df_num[col].dropna()
        sample = data.sample(min(len(data), 20000), random_state=42)
        parts = ax.violinplot(sample, vert=True, showmedians=True, showextrema=True)
        for pc in parts['bodies']:
            pc.set_facecolor('#AED6F1')
            pc.set_edgecolor(palette['highlight'])
            pc.set_alpha(0.75)
        parts['cmedians'].set_color(palette['success'])
        parts['cmedians'].set_linewidth(2)
        Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
        IQR = Q3 - Q1
        n_out = ((data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)).sum()
        ax.set_title(col, fontsize=10)
        ax.set_xlabel(f'IQR Outliers: {n_out:,}', fontsize=8.5, color=palette['accent'])
        ax.set_xticks([])
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.tick_params(labelsize=8)
    finish_grid(fig, axes, len(cols), os.path.join(PLOTS_DIR, f'03_violin_plots_page{p}.png'), n_rows)

# =============================================================================
# METHOD 4: SCATTER PLOTS — Index vs Value (Row-level anomalies)
# =============================================================================
print("\n[4/6] Generating Scatter Plots (Index vs Value)...")

for p, cols in enumerate(pages, start=1):
    fig, axes, n_rows = new_grid(len(cols), f'METHOD 4: Scatter Plots — Row-Level Anomaly Visualization '
                                             f'(page {p}/{len(pages)})', row_height=3.3)
    legend_handles = None
    for i, col in enumerate(cols):
        ax = axes[i]
        data = df_num[col].dropna()
        sample_idx = data.sample(min(len(data), 5000), random_state=42)
        Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
        IQR = Q3 - Q1
        lb, ub = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        is_out = (sample_idx < lb) | (sample_idx > ub)
        ax.scatter(sample_idx.index[~is_out], sample_idx[~is_out], s=2, alpha=0.3,
                   color=palette['highlight'], label='Normal')
        ax.scatter(sample_idx.index[is_out], sample_idx[is_out], s=6, alpha=0.7,
                   color=palette['accent'], label='Outlier')
        ax.axhline(ub, color=palette['warn'], linestyle='--', linewidth=1, alpha=0.8)
        ax.axhline(lb, color=palette['success'], linestyle='--', linewidth=1, alpha=0.8)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel('Row Index', fontsize=8)
        ax.grid(linestyle='--', alpha=0.3)
        ax.tick_params(labelsize=8)
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc='upper right', fontsize=9,
                   bbox_to_anchor=(0.99, 0.985), ncol=2, framealpha=0.9)
    finish_grid(fig, axes, len(cols), os.path.join(PLOTS_DIR, f'04_scatter_plots_page{p}.png'), n_rows)

# =============================================================================
# METHOD 5: Q-Q PLOTS — Normality & Tail Deviation Check
# =============================================================================
print("\n[5/6] Generating Q-Q Plots...")

for p, cols in enumerate(pages, start=1):
    fig, axes, n_rows = new_grid(len(cols), f'METHOD 5: Q-Q Plots — Tail Deviation from Normal Distribution '
                                             f'(page {p}/{len(pages)})')
    legend_handles = None
    for i, col in enumerate(cols):
        ax = axes[i]
        data = df_num[col].dropna()
        sample = data.sample(min(len(data), 5000), random_state=42)
        (osm, osr), (slope, intercept, r) = stats.probplot(sample, dist='norm')
        ax.plot(osm, osr, 'o', color=palette['highlight'], markersize=2, alpha=0.4, label='Data')
        ax.plot(osm, slope * np.array(osm) + intercept, color=palette['accent'], linewidth=2,
                label='Normal line')
        ax.set_title(col, fontsize=10)
        ax.set_xlabel(f'R² = {r**2:.3f}', fontsize=8.5, color=palette['primary'])
        ax.grid(linestyle='--', alpha=0.4)
        ax.tick_params(labelsize=8)
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc='upper right', fontsize=9,
                   bbox_to_anchor=(0.99, 0.985), ncol=2, framealpha=0.9)
    finish_grid(fig, axes, len(cols), os.path.join(PLOTS_DIR, f'05_qq_plots_page{p}.png'), n_rows)

# =============================================================================
# METHOD 6: HEATMAP — Correlation-based Anomaly Overview
# =============================================================================
print("\n[6/6] Generating Correlation Heatmap...")

n_feat = len(numeric_cols)
fig, ax = plt.subplots(figsize=heatmap_figsize(n_feat))
fig.suptitle('METHOD 6: Correlation Heatmap — Feature Relationship Overview',
             fontsize=15, fontweight='bold', color=palette['primary'], y=0.995)

corr = df_num.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(corr, mask=mask, ax=ax, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            annot=True, fmt='.2f', annot_kws={'size': annot_fontsize(n_feat)},
            linewidths=0.5, linecolor='white', cbar_kws={'shrink': 0.8, 'label': 'Pearson r'})

ax.set_title('Correlations across all numeric features (extreme values hint at potential anomalies)',
             fontsize=10.5, pad=10)
ax.tick_params(axis='x', rotation=45, labelsize=8.5)
ax.tick_params(axis='y', rotation=0, labelsize=8.5)
for label in ax.get_xticklabels():
    label.set_ha('right')

fig.tight_layout(rect=[0, 0, 1, 0.95])
save_path = os.path.join(PLOTS_DIR, '06_correlation_heatmap.png')
fig.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"   Saved: {save_path}")

# =============================================================================
# SUMMARY
# =============================================================================
saved_files = sorted(os.listdir(PLOTS_DIR))
print(f"\n{'='*55}")
print(f"ALL PLOTS SAVED  ->  folder: '{PLOTS_DIR}/'")
print(f"{'='*55}")
for f in saved_files:
    print(f"   OK  {f}")
print(f"\nTotal files: {len(saved_files)}")

# Final: statistically confirmed outlier columns (see boxplot/histogram/violin pages):
# benefit_per_order, order_item_profit_ratio, order_item_product_price, product_price, longitude
# Next: Statistical outlier detection methods.
