import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(ROOT_DIR, 'data', 'processed')

# =============================================================================
# OUTPUT FOLDER SETUP
# =============================================================================
PLOTS_DIR = os.path.join(ROOT_DIR, 'assets', 'plots', 'outlier_statistical')
os.makedirs(PLOTS_DIR, exist_ok=True)
print(f"Plots folder ready: '{PLOTS_DIR}/'")

df = pd.read_csv(os.path.join(DATA_DIR, 'Retail_demand_forecasting_CLEANED.csv'))
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

df_imputed = df[numeric_cols].copy()
for col in numeric_cols:
    df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())

# =============================================================================
# PLOTTING HELPERS — keep every figure a "normal", legible, non-overlapping size
# =============================================================================
PER_PAGE = 9            # 3 cols x 3 rows per page
GRID_COLS = 3


def paginate(items, per_page=PER_PAGE):
    return [items[i:i + per_page] for i in range(0, len(items), per_page)]


def new_grid(n_items, title, row_height=3.8):
    n_rows = int(np.ceil(n_items / GRID_COLS))
    fig, axes = plt.subplots(n_rows, GRID_COLS, figsize=(15.5, max(row_height * n_rows, 4)))
    axes = np.atleast_1d(axes).flatten()
    fig.suptitle(title, fontsize=14.5, fontweight='bold', y=0.995)
    return fig, axes, n_rows


def finish_grid(fig, axes, used, save_path, n_rows, legend_handles=None, legend_labels=None, ncol=3):
    for j in range(used, len(axes)):
        axes[j].set_visible(False)
    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc='upper right', fontsize=9,
                   bbox_to_anchor=(0.995, 0.985), ncol=ncol, framealpha=0.9)
    top = 0.92 if n_rows <= 2 else (0.94 if n_rows <= 3 else 0.96)
    fig.tight_layout(rect=[0, 0.02, 1, top])
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved: {save_path}")


pages = paginate(numeric_cols)
print(f"\n{len(numeric_cols)} numeric columns -> {len(pages)} page(s) per method, "
      f"so every chart stays a readable, normal size.")

palette = {
    'primary': '#2C3E50', 'accent': '#E74C3C', 'highlight': '#3498DB',
    'warn': '#F39C12', 'success': '#27AE60', 'purple': '#8E44AD',
    'teal': '#16A085', 'normal_blue': '#2980B9', 'neon_orange': '#FF5722',
    'bg': '#FAFAFA', 'grid': '#ECEFF1',
}
plt.rcParams.update({
    'figure.facecolor': palette['bg'], 'axes.facecolor': palette['bg'],
    'axes.edgecolor': palette['primary'], 'grid.color': palette['grid'],
    'font.family': 'DejaVu Sans', 'axes.titleweight': 'bold'
})

# =============================================================================
# METHOD 1: Standard Z-Score Method
# =============================================================================
print("\n[1/5] Generating Z-Score plots...")
zscore_summary = []

for p, cols in enumerate(pages, start=1):
    fig, axes, n_rows = new_grid(len(cols), f'METHOD 1: Parametric Standard Z-Score Outlier Detection '
                                             f'(|Z| > 3) — page {p}/{len(pages)}')
    legend_handles = legend_labels = None
    for i, col in enumerate(cols):
        ax = axes[i]
        data = df_imputed[col].values
        if data.std() == 0:
            ax.text(0.5, 0.5, "Zero Variance / Constant", ha='center', va='center', color='gray')
            ax.set_title(col, fontsize=10)
            zscore_summary.append({'Column': col, 'Outlier Count': 0, 'Outlier %': '0.00%'})
            continue
        z_scores = np.abs(stats.zscore(data))
        is_outlier = z_scores > 3
        out_count = int(is_outlier.sum())
        out_pct = (out_count / len(data)) * 100
        zscore_summary.append({'Column': col, 'Outlier Count': out_count, 'Outlier %': f"{out_pct:.2f}%"})

        ax.scatter(np.where(~is_outlier)[0], data[~is_outlier], s=1, alpha=0.3,
                   color=palette['highlight'], label='Normal')
        ax.scatter(np.where(is_outlier)[0], data[is_outlier], s=8, alpha=0.8,
                   color=palette['accent'], label='Outlier')
        mean_val, std_val = data.mean(), data.std()
        ax.axhline(mean_val + 3 * std_val, color=palette['warn'], linestyle='--', linewidth=1.2,
                   label='+3σ Boundary')
        ax.axhline(mean_val - 3 * std_val, color=palette['success'], linestyle='--', linewidth=1.2,
                   label='-3σ Boundary')

        ax.set_title(col, fontsize=10)
        ax.set_xlabel(f"Outliers: {out_count:,} ({out_pct:.2f}%)", fontsize=8.5,
                      color=palette['accent'] if out_count > 0 else palette['primary'])
        ax.grid(linestyle='--', alpha=0.4)
        ax.tick_params(labelsize=8)
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    finish_grid(fig, axes, len(cols), os.path.join(PLOTS_DIR, f'01_zscore_outliers_page{p}.png'),
                n_rows, legend_handles, legend_labels, ncol=4)

print("\n" + "=" * 50)
print("          Z-SCORE METHOD METRIC REPORT")
print("=" * 50)
print(pd.DataFrame(zscore_summary).to_string(index=False))
print("=" * 50)


# =============================================================================
# METHOD 2: Modified Z-Score (Robust MAD)
# =============================================================================
def compute_modified_zscore(series):
    median = np.median(series)
    mad = np.median(np.abs(series - median))
    if mad == 0:
        mad = np.mean(np.abs(series - median))
    mod_z = 0.6745 * (series - median) / (mad + 1e-10)
    return mod_z, median, mad


print("\n[2/5] Generating Modified Z-Score plots...")
mad_summary = []

for p, cols in enumerate(pages, start=1):
    fig, axes, n_rows = new_grid(len(cols), f'METHOD 2: Robust Modified Z-Score Outlier Detection '
                                             f'(|Mod Z| > 3.5) — page {p}/{len(pages)}')
    legend_handles = legend_labels = None
    for i, col in enumerate(cols):
        ax = axes[i]
        data = df_imputed[col].values
        mod_z, median_val, mad_val = compute_modified_zscore(data)
        is_outlier = np.abs(mod_z) > 3.5
        out_count = int(is_outlier.sum())
        out_pct = (out_count / len(data)) * 100
        mad_summary.append({'Column': col, 'Median': round(float(median_val), 2),
                             'MAD': round(float(mad_val), 2), 'Outlier Count': out_count,
                             'Outlier %': f"{out_pct:.2f}%"})

        ax.scatter(np.where(~is_outlier)[0], data[~is_outlier], s=1, alpha=0.3,
                   color=palette['teal'], label='Normal')
        ax.scatter(np.where(is_outlier)[0], data[is_outlier], s=8, alpha=0.8,
                   color=palette['accent'], label='Outlier')
        upper_threshold = median_val + (3.5 * mad_val / 0.6745)
        lower_threshold = median_val - (3.5 * mad_val / 0.6745)
        ax.axhline(upper_threshold, color=palette['purple'], linestyle='-.', linewidth=1.2, label='±3.5 Mod Z')
        ax.axhline(lower_threshold, color=palette['purple'], linestyle='-.', linewidth=1.2)

        ax.set_title(col, fontsize=10)
        ax.set_xlabel(f"Outliers: {out_count:,} ({out_pct:.2f}%)", fontsize=8.5,
                      color=palette['accent'] if out_count > 0 else palette['primary'])
        ax.grid(linestyle='--', alpha=0.4)
        ax.tick_params(labelsize=8)
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    finish_grid(fig, axes, len(cols), os.path.join(PLOTS_DIR, f'02_modified_zscore_outliers_page{p}.png'),
                n_rows, legend_handles, legend_labels, ncol=3)

print("\n" + "=" * 70)
print("          MODIFIED Z-SCORE (ROBUST MAD) METHOD METRIC REPORT")
print("=" * 70)
print(pd.DataFrame(mad_summary).to_string(index=False))
print("=" * 70)

# =============================================================================
# METHOD 3: IQR Method (Interquartile Range) / Tukey's Fences
# =============================================================================
print("\n[3/5] Generating IQR / Tukey's Fences plots...")
iqr_summary = []

for p, cols in enumerate(pages, start=1):
    fig, axes, n_rows = new_grid(len(cols), f"METHOD 3: IQR Method / Tukey's Fences "
                                             f"(1.5x Mild & 3.0x Extreme) — page {p}/{len(pages)}")
    legend_handles = legend_labels = None
    for i, col in enumerate(cols):
        ax = axes[i]
        data = df_imputed[col].values
        Q1, Q3 = np.percentile(data, 25), np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_mild, upper_mild = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        lower_extreme, upper_extreme = Q1 - 3.0 * IQR, Q3 + 3.0 * IQR
        is_mild = ((data < lower_mild) & (data >= lower_extreme)) | ((data > upper_mild) & (data <= upper_extreme))
        is_extreme = (data < lower_extreme) | (data > upper_extreme)
        is_normal = ~(is_mild | is_extreme)
        mild_count, extreme_count = int(is_mild.sum()), int(is_extreme.sum())
        iqr_summary.append({'Column': col, 'IQR': round(float(IQR), 2), 'Mild Outliers': mild_count,
                             'Mild %': f"{(mild_count/len(data))*100:.2f}%", 'Extreme Outliers': extreme_count,
                             'Extreme %': f"{(extreme_count/len(data))*100:.2f}%"})

        ax.scatter(np.where(is_normal)[0], data[is_normal], s=1, alpha=0.2,
                   color=palette['highlight'], label='Normal')
        ax.scatter(np.where(is_mild)[0], data[is_mild], s=5, alpha=0.7,
                   color=palette['warn'], label='Mild Outlier')
        ax.scatter(np.where(is_extreme)[0], data[is_extreme], s=8, alpha=0.9,
                   color=palette['accent'], label='Extreme Outlier')
        ax.axhline(upper_mild, color=palette['warn'], linestyle='--', linewidth=1, label='1.5x IQR Fence')
        ax.axhline(lower_mild, color=palette['warn'], linestyle='--', linewidth=1)
        ax.axhline(upper_extreme, color=palette['accent'], linestyle=':', linewidth=1.2, label='3.0x IQR Fence')
        ax.axhline(lower_extreme, color=palette['accent'], linestyle=':', linewidth=1.2)

        ax.set_title(col, fontsize=10)
        ax.set_xlabel(f"Mild: {mild_count:,} | Ext: {extreme_count:,}", fontsize=8.5, color=palette['primary'])
        ax.grid(linestyle='--', alpha=0.4)
        ax.tick_params(labelsize=8)
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    finish_grid(fig, axes, len(cols), os.path.join(PLOTS_DIR, f'03_iqr_tukey_outliers_page{p}.png'),
                n_rows, legend_handles, legend_labels, ncol=3)

print("\n" + "=" * 80)
print("          IQR / TUKEY'S FENCES METHOD METRIC REPORT")
print("=" * 80)
print(pd.DataFrame(iqr_summary).to_string(index=False))
print("=" * 80)


# =============================================================================
# METHOD 4: Grubbs' Test (Extreme Studentized Deviate)
# =============================================================================
def detect_grubbs_outliers(series, alpha=0.05):
    """Iterative Grubbs' test (numpy-vectorized for speed on large series)."""
    data_points = np.asarray(series, dtype=float).copy()
    indices = np.arange(len(series))
    outlier_indices = []
    while True:
        n = len(data_points)
        if n < 3:
            break
        mean_val = data_points.mean()
        std_val = data_points.std(ddof=1)
        if std_val == 0:
            break
        abs_deviations = np.abs(data_points - mean_val)
        max_idx = int(np.argmax(abs_deviations))
        g_stat = abs_deviations[max_idx] / std_val
        t_crit = stats.t.ppf(1 - alpha / (2 * n), df=n - 2)
        g_critical = ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit ** 2 / (n - 2 + t_crit ** 2))
        if g_stat > g_critical:
            outlier_indices.append(int(indices[max_idx]))
            data_points = np.delete(data_points, max_idx)
            indices = np.delete(indices, max_idx)
        else:
            break
    return outlier_indices


print("\n[4/5] Generating Grubbs' Test plots...")
grubbs_summary = []

for p, cols in enumerate(pages, start=1):
    fig, axes, n_rows = new_grid(len(cols), f"METHOD 4: Grubbs' Test (Extreme Studentized Deviate, "
                                             f"alpha=0.05) — page {p}/{len(pages)}")
    legend_handles = legend_labels = None
    for i, col in enumerate(cols):
        ax = axes[i]
        data = df_imputed[col].values
        out_indices = detect_grubbs_outliers(data, alpha=0.05)
        is_outlier = np.zeros(len(data), dtype=bool)
        if out_indices:
            is_outlier[out_indices] = True
        out_count = len(out_indices)
        out_pct = (out_count / len(data)) * 100
        grubbs_summary.append({'Column': col, 'Grubbs Outliers': out_count, 'Outlier %': f"{out_pct:.4f}%",
                                'Max Value': f"{np.max(data):.2f}", 'Min Value': f"{np.min(data):.2f}"})

        ax.scatter(np.where(~is_outlier)[0], data[~is_outlier], s=1, alpha=0.3,
                   color=palette['normal_blue'], label='Normal')
        if out_count > 0:
            ax.scatter(np.where(is_outlier)[0], data[is_outlier], s=15, alpha=1.0,
                       color=palette['accent'], label='Grubbs Outlier')

        ax.set_title(col, fontsize=10)
        ax.set_xlabel(f"Outliers: {out_count:,}", fontsize=8.5,
                      color=palette['accent'] if out_count > 0 else palette['primary'])
        ax.grid(linestyle='--', alpha=0.4)
        ax.tick_params(labelsize=8)
        if legend_handles is None or len(legend_handles) < 2:
            h, l = ax.get_legend_handles_labels()
            if len(h) >= len(legend_handles or []):
                legend_handles, legend_labels = h, l

    finish_grid(fig, axes, len(cols), os.path.join(PLOTS_DIR, f'04_grubbs_test_outliers_page{p}.png'),
                n_rows, legend_handles, legend_labels, ncol=2)

print("\n" + "=" * 80)
print(" GRUBBS' TEST (EXTREME STUDENTIZED DEVIATE) METRIC REPORT")
print("=" * 80)
print(pd.DataFrame(grubbs_summary).to_string(index=False))
print("=" * 80)


# =============================================================================
# METHOD 5: Generalized ESD (Extreme Studentized Deviate) Test
# =============================================================================
def generalized_esd(series, max_outliers=None, alpha=0.05):
    """Numpy-vectorized Generalized ESD (avoids slow Python-list pop on large series)."""
    x = np.asarray(series, dtype=float).copy()
    n = len(x)
    if max_outliers is None:
        max_outliers = max(1, int(n * 0.01))
    max_outliers = min(max_outliers, 500)
    R_stats, L_crit, removed_indices = [], [], []
    remaining_indices = np.arange(n)
    for r in range(1, max_outliers + 1):
        curr_n = len(x)
        if curr_n < 3:
            break
        mean_val = x.mean()
        std_val = x.std(ddof=1)
        if std_val == 0:
            break
        abs_dev = np.abs(x - mean_val)
        max_idx = int(np.argmax(abs_dev))
        R_stats.append(abs_dev[max_idx] / std_val)
        removed_indices.append(int(remaining_indices[max_idx]))
        p = 1 - alpha / (2 * curr_n)
        t_crit = stats.t.ppf(p, df=curr_n - 2)
        lambda_val = ((curr_n - 1) * t_crit) / np.sqrt(curr_n * (curr_n - 2 + t_crit ** 2))
        L_crit.append(lambda_val)
        x = np.delete(x, max_idx)
        remaining_indices = np.delete(remaining_indices, max_idx)
    num_outliers = 0
    for k in range(len(R_stats) - 1, -1, -1):
        if R_stats[k] > L_crit[k]:
            num_outliers = k + 1
            break
    return removed_indices[:num_outliers]


print("\n[5/5] Generating Generalized ESD plots...")
gesd_summary = []

for p, cols in enumerate(pages, start=1):
    fig, axes, n_rows = new_grid(len(cols), f"METHOD 5: Generalized ESD Test "
                                             f"(Multiple Extrema Sequential Detection) — page {p}/{len(pages)}")
    legend_handles = legend_labels = None
    for i, col in enumerate(cols):
        ax = axes[i]
        data = df_imputed[col].values
        out_indices = generalized_esd(data, max_outliers=100, alpha=0.05)
        is_outlier = np.zeros(len(data), dtype=bool)
        if out_indices:
            is_outlier[out_indices] = True
        out_count = len(out_indices)
        out_pct = (out_count / len(data)) * 100
        gesd_summary.append({'Column': col, 'GESD Outliers': out_count, 'Outlier %': f"{out_pct:.2f}%",
                              'Mean': f"{data.mean():.2f}", 'Std Dev': f"{data.std():.2f}"})

        ax.scatter(np.where(~is_outlier)[0], data[~is_outlier], s=1, alpha=0.3,
                   color=palette['normal_blue'], label='Normal')
        if out_count > 0:
            ax.scatter(np.where(is_outlier)[0], data[is_outlier], s=12, alpha=1.0,
                       color=palette['neon_orange'], label='GESD Outlier')

        ax.set_title(col, fontsize=10)
        ax.set_xlabel(f"Outliers: {out_count:,}", fontsize=8.5,
                      color=palette['neon_orange'] if out_count > 0 else palette['primary'])
        ax.grid(linestyle='--', alpha=0.4)
        ax.tick_params(labelsize=8)
        if legend_handles is None or len(legend_handles) < 2:
            h, l = ax.get_legend_handles_labels()
            if len(h) >= len(legend_handles or []):
                legend_handles, legend_labels = h, l

    finish_grid(fig, axes, len(cols), os.path.join(PLOTS_DIR, f'05_gesd_outliers_page{p}.png'),
                n_rows, legend_handles, legend_labels, ncol=2)

print("\n" + "=" * 80)
print("          GENERALIZED ESD (GESD) METHOD METRIC REPORT")
print("=" * 80)
print(pd.DataFrame(gesd_summary).to_string(index=False))
print("=" * 80)

# =============================================================================
# FINAL SUMMARY
# =============================================================================
saved_files = sorted(os.listdir(PLOTS_DIR))
print(f"\n{'='*55}")
print(f"ALL PLOTS SAVED  ->  folder: '{PLOTS_DIR}/'")
print(f"{'='*55}")
for f in saved_files:
    print(f"   OK  {f}")
print(f"\nTotal files: {len(saved_files)}")
