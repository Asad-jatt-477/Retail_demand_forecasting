# =============================================================================
#  BUILD GROUP-ID MAPPING FOR THE STREAMLIT APP
#  Reads: data/processed/Retail_demand_FS.csv, data/processed/Retail_demand_outliers_TREATED.csv
#  Writes: models/v3/group_id_mapping.json
#
#  WHY THIS SCRIPT EXISTS
#  ─────────────────────────────────────────────────────────────────────────
#  10_model_training.py recovers a numeric "product_group" / "dept_group" id
#  from product_avg_qty / dept_avg_qty via pd.factorize() (see its Step 2) —
#  this is what the trained pipeline's KFoldTargetEncoder actually consumes.
#  The Streamlit app needs to let a human pick a product/department by NAME,
#  then translate that name to the correct group-id. Since product_name /
#  department_name are dropped before Retail_demand_FS.csv is written, this
#  script rebuilds the name -> group-id lookup once, from the last pipeline
#  stage that still has both the names and the (pre-drop) average-quantity
#  values, joined positionally with Retail_demand_FS.csv.
#
#  This mapping is descriptive lookup data only (name -> the same group-id
#  the trained pipeline already encodes internally) — it introduces no new
#  leakage; it does not feed a target-encoded VALUE anywhere.
# =============================================================================

import pandas as pd
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))          # .../src/pipeline
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))            # project root
DATA_DIR = os.path.join(ROOT_DIR, 'data', 'processed')
OUT_PATH = os.path.join(ROOT_DIR, 'models', 'v3', 'group_id_mapping.json')

_fs_cols = pd.read_csv(os.path.join(DATA_DIR, 'Retail_demand_FS.csv'), nrows=1).columns
_use = [c for c in ['product_avg_qty', 'dept_avg_qty'] if c in _fs_cols]
HAS_DEPT = 'dept_avg_qty' in _use
fs = pd.read_csv(os.path.join(DATA_DIR, 'Retail_demand_FS.csv'), usecols=_use)
raw = pd.read_csv(os.path.join(DATA_DIR, 'Retail_demand_outliers_TREATED.csv'),
                   usecols=['product_name', 'department_name'])

assert len(fs) == len(raw), (
    "Row-count mismatch between Retail_demand_FS.csv and "
    "Retail_demand_outliers_TREATED.csv — positional join is unsafe. "
    "Re-check the pipeline stages haven't reordered/dropped rows."
)

df = fs.copy()
df['product_group'] = pd.factorize(df['product_avg_qty'])[0]
if HAS_DEPT:
    df['dept_group'] = pd.factorize(df['dept_avg_qty'])[0]
df['product_name'] = raw['product_name'].values
df['department_name'] = raw['department_name'].values

prod_map = df.drop_duplicates('product_name').set_index('product_name')['product_group'].to_dict()
# empty when the department is not a model feature (Phase 9 dropped it) -> the app hides the field
dept_map = (df.drop_duplicates('department_name').set_index('department_name')['dept_group'].to_dict()
            if HAS_DEPT else {})
prod_map = {k: int(v) for k, v in sorted(prod_map.items())}
dept_map = {k: int(v) for k, v in sorted(dept_map.items())}

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, 'w') as f:
    json.dump({'product_name_to_group': prod_map, 'department_name_to_group': dept_map}, f, indent=2)

print(f"Saved {len(prod_map)} products, {len(dept_map)} departments -> {OUT_PATH}")

# ── Per-product order history for the app (descriptive statistics only) ───────
# Lets the app show, for the chosen product, what actually happened historically
# (how many orders, what share were single-unit) next to the model's prediction.
# Uses the CLEANED data because the outlier-treated file has winsorized prices.
# No model input is derived from this file.
cl = pd.read_csv(os.path.join(DATA_DIR, 'Retail_demand_forecasting_CLEANED.csv'),
                 usecols=['product_name', 'department_name', 'product_price', 'order_item_quantity'])
stats = []
for name, g in cl.groupby('product_name'):
    q = g['order_item_quantity']
    stats.append({
        'name': name,
        'department': g['department_name'].mode().iat[0],
        'group': prod_map.get(name),
        'price': round(float(g['product_price'].median()), 2),
        'orders': int(len(g)),
        'single_share': round(float((q == 1).mean()), 4),
        'always_single': bool(q.max() == 1),
        'qty_share': {str(k): round(float((q == k).mean()), 4) for k in range(1, 6)},
    })
stats.sort(key=lambda r: r['name'])
STATS_PATH = os.path.join(ROOT_DIR, 'models', 'v3', 'product_stats.json')
with open(STATS_PATH, 'w', encoding='utf-8') as f:
    json.dump({'source': 'Retail_demand_forecasting_CLEANED.csv', 'n_orders': int(len(cl)),
               'products': stats}, f, indent=1, ensure_ascii=True)
print(f"Saved per-product order history for {len(stats)} products -> {STATS_PATH}")
