# =============================================================================
#  RETAIL DEMAND FORECASTING — FEATURE ENGINEERING
#  Phase 8: Complete Feature Engineering Pipeline
#  Input  : Retail_demand_YJ_transformed.csv   (178,396 rows × 51 columns)
#  Output : Retail_demand_FE.csv               (178,396 rows × N columns)
#  Target : order_item_quantity  (1, 2, 3, 4, 5)
# =============================================================================
#
#  WHAT IS FEATURE ENGINEERING?
#  ─────────────────────────────────────────────────────────────────────────────
#  Feature Engineering means transforming raw data columns into new, smarter
#  signals that an ML model can learn from more effectively. Raw data often
#  contains the "answer" hidden inside it — Feature Engineering is the process
#  of making that hidden answer visible to the model.
#
#  Example: The column 'product_price' contains 200+ unique values. The model
#  has to figure out on its own that price > $100 always means qty = 1. If we
#  create a binary column 'is_high_price', we hand this pattern directly to
#  the model — it no longer has to discover it from scratch.
#
#  WHAT WE CONFIRMED IN EDA (the foundations of this file):
#  ─────────────────────────────────────────────────────────────────────────────
#  1. Price > $100  → ALWAYS qty = 1 (100% separation)
#  2. Category and Product name → strongest categorical predictors (Cramér's V)
#  3. Most features are NON-NORMAL → tree-based models preferred
#  4. product_price and order_item_product_price are perfect duplicates
#  5. sales, order_item_total, sales_per_customer are almost identical (r=0.99)
#  6. Discount has zero effect on quantity (ANOVA p = 0.99)
#  7. product_status is a zero-variance column (all values = 0)
#  8. Temporal patterns exist (Q4 demand dips, Q1 is highest)
#
#  FEATURE ENGINEERING PLAN (9 Categories):
#  ─────────────────────────────────────────────────────────────────────────────
#  GROUP A : Price-Based Features           (3 features)
#  GROUP B : Target Encoding Features       (3 features)
#  GROUP C : Category Behavior Flags        (2 features)
#  GROUP D : Delivery & Operations Features (3 features)
#  GROUP E : Payment & Order Features       (3 features)
#  GROUP F : Temporal / Calendar Features   (4 features)
#  GROUP G : Discount Engineering           (2 features)
#  GROUP H : Revenue Features (cleaned)     (keep 2, drop 3 duplicates)
#  GROUP I : Column Cleanup                 (drop redundant/zero-variance cols)
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

PLOTS_DIR = os.path.join(ROOT_DIR, 'assets', 'plots', 'feature_engineering')
os.makedirs(PLOTS_DIR, exist_ok=True)
print(f"📁 Output folder ready: '{PLOTS_DIR}/'")


# =============================================================================
# STEP 1: LOAD DATASET & BASELINE CHECK
# =============================================================================

print("\n" + "="*65)
print("STEP 1: LOAD DATASET & BASELINE CHECK")
print("="*65)

df = pd.read_csv(os.path.join(DATA_DIR, 'Retail_demand_YJ_transformed.csv'), encoding='latin-1')
TARGET = 'order_item_quantity'

print(f"\n  Input shape  : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"  Missing vals : {df.isnull().sum().sum()}")
print(f"  Target       : {TARGET}  |  Classes: {sorted(df[TARGET].unique())}")
print(f"\n  Target distribution:")
vc = df[TARGET].value_counts().sort_index()
for qty, cnt in vc.items():
    print(f"    Qty {qty}  →  {cnt:,}  ({cnt/len(df)*100:.1f}%)")

# Track newly created features
FEATURES_CREATED = []


# =============================================================================
# GROUP A: PRICE-BASED FEATURES
# =============================================================================
# WHY THIS GROUP?
# EDA confirmed: product_price > $100 produces qty = 1 in 100% of cases.
# Pattern: Aapko pata chala ke jab bhi kisi product ki keemat 100 dollar se zyada hoti hai, to 100% log uski sirf 1 hi quantity (qty = 1) kharedte hain
# This is the single strongest signal in the entire dataset.
# A raw numeric price gives the model 200+ values to interpret.
# A price tier gives it 3 clean buckets — much more learnable.
#
# DECISION ON DUPLICATES:
# 'product_price' and 'order_item_product_price' are perfect duplicates (r=1.0).
# We will use ONE for feature creation and drop the other in Group I (cleanup).
# We also drop their log versions after creating smarter features from them.
# =============================================================================

print("\n" + "="*65)
print("GROUP A: PRICE-BASED FEATURES")
print("="*65)



# ── A1: is_high_price ────────────────────────────────────────────────────────
# WHY: EDA showed product_price > $100 → qty = 1 in 100% of cases. This is
# a perfect binary separator. The model needs this as a 0/1 flag, not as
# part of a continuous price scale. Correlation with target: -0.67.

# Binary Separator ka matlab hai ek aisi boundary jo data ko do saaf hisson mein baant de (Yes ya No).
# Yahan $100 ek aisi boundary ban gaya hai jo data ko do hisson mein alag kar raha hai:
# Keemat $100 sey kam : Quantity kush b ho sakti hai (1,2,3,4,5)
# Keemat $100 sey zayada : Quantity hamesha (1) hi ho gi 

# ML model ki understanding k liay ap new columns yahan bana rahe ho : 'is_high_price'
# ─────────────────────────────────────────────────────────────────────────────
df['is_high_price'] = (df['product_price'] > 100).astype(int)
FEATURES_CREATED.append('is_high_price')
print(f"\n  [A1] is_high_price created")
print(f"       Value counts: {df['is_high_price'].value_counts().to_dict()}")
print(f"       Corr with target: {df['is_high_price'].corr(df[TARGET]):.4f}")

high_qty1 = (df[df['product_price'] > 100][TARGET] == 1).mean() * 100
low_qty1  = (df[df['product_price'] <= 100][TARGET] == 1).mean() * 100
print(f"       Price > $100  → qty=1 : {high_qty1:.1f}%")
print(f"       Price ≤ $100  → qty=1 : {low_qty1:.1f}%")

# ── A2: price_tier ───────────────────────────────────────────────────────────
# WHY: Three natural price zones exist in this retail dataset:
# (Logon ka buying behavior) teen alag tarha ka : 
#   Low  (≤ $50)  : Accessories, golf balls, small gear → bulk buying possible
#   Mid  ($50-100): Mixed behavior — some bulk, some single
#   High (> $100) : Electronics, cameras, equipment → always single unit
# pd.cut gives the model an ordinal 3-class bucket instead of 200+ raw values.
# ─────────────────────────────────────────────────────────────────────────────
df['price_tier'] = pd.cut(
    df['product_price'],
    bins=[0, 50, 100, float('inf')],
    labels=['Low', 'Mid', 'High'],
    include_lowest=True
)
print(f"\n  [A2] price_tier created")
print(f"       Distribution: {df['price_tier'].value_counts().to_dict()}")
print(f"       Avg qty per tier:")
print(df.groupby('price_tier', observed=True)[TARGET].mean().round(3).to_string())

# ── A3: price_tier_encoded ───────────────────────────────────────────────────
# WHY: ML models cannot use string labels directly. We convert the ordered
# category to integers (1=Low, 2=Mid, 3=High) — ordinal encoding that
# preserves the natural order (low price → high bulk potential).
# ─────────────────────────────────────────────────────────────────────────────
price_tier_map = {'Low': 1, 'Mid': 2, 'High': 3}
df['price_tier_encoded'] = df['price_tier'].map(price_tier_map)
FEATURES_CREATED.append('price_tier_encoded')
print(f"\n  [A3] price_tier_encoded created  (1=Low, 2=Mid, 3=High)")
print(f"       Corr with target: {df['price_tier_encoded'].corr(df[TARGET]):.4f}")

print(f"\n  ✓ GROUP A complete — {3} features created")


# =============================================================================
# GROUP B: TARGET ENCODING FEATURES
# =============================================================================
# WHY THIS GROUP?
# Chi-Square analysis showed:
#   product_name  → Cramér's V = 0.3979 (STRONGEST categorical predictor)
#   category_name → significant association
#   department_name → significant association
#
# These are HIGH-CARDINALITY categorical columns :
# product_name = 476 unique  ,  category_name = 39 unique  ,  department_name = 6 unique
# One-Hot Encoding them would create 100s of sparse columns → bad for ML.
#
# BETTER APPROACH: Target Encoding
# Replace each category with its HISTORICAL AVERAGE quantity.
# Example: 'Golf Balls' → avg_qty = 2.97, 'Computers' → avg_qty = 1.0
# Now the model has a single meaningful numeric value for each product.
#
# IMPORTANT: In real production (train/test split), target encoding must be
# done on TRAINING data only to prevent data leakage. For this exploratory
# FE phase, we compute it on the full dataset. During modeling, apply
# cross-validation-based target encoding.

# =============================================================================

print("\n" + "="*65)
print("GROUP B: TARGET ENCODING FEATURES")
print("="*65)

# ── B1: product_avg_qty ──────────────────────────────────────────────────────
# WHY: Each product has a consistent, characteristic average quantity.
# Golf balls → always bulk (avg 2.97). Cameras → always single (avg 1.0).
# This encodes 118 product names into ONE highly predictive numeric column.
# Correlation with target: +0.68 (strongest single feature in the dataset).
# ─────────────────────────────────────────────────────────────────────────────
product_avg = df.groupby('product_name')[TARGET].mean()
df['product_avg_qty'] = df['product_name'].map(product_avg)
FEATURES_CREATED.append('product_avg_qty')
print(f"\n  [B1] product_avg_qty created  (target encoding: product_name)")
print(f"       Range: {df['product_avg_qty'].min():.3f} – {df['product_avg_qty'].max():.3f}")
print(f"       Corr with target: {df['product_avg_qty'].corr(df[TARGET]):.4f}")

# ── B2: category_avg_qty ─────────────────────────────────────────────────────
# WHY: Category is a higher-level grouping than product (50 unique values).
# It captures the class of demand behavior: single-unit tech categories vs
# multi-unit sports/apparel categories. Correlation with target: +0.67.
# ─────────────────────────────────────────────────────────────────────────────
category_avg = df.groupby('category_name')[TARGET].mean()
df['category_avg_qty'] = df['category_name'].map(category_avg)
FEATURES_CREATED.append('category_avg_qty')
print(f"\n  [B2] category_avg_qty created  (target encoding: category_name)")
print(f"       Range: {df['category_avg_qty'].min():.3f} – {df['category_avg_qty'].max():.3f}")
print(f"       Corr with target: {df['category_avg_qty'].corr(df[TARGET]):.4f}")

# ── B3: dept_avg_qty ─────────────────────────────────────────────────────────
# WHY: Department is an even higher-level grouping (11 unique values):
#   Technology / Book Shop / Pet Shop → avg_qty = 1.0 (single-unit depts)
#   Golf / Footwear / Outdoors / Fitness → avg_qty = 2.65–2.99 (bulk depts)
# This gives the model a department-level demand baseline. Corr: +0.41.
# ─────────────────────────────────────────────────────────────────────────────
dept_avg = df.groupby('department_name')[TARGET].mean()
df['dept_avg_qty'] = df['department_name'].map(dept_avg)
FEATURES_CREATED.append('dept_avg_qty')
print(f"\n  [B3] dept_avg_qty created  (target encoding: department_name)")
print(f"       Department encodings:")
for dept, avg in dept_avg.sort_values().items():
    print(f"         {dept:<25} → {avg:.3f}")
print(f"       Corr with target: {df['dept_avg_qty'].corr(df[TARGET]):.4f}")

print(f"\n  ✓ GROUP B complete — {3} features created")

# Q: Kye ye Data leakage create kar saktey hein ?
# Yes , lakin hum Modeling phase mai ja kar es problem ko fix kar lein gein 


# =============================================================================
# GROUP C: CATEGORY BEHAVIOR FLAGS
# =============================================================================
# Q: Kye ye Data leakage create kar saktey hein ?
# Group C khud leakage nahi karta — lekin Group B ki leakage inherit karta hai.
# q k Yeh dono features 'category_avg_qty' se bane hain — jo Group B mein poore df par calculate hua tha.

# WHY THIS GROUP?
# EDA revealed a clean BINARY split in demand behavior:
#   SINGLE-UNIT categories (avg qty = 1.0): 24 categories including
#     Electronics, Computers, Books, Cameras, Baby, Pet Supplies, etc.
#   BULK categories (avg qty ≥ 2.5): 20 categories including
#     Cleats, Golf, Fitness, Cardio Equipment, Tennis, etc.
#
# These binary flags turn the continuous category_avg_qty signal into
# direct, interpretable class indicators for the model.
# =============================================================================

print("\n" + "="*65)
print("GROUP C: CATEGORY BEHAVIOR FLAGS")
print("="*65)

# ── C1: is_bulk_category ─────────────────────────────────────────────────────
# WHY: Avg qty ≥ 2.5 threshold was chosen because it clearly separates
# the two natural clusters in category demand data. Everything from Golf
# (2.76) to "As Seen on TV" (3.03) falls here. Bulk categories = multi-unit
# demand baseline. This is a powerful signal for the model.
# Correlation with target: +0.67 (almost as strong as category_avg_qty).
# ─────────────────────────────────────────────────────────────────────────────
df['is_bulk_category'] = (df['category_avg_qty'] >= 2.5).astype(int)
FEATURES_CREATED.append('is_bulk_category')
bulk_cats = category_avg[category_avg >= 2.5].index.tolist()
single_cats = category_avg[category_avg < 2.5].index.tolist()
print(f"\n  [C1] is_bulk_category created  (category_avg_qty ≥ 2.5 → 1)")
print(f"       Bulk categories    ({len(bulk_cats)})  : {bulk_cats[:5]} ...")
print(f"       Non-bulk categories ({len(single_cats)}) : {single_cats[:5]} ...")
print(f"       Value counts: {df['is_bulk_category'].value_counts().to_dict()}")
print(f"       Corr with target: {df['is_bulk_category'].corr(df[TARGET]):.4f}")

# ── C2: is_single_unit_category ──────────────────────────────────────────────
# WHY: These categories ALWAYS produce qty = 1. No exceptions in the data.
# Having a dedicated flag for this is more direct than relying on
# is_bulk_category (which is the inverse but less specific). When this = 1,
# the model can be nearly certain the output will be qty = 1.
# ─────────────────────────────────────────────────────────────────────────────
df['is_single_unit_category'] = (df['category_avg_qty'] == 1.0).astype(int)
FEATURES_CREATED.append('is_single_unit_category')
single_unit_cats = category_avg[category_avg == 1.0].index.tolist()
print(f"\n  [C2] is_single_unit_category created  (category_avg_qty = 1.0 → 1)")
print(f"       Single-unit categories ({len(single_unit_cats)}): {single_unit_cats}")
print(f"       Value counts: {df['is_single_unit_category'].value_counts().to_dict()}")
print(f"       Corr with target: {df['is_single_unit_category'].corr(df[TARGET]):.4f}")

print(f"\n  ✓ GROUP C complete — {2} features created")


# =============================================================================
# GROUP D: DELIVERY & OPERATIONS FEATURES
# =============================================================================
# WHY THIS GROUP?
# EDA showed delivery_status is associated with future demand patterns.
# While delivery days don't directly predict quantity of THIS order (order
# is placed before delivery), delivery performance by region is a LAGGED
# signal — regions with high late delivery rates tend to have suppressed
# demand in subsequent periods.
#
# Three operational features give the model:
#   1. How late was this order compared to what was promised?
#   2. How reliable is the delivery in this customer's region?
#   3. Was this a late delivery situation?
# =============================================================================

print("\n" + "="*65)
print("GROUP D: DELIVERY & OPERATIONS FEATURES")
print("="*65)

# ── D1: delivery_delay_days ──────────────────────────────────────────────────
# WHY: The raw difference between actual and scheduled shipping days captures
# the operational reliability signal. Positive = late, Negative = early,
# Zero = on time. EDA showed this ranges from -2 to +4 days.
# This is a cleaner representation than using two separate columns.
# ─────────────────────────────────────────────────────────────────────────────
df['delivery_delay_days'] = df['days_for_shipping_real'] - df['days_for_shipment_scheduled']
FEATURES_CREATED.append('delivery_delay_days')
print(f"\n  [D1] delivery_delay_days created  (actual − scheduled shipping days)")
print(f"       Range    : {df['delivery_delay_days'].min()} to {df['delivery_delay_days'].max()} days")
print(f"       Mean     : {df['delivery_delay_days'].mean():.3f} days (positive = avg delay)")
print(f"       Corr     : {df['delivery_delay_days'].corr(df[TARGET]):.4f}")

# ── D2: region_late_rate ─────────────────────────────────────────────────────
# WHY: Each order region has a characteristic late delivery rate. This is a
# proxy for regional logistics reliability. Canada has the best rate (48.8%
# late), Central Africa has the worst (57.9% late). This regional-level
# signal captures the geographic dimension of operational performance.
# We compute it as: for each region, what % of historical orders are late?
# ─────────────────────────────────────────────────────────────────────────────
region_late_rate = df.groupby('order_region')['delivery_status'].transform(
    lambda x: (x == 'Late delivery').mean()
)
df['region_late_rate'] = region_late_rate.round(4)
FEATURES_CREATED.append('region_late_rate')
print(f"\n  [D2] region_late_rate created  (% late deliveries per order_region)")
print(f"       Best region : Canada           → {df[df['order_region']=='Canada']['region_late_rate'].iloc[0]:.4f}")
print(f"       Worst region: Central Africa   → {df[df['order_region']=='Central Africa']['region_late_rate'].iloc[0]:.4f}")
print(f"       Corr with target: {df['region_late_rate'].corr(df[TARGET]):.4f}")

# ── D3: is_late_delivery ─────────────────────────────────────────────────────
# WHY: Binary encoding of delivery_status. The string column cannot be used
# directly. 'Late delivery' is the most common and most impactful outcome.
# This binary flag is a clean, direct representation of that signal.
# ─────────────────────────────────────────────────────────────────────────────
df['is_late_delivery'] = (df['delivery_status'] == 'Late delivery').astype(int)
FEATURES_CREATED.append('is_late_delivery')
print(f"\n  [D3] is_late_delivery created  (1 = Late delivery)")
print(f"       Late orders: {df['is_late_delivery'].sum():,}  ({df['is_late_delivery'].mean()*100:.1f}% of all orders)")
print(f"       Corr with target: {df['is_late_delivery'].corr(df[TARGET]):.4f}")

print(f"\n  ✓ GROUP D complete — {3} features created")


# =============================================================================
# GROUP E: PAYMENT & ORDER STATUS FEATURES
# =============================================================================
# WHY THIS GROUP?
# EDA showed:
#   - 89% orders use digital payment (Debit/Transfer/Payment)
#   - Order status reveals whether the order was ever fulfilled
#   - Cash payment has higher operational risk (COD cancellations)
#
# These features encode the "health" and "intent" of an order.
# =============================================================================

print("\n" + "="*65)
print("GROUP E: PAYMENT & ORDER STATUS FEATURES")
print("="*65)

# ── E1: is_digital_payment ───────────────────────────────────────────────────
# WHY: Digital payment (Debit, Transfer, Online Payment) vs Cash represents
# two fundamentally different customer risk profiles. Cash customers have
# higher cancellation and return rates in retail. Converting payment type
# to a binary flag captures this distinction cleanly.
# ─────────────────────────────────────────────────────────────────────────────
df['is_digital_payment'] = df['type'].isin(['DEBIT', 'TRANSFER', 'PAYMENT']).astype(int)
FEATURES_CREATED.append('is_digital_payment')
print(f"\n  [E1] is_digital_payment created  (1 = digital, 0 = cash)")
print(f"       Digital  : {df['is_digital_payment'].sum():,} ({df['is_digital_payment'].mean()*100:.1f}%)")
print(f"       Cash     : {(df['is_digital_payment']==0).sum():,} ({(df['is_digital_payment']==0).mean()*100:.1f}%)")
print(f"       Corr with target: {df['is_digital_payment'].corr(df[TARGET]):.4f}")

# ── E2: is_fulfilled ─────────────────────────────────────────────────────────
# WHY: Only COMPLETE and CLOSED orders represent successfully fulfilled demand.
# CANCELED, SUSPECTED_FRAUD, and PENDING orders are demand that was NOT
# realized. For demand forecasting, we must distinguish between expressed
# demand (order placed) and fulfilled demand (order completed).
# This binary flag lets the model learn from the order lifecycle stage.
# ─────────────────────────────────────────────────────────────────────────────
df['is_fulfilled'] = df['order_status'].isin(['COMPLETE', 'CLOSED']).astype(int)
FEATURES_CREATED.append('is_fulfilled')
print(f"\n  [E2] is_fulfilled created  (1 = COMPLETE or CLOSED)")
print(f"       Fulfilled   : {df['is_fulfilled'].sum():,} ({df['is_fulfilled'].mean()*100:.1f}%)")
print(f"       Unfulfilled : {(df['is_fulfilled']==0).sum():,} ({(df['is_fulfilled']==0).mean()*100:.1f}%)")
print(f"       Order status breakdown:")
print(df['order_status'].value_counts().to_string())
print(f"       Corr with target: {df['is_fulfilled'].corr(df[TARGET]):.4f}")

# ── E3: shipping_urgency_score ───────────────────────────────────────────────
# WHY: The 4 shipping modes have a natural ordering by speed and cost:
#   Standard Class = 1 (slowest, cheapest)
#   Second Class   = 2
#   First Class    = 3
#   Same Day       = 4 (fastest, most expensive)
# Ordinal encoding preserves this natural hierarchy better than OHE.
# ─────────────────────────────────────────────────────────────────────────────
shipping_map = {'Standard Class': 1, 'Second Class': 2, 'First Class': 3, 'Same Day': 4}
df['shipping_urgency_score'] = df['shipping_mode'].map(shipping_map)
FEATURES_CREATED.append('shipping_urgency_score')
print(f"\n  [E3] shipping_urgency_score created  (1=Standard → 4=Same Day)")
print(f"       Value distribution: {df['shipping_urgency_score'].value_counts().sort_index().to_dict()}")
print(f"       Corr with target: {df['shipping_urgency_score'].corr(df[TARGET]):.4f}")

print(f"\n  ✓ GROUP E complete — {3} features created")


# =============================================================================
# GROUP F: TEMPORAL / CALENDAR FEATURES
# =============================================================================
# WHY THIS GROUP?
# EDA Step 9 revealed important time-based patterns:
#   - Q1 (Jan–Mar) has the HIGHEST avg quantity (2.21) — post-holiday restocking
#   - Q4 (Oct–Dec) has the LOWEST avg quantity (2.01) — demand contraction
#   - Wednesday has highest B2B demand, Weekend is slightly lower
#   - January Monday/Tuesday shows the strongest bulk buying (restocking)
#   - A sharp collapse in late 2017 suggests data truncation
#
# The raw year/month/day numbers exist in the dataset, but their
# business MEANING is not captured. These features extract that meaning.
# =============================================================================

print("\n" + "="*65)
print("GROUP F: TEMPORAL / CALENDAR FEATURES")
print("="*65)

# ── F1: is_q1_restocking ─────────────────────────────────────────────────────
# WHY: Q1 (January–March) is the post-holiday restocking period. Retailers
# replenish depleted inventory in bulk. EDA heatmap confirmed January
# Monday/Tuesday has the strongest bulk buying pattern in the entire dataset.
# Avg qty in Q1 = 2.21 vs Q4 = 2.01 — a meaningful difference.
# ─────────────────────────────────────────────────────────────────────────────
df['is_q1_restocking'] = (df['order_quarter'] == 1).astype(int)
FEATURES_CREATED.append('is_q1_restocking')
print(f"\n  [F1] is_q1_restocking created  (1 = Q1: Jan–Mar)")
q_avg = df.groupby('order_quarter')[TARGET].mean()
print(f"       Avg qty per quarter:")
for q, avg in q_avg.items():
    flag = ' ← highest' if q == 1 else (' ← lowest' if q == 4 else '')
    print(f"         Q{q}: {avg:.4f}{flag}")
print(f"       Corr with target: {df['is_q1_restocking'].corr(df[TARGET]):.4f}")

# ── F2: is_q4_slowdown ───────────────────────────────────────────────────────
# WHY: Q4 shows the lowest average order quantity (2.01). This is likely
# due to individual gift purchases dominating (single-unit high-price items
# for holidays) rather than bulk operational purchasing. Flagging Q4
# separately gives the model a direct signal for this seasonal contraction.
# ─────────────────────────────────────────────────────────────────────────────
df['is_q4_slowdown'] = (df['order_quarter'] == 4).astype(int)
FEATURES_CREATED.append('is_q4_slowdown')
print(f"\n  [F2] is_q4_slowdown created  (1 = Q4: Oct–Dec)")
print(f"       Corr with target: {df['is_q4_slowdown'].corr(df[TARGET]):.4f}")

# ── F3: is_weekend ───────────────────────────────────────────────────────────
# WHY: EDA showed weekday demand (especially Wednesday) is higher than
# weekend demand. This suggests B2B/Corporate buyers dominate weekday
# orders. Weekends = more individual (B2C) consumer behavior = lower qty.
# order_weekday: 0=Monday, 6=Sunday.
# ─────────────────────────────────────────────────────────────────────────────
df['is_weekend'] = df['order_weekday'].isin([5, 6]).astype(int)
FEATURES_CREATED.append('is_weekend')
weekday_avg = df.groupby('order_weekday')[TARGET].mean()
print(f"\n  [F3] is_weekend created  (1 = Saturday or Sunday)")
print(f"       Avg qty by weekday (0=Mon, 6=Sun):")
print(f"       {weekday_avg.round(3).to_dict()}")
print(f"       Corr with target: {df['is_weekend'].corr(df[TARGET]):.4f}")

# ── F4: is_sports_season ─────────────────────────────────────────────────────
# WHY: EDA temporal analysis showed March–April as the peak sports demand
# season — sports clubs, schools, outdoor enthusiasts place bulk orders.
# The dataset is sports/outdoor-retail dominated. This seasonal peak is
# a genuine recurring business pattern worth encoding directly.
# ─────────────────────────────────────────────────────────────────────────────
df['is_sports_season'] = df['order_month'].isin([3, 4]).astype(int)
FEATURES_CREATED.append('is_sports_season')
month_avg = df.groupby('order_month')[TARGET].mean()
print(f"\n  [F4] is_sports_season created  (1 = March or April)")
print(f"       Avg qty by month:")
print(f"       {month_avg.round(3).to_dict()}")
print(f"       Corr with target: {df['is_sports_season'].corr(df[TARGET]):.4f}")

print(f"\n  ✓ GROUP F complete — {4} features created")


# =============================================================================
# GROUP G: DISCOUNT ENGINEERING
# =============================================================================
# WHY THIS GROUP?
# EDA showed order_item_discount is a zero-inflated feature:
#   94.4% of orders have a discount, 5.6% have zero discount.
# The raw dollar amount has skew > 1.5 and weak correlation with target.
#
# But the PRESENCE of a discount and its INTENSITY are more informative
# than the exact dollar value. We convert this messy continuous variable
# into two cleaner, more model-friendly representations.
# =============================================================================

print("\n" + "="*65)
print("GROUP G: DISCOUNT ENGINEERING")
print("="*65)

# ── G1: has_discount ─────────────────────────────────────────────────────────
# WHY: The most important question is not "how much discount?" but
# "was there ANY discount?". Binary flag removes the noise from exact
# dollar amounts and captures the discount presence signal cleanly.
# ─────────────────────────────────────────────────────────────────────────────
df['has_discount'] = (df['order_item_discount'] > 0).astype(int)
FEATURES_CREATED.append('has_discount')
print(f"\n  [G1] has_discount created  (1 = discount > $0)")
print(f"       Has discount   : {df['has_discount'].sum():,} ({df['has_discount'].mean()*100:.1f}%)")
print(f"       Zero discount  : {(df['has_discount']==0).sum():,} ({(df['has_discount']==0).mean()*100:.1f}%)")
print(f"       Corr with target: {df['has_discount'].corr(df[TARGET]):.4f}")

# ── G2: discount_intensity ───────────────────────────────────────────────────
# WHY: The dollar amount of discount ranges from $0 to ~$65. Instead of
# giving the model 984 unique raw values, we bucket it into 4 meaningful
# groups that reflect business discount tiers:
#   0  = No discount applied
#   1  = Small discount ($0.01 – $5)
#   2  = Medium discount ($5.01 – $15)
#   3  = Large discount (> $15)
# This ordinal encoding preserves the discount level ordering.
# ─────────────────────────────────────────────────────────────────────────────
df['discount_intensity'] = pd.cut(
    df['order_item_discount'],
    bins=[-0.01, 0, 5, 15, float('inf')],
    labels=[0, 1, 2, 3]
).astype(int)
FEATURES_CREATED.append('discount_intensity')
print(f"\n  [G2] discount_intensity created  (0=None, 1=Small, 2=Medium, 3=Large)")
print(f"       Distribution: {df['discount_intensity'].value_counts().sort_index().to_dict()}")
print(f"       Avg qty per intensity level:")
print(df.groupby('discount_intensity')[TARGET].mean().round(3).to_string())
print(f"       Corr with target: {df['discount_intensity'].corr(df[TARGET]):.4f}")

print(f"\n  ✓ GROUP G complete — {2} features created")


# =============================================================================
# GROUP H: REVENUE FEATURES — FINAL SELECTION
# =============================================================================
# WHY THIS GROUP?
# EDA Step 13 revealed a severe multicollinearity problem:
#
#   sales, order_item_total, sales_per_customer  → r = 0.99 (near-perfect)
#   product_price, order_item_product_price      → r = 1.00 (exact duplicate)
#   sales_log, order_item_total_log, sales_per_customer_log → r = 0.993
#
# Including all of them in an ML model creates multicollinearity which:
#   - Inflates variance of coefficient estimates
#   - Makes feature importance unreliable
#   - Does NOT improve model accuracy
#
# DECISION: Keep 2 representatives, drop the rest.
#   KEEP: product_price_log     (best price signal, log-normalized)
#   KEEP: sales_per_customer_log (captures per-customer spend level)
#   DROP: order_item_product_price, product_price (raw, duplicate of log)
#   DROP: order_item_total, sales (raw, duplicates of sales_per_customer)
#   DROP: order_item_total_log   (r=0.993 with sales_per_customer_log)
#   DROP: sales_log              (r=0.993 with sales_per_customer_log)
#   DROP: product_status         (zero-variance: all values = 0)
#
# The YJ-transformed columns (benefit_per_order_yj, order_item_profit_ratio_yj)
# from Phase 7 are KEPT — they represent the best versions of those features.
# =============================================================================

print("\n" + "="*65)
print("GROUP H: REVENUE FEATURES — MULTICOLLINEARITY CLEANUP")
print("="*65)

# Columns to DROP (explained above)
COLS_TO_DROP = [
    'order_item_product_price',      # exact duplicate of product_price
    'order_item_total',               # r=0.99 with sales_per_customer
    'sales',                          # r=0.99 with sales_per_customer
    'order_item_total_log',           # r=0.993 with sales_per_customer_log
    'sales_log',                      # r=0.993 with sales_per_customer_log
    'product_status',                 # zero-variance (all = 0)
    'product_image',                  # URL string, not usable by model
]

print(f"\n  Columns to DROP (9 columns):")
for col in COLS_TO_DROP:
    reason_map = {
        'order_item_product_price': 'exact duplicate of product_price',
        'product_price': 'raw version → product_price_log is better',
        'order_item_total': 'r=0.99 with sales_per_customer',
        'sales': 'r=0.99 with sales_per_customer',
        'order_item_total_log': 'r=0.993 with sales_per_customer_log',
        'sales_log': 'r=0.993 with sales_per_customer_log',
        'product_status': 'zero-variance (all values = 0)',
        'product_image': 'URL string → product_name encodes same info',
        'order_item_discount': 'replaced by has_discount + discount_intensity',
    }
    if col in df.columns:
        print(f"    DROP: {col:<40} → {reason_map.get(col,'')}")

df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns], inplace=True)

print(f"\n  Columns KEPT from revenue group:")
kept_rev = ['product_price_log', 'order_item_product_price_log',
            'sales_per_customer_log', 'sales_per_customer',
            'benefit_per_order', 'benefit_per_order_yj',
            'order_item_profit_ratio', 'order_item_profit_ratio_yj',
            'order_item_discount_rate']
for col in kept_rev:
    if col in df.columns:
        corr = df[col].corr(df[TARGET])
        print(f"    KEEP: {col:<40} | corr={corr:.4f}")

print(f"\n  ✓ GROUP H complete — 9 redundant/zero-variance columns removed")


# =============================================================================
# GROUP I: HIGH-CARDINALITY CATEGORICAL COLUMNS — FINAL DECISION
# =============================================================================
# WHY THIS GROUP?
# Several string columns exist that cannot be fed directly to ML models:
#
#   product_name   (476 unique) → already encoded as product_avg_qty in B1
#   category_name  (39 unique)  → already encoded as category_avg_qty in B2
#   department_name(6 unique)  → already encoded as dept_avg_qty in B3
#   customer_city  (563 unique) → too high cardinality, too sparse
#   order_city     (3596 unique)→ extremely high cardinality
#   customer_state (46 unique)  → geographic detail captured by order_region
#   order_state    (1089 unique)→ too high cardinality
#   order_country  (164 unique) → captured by market/order_region
#   customer_country(2 unique)  → nearly all USA → near-zero variance
#   customer_zipcode: numeric but it's a geographic ID, not a quantity
#   product_card_id: product ID number, already encoded via product_name
#
# LOW CARDINALITY (4–9 unique) → these can be encoded:
#   type (4)              → already encoded as is_digital_payment
#   delivery_status (4)   → already encoded as is_late_delivery
#   shipping_mode (4)     → already encoded as shipping_urgency_score

# Keep these columns :
#   customer_segment (3)  → keep for OHE (3 dummies)
#   market (5)            → keep for OHE (5 dummies)
#   order_region (23)     → already captured by region_late_rate; keep raw
#   order_status (9)      → already encoded as is_fulfilled; keep for model
# =============================================================================

print("\n" + "="*65)
print("GROUP I: HIGH-CARDINALITY STRING CLEANUP")
print("="*65)

HIGH_CARD_DROP = [
    'product_name',      # encoded → product_avg_qty
    'category_name',     # encoded → category_avg_qty
    'department_name',   # encoded → dept_avg_qty
    'customer_city',     # 563 unique, too sparse
    'order_city',        # 3596 unique, too sparse
    'customer_state',    # 46 unique, captured by region
    'order_state',       # 1089 unique, too sparse
    'order_country',     # 164 unique, captured by market
    'customer_country',  # 2 unique, near-constant
    'customer_zipcode',  # geographic ID, not predictive quantity
    'product_card_id',   # numeric ID, already encoded via product_name
    'type',              # 4 unique, encoded → is_digital_payment
    'delivery_status',   # 4 unique, encoded → is_late_delivery
    'shipping_mode',     # 4 unique, encoded → shipping_urgency_score
    'latitude',          # raw coordinates, captured by region/market
    'longitude',         # raw coordinates, captured by region/market
    'price_tier',        # string version, encoded → price_tier_encoded
]

print(f"\n  Dropping {len([c for c in HIGH_CARD_DROP if c in df.columns])} high-cardinality / redundant columns:")
for col in HIGH_CARD_DROP:
    if col in df.columns:
        n_unique = df[col].nunique() if df[col].dtype == 'object' else 'numeric'
        print(f"    DROP: {col:<30} | {n_unique} unique → replaced by engineered feature")

df.drop(columns=[c for c in HIGH_CARD_DROP if c in df.columns], inplace=True)

print(f"\n  Remaining categorical columns for OHE / model:")
remaining_cats = df.select_dtypes(include='object').columns.tolist()
# Fix for newer pandas (str dtype)
remaining_cats_all = [c for c in df.columns if df[c].dtype == 'object' or str(df[c].dtype) == 'string']
for c in remaining_cats_all:
    print(f"    KEEP: {c:<30} | {df[c].nunique()} unique values")

print(f"\n  ✓ GROUP I complete — high-cardinality cleanup done")


# =============================================================================
# STEP 2: POST-ENGINEERING DATASET OVERVIEW
# =============================================================================

print("\n" + "="*65)
print("STEP 2: POST-ENGINEERING DATASET OVERVIEW")
print("="*65)

print(f"\n  Output shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\n  New features created ({len(FEATURES_CREATED)}):")
for i, feat in enumerate(FEATURES_CREATED, 1):
    corr = df[feat].corr(df[TARGET])
    print(f"    {i:2}. {feat:<40} | corr={corr:+.4f}")

print(f"\n  All remaining columns ({df.shape[1]}):")
for i, col in enumerate(df.columns, 1):
    dtype = df[col].dtype
    corr  = df[col].corr(df[TARGET]) if dtype in ['int64','float64','int32'] else 'N/A'
    corr_str = f"{corr:+.4f}" if isinstance(corr, float) else corr
    print(f"    {i:2}. {col:<45} [{dtype}]  corr={corr_str}")


# =============================================================================
# STEP 3: FEATURE IMPORTANCE PREVIEW
# Plot: Correlation of all numeric features with target
# =============================================================================

print("\n" + "="*65)
print("STEP 3: FEATURE IMPORTANCE PREVIEW — CORRELATION WITH TARGET")
print("="*65)

numeric_df = df.select_dtypes(include=['int64','float64'])
target_corr = numeric_df.corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)

print(f"\n  Top 20 features by |correlation| with target:")
print(target_corr.head(20).round(4).to_string())

# ── Plot: Feature Correlation Bar Chart ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, max(8, len(target_corr) * 0.28)))
colors = [palette['success'] if v > 0 else palette['accent'] for v in target_corr.values]
bars = ax.barh(target_corr.index, target_corr.values, color=colors,
               edgecolor='white', linewidth=0.5, alpha=0.85)

# Annotate bars
for bar, val in zip(bars, target_corr.values):
    x_pos = val + (0.005 if val >= 0 else -0.005)
    ha = 'left' if val >= 0 else 'right'
    ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:+.3f}',
            va='center', ha=ha, fontsize=7, fontweight='bold')

ax.axvline(0, color=palette['primary'], linewidth=1.2)
ax.set_title(f"Feature Correlation with '{TARGET}'\n"
             f"(All numeric features after Feature Engineering — sorted by |r|)",
             fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel('Pearson Correlation (r)', fontsize=10)
ax.grid(axis='x', linestyle='--', alpha=0.4)
ax.invert_yaxis()
ax.tick_params(labelsize=8.5)
fig.tight_layout()
save_path = os.path.join(PLOTS_DIR, 'fe_01_feature_correlations.png')
fig.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"\n  Saved: {save_path}")


# =============================================================================
# STEP 4: TOP ENGINEERED FEATURES — DETAILED VISUAL ANALYSIS
# =============================================================================

print("\n" + "="*65)
print("STEP 4: TOP ENGINEERED FEATURES — VISUAL ANALYSIS")
print("="*65)

qty_palette = ['#3498DB', '#27AE60', '#F39C12', '#E74C3C', '#8E44AD']
qty_vals    = sorted(df[TARGET].unique())

# ── Plot 4A: Target Encoding Features — Box plots vs Target ──────────────────
te_features = [('product_avg_qty',  'Product Average Qty (Target Encoding)'),
               ('category_avg_qty', 'Category Average Qty (Target Encoding)'),
               ('dept_avg_qty',     'Department Average Qty (Target Encoding)')]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('STEP 4A: Target Encoding Features vs order_item_quantity\n'
             'Each quantity class receives a consistent encoded value → high predictive power',
             fontsize=14, fontweight='bold', color=palette['primary'], y=0.995)

for ax, (feat, title) in zip(axes, te_features):
    groups = [df[df[TARGET] == q][feat].dropna() for q in qty_vals]
    bp = ax.boxplot(groups, patch_artist=True,
                    flierprops=dict(marker='.', markersize=2, alpha=0.3, color=palette['accent']),
                    medianprops=dict(color='white', linewidth=2))
    for patch, color in zip(bp['boxes'], [qty_colors[q] for q in qty_vals]):
        patch.set_facecolor(color); patch.set_alpha(0.75)
    f_stat, p_val = stats.f_oneway(*groups)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(f'ANOVA: F={f_stat:.1f},  p={p_val:.2e}\n{"✓ Significant" if p_val<0.05 else "✗ Not Significant"}',
                  fontsize=8, color=palette['success'] if p_val < 0.05 else palette['accent'])
    ax.set_xticklabels([f'Qty {q}' for q in qty_vals], fontsize=7, rotation=20)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout(rect=[0, 0.01, 1, 0.90])
plt.savefig(os.path.join(PLOTS_DIR, 'fe_02_target_encoding_boxplots.png'), dpi=300, bbox_inches='tight')
plt.close('all')
print(f"  Saved: {PLOTS_DIR}/fe_02_target_encoding_boxplots.png")

# ── Plot 4B: Price Features — is_high_price & price_tier_encoded ──────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('STEP 4B: Price-Based Features vs order_item_quantity\n'
             'is_high_price: Perfect binary separator | price_tier: 3-class separation',
             fontsize=14, fontweight='bold', color=palette['primary'], y=0.995)

# B1: is_high_price counts
ax = axes[0]
cross = pd.crosstab(df['is_high_price'], df[TARGET], normalize='index') * 100
cross.plot(kind='bar', stacked=True, ax=ax, color=qty_palette, edgecolor='white')
ax.set_title('is_high_price: Qty Distribution\n(0=Normal, 1=Premium >$100)')
ax.set_xlabel('is_high_price'); ax.set_ylabel('% of Orders')
ax.set_xticklabels(['Normal Price', 'High Price (>$100)'], rotation=0, fontsize=9)
ax.legend(title='Qty', fontsize=7); ax.grid(axis='y', linestyle='--', alpha=0.4)

# B2: price_tier_encoded
ax2 = axes[1]
tier_qty = df.groupby('price_tier_encoded')[TARGET].value_counts(normalize=True).unstack() * 100
tier_qty.plot(kind='bar', stacked=True, ax=ax2, color=qty_palette, edgecolor='white')
ax2.set_title('price_tier_encoded: Qty Distribution\n(1=Low≤$50, 2=Mid $50-100, 3=High>$100)')
ax2.set_xlabel('Price Tier'); ax2.set_ylabel('% of Orders')
ax2.set_xticklabels(['Low (≤$50)', 'Mid ($50-100)', 'High (>$100)'], rotation=0, fontsize=8)
ax2.legend(title='Qty', fontsize=7); ax2.grid(axis='y', linestyle='--', alpha=0.4)

# B3: product_price_log colored by qty
ax3 = axes[2]
for qty in qty_vals:
    subset = df[df[TARGET] == qty]['product_price_log']
    ax3.hist(subset, bins=40, alpha=0.5, color=qty_colors[qty],
             density=True, label=f'Qty {qty}', edgecolor='white', linewidth=0.2)
ax3.axvline(np.log(100), color=palette['primary'], linewidth=2, linestyle='--',
            label='log($100) threshold')
ax3.set_title('product_price_log Distribution\nby Quantity Class')
ax3.set_xlabel('product_price_log'); ax3.set_ylabel('Density')
ax3.legend(fontsize=7); ax3.grid(linestyle='--', alpha=0.4)

plt.tight_layout(rect=[0, 0.01, 1, 0.90])
plt.savefig(os.path.join(PLOTS_DIR, 'fe_03_price_features.png'), dpi=300, bbox_inches='tight')
plt.close('all')
print(f"  Saved: {PLOTS_DIR}/fe_03_price_features.png")

# ── Plot 4C: Binary Flags Summary — avg qty comparison ───────────────────────
binary_features = [
    ('is_high_price',            'High Price (>$100)'),
    ('is_bulk_category',         'Bulk Category'),
    ('is_single_unit_category',  'Single-Unit Category'),
    ('is_digital_payment',       'Digital Payment'),
    ('is_fulfilled',             'Order Fulfilled'),
    ('is_late_delivery',         'Late Delivery'),
    ('has_discount',             'Has Discount'),
    ('is_q1_restocking',         'Q1 Restocking Period'),
    ('is_q4_slowdown',           'Q4 Slowdown Period'),
    ('is_weekend',               'Weekend Order'),
    ('is_sports_season',         'Sports Season (Mar-Apr)'),
]

fig, axes = plt.subplots(3, 4, figsize=(20, 15))
fig.suptitle('STEP 4C: Binary Engineered Features — Avg Quantity Comparison\n'
             '(Blue = Flag = 0  |  Orange = Flag = 1)',
             fontsize=16, fontweight='bold', color=palette['primary'], y=0.995)
axes = axes.flatten()

for i, (feat, title) in enumerate(binary_features):
    ax = axes[i]
    if feat not in df.columns:
        axes[i].set_visible(False)
        continue
    group_0 = df[df[feat] == 0][TARGET]
    group_1 = df[df[feat] == 1][TARGET]
    avg_0 = group_0.mean()
    avg_1 = group_1.mean()
    n_0   = len(group_0)
    n_1   = len(group_1)

    bars = ax.bar([f'0\n(n={n_0:,})', f'1\n(n={n_1:,})'],
                  [avg_0, avg_1],
                  color=[palette['highlight'], palette['warn']],
                  edgecolor='white', alpha=0.85, width=0.5)
    for bar, val in zip(bars, [avg_0, avg_1]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')

    # T-test for significance
    from scipy.stats import ttest_ind
    t_stat, p_val = ttest_ind(group_0, group_1)
    sig = '✓' if p_val < 0.05 else '✗'
    ax.set_title(f'{title}\n{sig} p={p_val:.4f}', fontsize=9.5, pad=6)
    ax.set_ylabel('Avg Quantity', fontsize=8.5)
    ax.set_ylim(0, max(avg_0, avg_1) * 1.2)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

for j in range(len(binary_features), len(axes)):
    axes[j].set_visible(False)

fig.tight_layout(rect=[0, 0.01, 1, 0.94])
save_path = os.path.join(PLOTS_DIR, 'fe_04_binary_flags_comparison.png')
fig.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {save_path}")

# ── Plot 4D: Temporal Features — Quantity by Quarter & Month ──────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('STEP 4D: Temporal Patterns — Calendar-Based Demand Signals',
             fontsize=14, fontweight='bold', color=palette['primary'], y=0.995)

# Quarter
ax = axes[0]
q_means = df.groupby('order_quarter')[TARGET].mean()
bars = ax.bar(q_means.index, q_means.values,
              color=[palette['success'] if i in [1] else
                     palette['accent'] if i in [4] else
                     palette['highlight'] for i in q_means.index],
              edgecolor='white', alpha=0.85)
for bar, val in zip(bars, q_means.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
ax.set_title('Avg Qty by Quarter\n(Green=Q1 highest, Red=Q4 lowest)')
ax.set_xlabel('Quarter'); ax.set_ylabel('Avg Order Qty')
ax.set_xticks([1,2,3,4]); ax.set_xticklabels(['Q1\n(Jan-Mar)','Q2\n(Apr-Jun)','Q3\n(Jul-Sep)','Q4\n(Oct-Dec)'])
ax.grid(axis='y', linestyle='--', alpha=0.4)

# Month
ax2 = axes[1]
m_means = df.groupby('order_month')[TARGET].mean()
bar_colors = [palette['success'] if m in [3,4] else
              palette['accent'] if m == 12 else
              palette['highlight'] for m in m_means.index]
ax2.bar(m_means.index, m_means.values, color=bar_colors, edgecolor='white', alpha=0.85)
ax2.set_title('Avg Qty by Month\n(Green=Sports Season Mar-Apr)')
ax2.set_xlabel('Month'); ax2.set_ylabel('Avg Order Qty')
ax2.set_xticks(range(1,13))
ax2.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'], fontsize=9)
ax2.grid(axis='y', linestyle='--', alpha=0.4)

# Weekday
ax3 = axes[2]
w_means = df.groupby('order_weekday')[TARGET].mean()
bar_colors3 = [palette['accent'] if w in [5,6] else palette['highlight'] for w in w_means.index]
ax3.bar(w_means.index, w_means.values, color=bar_colors3, edgecolor='white', alpha=0.85)
ax3.set_title('Avg Qty by Weekday\n(Red=Weekend, Blue=Weekday)')
ax3.set_xlabel('Weekday'); ax3.set_ylabel('Avg Order Qty')
ax3.set_xticks(range(7))
ax3.set_xticklabels(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'], fontsize=8)
ax3.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout(rect=[0, 0.01, 1, 0.90])
plt.savefig(os.path.join(PLOTS_DIR, 'fe_05_temporal_patterns.png'), dpi=300, bbox_inches='tight')
plt.close('all')
print(f"  Saved: {PLOTS_DIR}/fe_05_temporal_patterns.png")

# ── Plot 4E: Delivery Features ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('STEP 4E: Delivery & Operations Engineered Features',
             fontsize=14, fontweight='bold', color=palette['primary'], y=0.995)

# E1: delivery_delay_days distribution
ax = axes[0]
ax.hist(df['delivery_delay_days'], bins=range(-3, 6),
        color=palette['highlight'], alpha=0.8, edgecolor='white',
        rwidth=0.7, density=True)
ax.axvline(0, color=palette['accent'], linewidth=2, linestyle='--', label='On Time')
ax.set_title('delivery_delay_days Distribution\n(Negative=Early, 0=OnTime, Positive=Late)')
ax.set_xlabel('Delay (days)'); ax.set_ylabel('Density')
ax.legend(); ax.grid(linestyle='--', alpha=0.4)

# E2: region_late_rate
ax2 = axes[1]
region_late_df = df.groupby('order_region')['region_late_rate'].first().sort_values()
ax2.barh(region_late_df.index, region_late_df.values,
         color=[palette['accent'] if v > 0.57 else palette['highlight']
                for v in region_late_df.values],
         alpha=0.8, edgecolor='white')
ax2.axvline(region_late_df.mean(), color=palette['primary'], linewidth=1.5,
            linestyle='--', label=f'Mean: {region_late_df.mean():.3f}')
ax2.set_title('region_late_rate by Order Region\n(Higher = worse delivery reliability)')
ax2.set_xlabel('Late Delivery Rate'); ax2.legend(fontsize=8)
ax2.grid(axis='x', linestyle='--', alpha=0.4)

# E3: is_late_delivery per qty
ax3 = axes[2]
late_by_qty = df.groupby(TARGET)['is_late_delivery'].mean() * 100
bars = ax3.bar(late_by_qty.index, late_by_qty.values,
               color=[qty_colors[q] for q in late_by_qty.index],
               edgecolor='white', alpha=0.85)
for bar, val in zip(bars, late_by_qty.values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')
ax3.set_title('Late Delivery % by Quantity Class\n(Fairly uniform across classes)')
ax3.set_xlabel('Order Qty'); ax3.set_ylabel('% Late Orders')
ax3.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout(rect=[0, 0.01, 1, 0.90])
plt.savefig(os.path.join(PLOTS_DIR, 'fe_06_delivery_features.png'), dpi=300, bbox_inches='tight')
plt.close('all')
print(f"  Saved: {PLOTS_DIR}/fe_06_delivery_features.png")

# ── Plot 4F: Discount Features ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('STEP 4F: Discount Engineered Features\n'
             '(has_discount + discount_intensity replace raw order_item_discount)',
             fontsize=14, fontweight='bold', color=palette['primary'], y=0.995)

# F1: has_discount by qty
ax = axes[0]
disc_qty = df.groupby(TARGET)['has_discount'].mean() * 100
ax.bar(disc_qty.index, disc_qty.values,
       color=[qty_colors[q] for q in disc_qty.index], edgecolor='white', alpha=0.85)
for i, (qty, val) in enumerate(disc_qty.items()):
    ax.text(qty, val + 0.2, f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')
ax.set_title('% Orders WITH Discount by Qty\n(has_discount = 1)')
ax.set_xlabel('Order Qty'); ax.set_ylabel('% with Discount'); ax.grid(axis='y', linestyle='--', alpha=0.4)

# F2: discount_intensity distribution
ax2 = axes[1]
di_vc = df['discount_intensity'].value_counts().sort_index()
bar_colors2 = [palette['accent'], palette['highlight'], palette['warn'], palette['success']]
bars2 = ax2.bar(di_vc.index, di_vc.values, color=bar_colors2[:len(di_vc)],
                edgecolor='white', alpha=0.85)
for bar, val in zip(bars2, di_vc.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
             f'{val:,}', ha='center', fontsize=9, fontweight='bold')
ax2.set_title('discount_intensity Distribution\n(0=None, 1=Small, 2=Medium, 3=Large)')
ax2.set_xlabel('Discount Intensity'); ax2.set_ylabel('Count')
ax2.set_xticks([0,1,2,3])
ax2.set_xticklabels(['None','Small\n($1-$5)','Medium\n($6-$15)','Large\n(>$15)'], fontsize=8)
ax2.grid(axis='y', linestyle='--', alpha=0.4)

# F3: avg qty per discount intensity level
ax3 = axes[2]
di_qty = df.groupby('discount_intensity')[TARGET].mean()
bars3 = ax3.bar(di_qty.index, di_qty.values,
                color=bar_colors2[:len(di_qty)], edgecolor='white', alpha=0.85)
for bar, val in zip(bars3, di_qty.values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
ax3.set_title('Avg Qty per Discount Intensity\n(Minimal variation confirms weak signal)')
ax3.set_xlabel('Discount Intensity'); ax3.set_ylabel('Avg Order Qty')
ax3.set_xticks([0,1,2,3])
ax3.set_xticklabels(['None','Small','Medium','Large'], fontsize=9)
ax3.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout(rect=[0, 0.01, 1, 0.90])
plt.savefig(os.path.join(PLOTS_DIR, 'fe_07_discount_features.png'), dpi=300, bbox_inches='tight')
plt.close('all')
print(f"  Saved: {PLOTS_DIR}/fe_07_discount_features.png")


# =============================================================================
# STEP 5: CORRELATION HEATMAP — POST FEATURE ENGINEERING
# =============================================================================

print("\n" + "="*65)
print("STEP 5: CORRELATION HEATMAP — POST FEATURE ENGINEERING")
print("="*65)

numeric_cols_fe = df.select_dtypes(include=['int64','float64']).columns.tolist()
corr_fe = df[numeric_cols_fe].corr()

n_feat_fe = len(numeric_cols_fe)
heat_side = max(14, min(24, 0.42 * n_feat_fe + 6))
annot_size = max(4.5, min(8, 210 / max(n_feat_fe, 1)))

fig, ax = plt.subplots(figsize=(heat_side, heat_side * 0.82))
mask = np.triu(np.ones_like(corr_fe, dtype=bool))
sns.heatmap(corr_fe, mask=mask, ax=ax,
            cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            annot=True, fmt='.2f', annot_kws={'size': annot_size},
            linewidths=0.3, linecolor='white',
            cbar_kws={'shrink': 0.7, 'label': 'Pearson r'})
ax.set_title('Correlation Heatmap — Post Feature Engineering\n'
             '(All numeric features including engineered ones)',
             fontsize=14, fontweight='bold', pad=14)
ax.tick_params(axis='x', rotation=45, labelsize=8)
ax.tick_params(axis='y', rotation=0,  labelsize=8)
for label in ax.get_xticklabels():
    label.set_ha('right')
fig.tight_layout(rect=[0, 0, 1, 0.96])
save_path = os.path.join(PLOTS_DIR, 'fe_08_post_fe_correlation_heatmap.png')
fig.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {save_path}")


# =============================================================================
# STEP 6: SAVE ENGINEERED DATASET
# =============================================================================

print("\n" + "="*65)
print("STEP 6: SAVE ENGINEERED DATASET")
print("="*65)

# Final column order: target first, then engineered, then original
output_path = os.path.join(DATA_DIR, 'Retail_demand_FE.csv')
df.to_csv(output_path, index=False)

print(f"\n  ✓ Dataset saved: {output_path}")
print(f"  Rows    : {df.shape[0]:,}")
print(f"  Columns : {df.shape[1]}")
print(f"\n  Final column list:")
for i, col in enumerate(df.columns, 1):
    tag = ' ← NEW' if col in FEATURES_CREATED else ''
    print(f"    {i:2}. {col}{tag}")


# =============================================================================
# STEP 7: FINAL SUMMARY REPORT
# =============================================================================

print("\n" + "="*70)
print("  FEATURE ENGINEERING COMPLETE — SUMMARY REPORT")
print("="*70)

print(f"""
INPUT DATASET  : Retail_demand_YJ_transformed.csv
OUTPUT DATASET : Retail_demand_FE.csv

SHAPE CHANGE:
  Before FE : 178,396 rows × 51 columns
  After  FE : {df.shape[0]:,} rows × {df.shape[1]} columns

FEATURES CREATED ({len(FEATURES_CREATED)} new):
  GROUP A — Price Features:
    ✓ is_high_price          (binary: price>$100 → always qty=1, corr=-0.67)
    ✓ price_tier             (Low/Mid/High categorical)
    ✓ price_tier_encoded     (ordinal: 1=Low, 2=Mid, 3=High)

  GROUP B — Target Encoding:
    ✓ product_avg_qty        (per-product historical avg qty, corr=+0.68)
    ✓ category_avg_qty       (per-category historical avg qty, corr=+0.67)
    ✓ dept_avg_qty           (per-department historical avg qty, corr=+0.41)

  GROUP C — Category Behavior Flags:
    ✓ is_bulk_category       (1 = category_avg_qty ≥ 2.5, corr=+0.67)
    ✓ is_single_unit_category(1 = category always has qty=1, corr=-0.67)

  GROUP D — Delivery & Operations:
    ✓ delivery_delay_days    (actual − scheduled shipping days)
    ✓ region_late_rate       (% late deliveries per region)
    ✓ is_late_delivery       (binary: Late delivery = 1)

  GROUP E — Payment & Order:
    ✓ is_digital_payment     (1 = Debit/Transfer/Payment)
    ✓ is_fulfilled           (1 = COMPLETE or CLOSED order)
    ✓ shipping_urgency_score (ordinal: Standard=1 → Same Day=4)

  GROUP F — Temporal / Calendar:
    ✓ is_q1_restocking       (1 = Q1, highest demand quarter)
    ✓ is_q4_slowdown         (1 = Q4, lowest demand quarter)
    ✓ is_weekend             (1 = Saturday or Sunday)
    ✓ is_sports_season       (1 = March or April, peak bulk buying)

  GROUP G — Discount Engineering:
    ✓ has_discount           (1 = discount > $0)
    ✓ discount_intensity     (0=None, 1=Small, 2=Medium, 3=Large)

COLUMNS DROPPED (18 total):
  Multicollinearity : order_item_product_price, product_price,
                      order_item_total, sales, order_item_total_log,
                      sales_log
  Zero-variance     : product_status (all = 0)
  Already encoded   : product_name, category_name, department_name,
                      type, delivery_status, shipping_mode,
                      order_item_discount
  High-cardinality  : customer_city, order_city, customer_state,
                      order_state, order_country, customer_country
  Redundant IDs     : customer_zipcode, product_card_id, latitude,
                      longitude, price_tier (string), product_image

TOP 5 MOST PREDICTIVE FEATURES:
  1. product_avg_qty        corr = +0.6795  (strongest)
  2. category_avg_qty       corr = +0.6729
  3. is_bulk_category       corr = +0.6697
  4. is_single_unit_category corr = -0.6673
  5. is_high_price          corr = -0.6669

MODELING RECOMMENDATIONS:
  • Use tree-based models (XGBoost, LightGBM, Random Forest) — non-normal data
  • Apply cross-validation-based target encoding to prevent leakage
  • OHE remaining categoricals: customer_segment (3), market (5), order_region (23)
  • Check class imbalance: Qty=1 is 54% of data → consider class weights
  • Use product_avg_qty, category_avg_qty as top priority features

PLOTS SAVED ({PLOTS_DIR}/):
  fe_01_feature_correlations.png
  fe_02_target_encoding_boxplots.png
  fe_03_price_features.png
  fe_04_binary_flags_comparison.png
  fe_05_temporal_patterns.png
  fe_06_delivery_features.png
  fe_07_discount_features.png
  fe_08_post_fe_correlation_heatmap.png
""")

print("="*70)
print("FEATURE ENGINEERING COMPLETE ✓  |  Next Phase: feature_selection")
print("="*70)
