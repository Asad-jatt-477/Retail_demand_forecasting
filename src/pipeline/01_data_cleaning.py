# =============================================================================
#  RETAIL DEMAND FORECASTING — DATA CLEANING (FIXED VERSION)
#  Target Variable: Order Item Quantity (1-5)
#
#  BUGS FIXED vs ORIGINAL:

#  FIX-1: order_profit_per_order DROPPED — 100% Duplicate of benefit_per_order
#  FIX-2: 2018 partial-year data DROPPED — only Jan 2018 (2,123 rows, 1.2%)
#          causes temporal inconsistency in YoY analysis
#  FIX-3: days_for_shipment_scheduled KEPT with note — it is a pre-order
#          scheduled estimate, NOT a post-order leakage column
# =============================================================================

import pandas as pd
import numpy as np
import os

# ─── PROJECT-ROOT-RELATIVE PATHS ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_RAW_DIR = os.path.join(ROOT_DIR, 'data', 'raw')
DATA_DIR = os.path.join(ROOT_DIR, 'data', 'processed')
os.makedirs(DATA_DIR, exist_ok=True)

# =============================================================================
# STEP 1: DATASET LOAD
# =============================================================================

print("=" * 60)
print("STEP 1: DATASET LOAD")
print("=" * 60)

df = pd.read_csv(os.path.join(DATA_RAW_DIR, 'Retail_demand_forecasting_dataset.csv'), encoding='latin-1')
print(f"Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
print(f"Target column: 'Order Item Quantity'")
print(f"Target distribution:\n{df['Order Item Quantity'].value_counts().sort_index()}")


# =============================================================================
# STEP 2: DROP COLUMNS — MISSING / USELESS / PII / LEAKAGE
# =============================================================================

print("\n" + "=" * 60)
print("STEP 2: USELESS / PII / HIGH-MISSING / LEAKAGE COLUMNS DROP")
print("=" * 60)

cols_to_drop = [

    # P1 — 86% missing
    'Order Zipcode',

    # P2 — 100% empty column
    'Product Description',

    # P3 — PII / masked values
    'Customer Email',
    'Customer Password',
    'Customer Fname',
    'Customer Lname',
    'Customer Street',

    # NOTE: Sales, Order Item Total, Order Item Discount, Sales per customer,
    # Product Image, Product Status — USER REQUEST: yeh columns KEPT hain,
    # drop nahi kiye ja rahe. Inhe aap khud modeling phase mein evaluate karein.

    # FIX-3: DUPLICATE COLUMN — order_profit_per_order = benefit_per_order (100%)
    # Confirmed: abs difference between both columns = 0.0000 across all rows
    'Order Profit Per Order',

    # P6 — Redundant ID columns
    'Order Customer Id',
    'Product Category Id',

    # ID-only columns
    'Order Item Id',
]

df.drop(columns=cols_to_drop, inplace=True)
print(f"Dropped {len(cols_to_drop)} columns.")
print(f"Remaining columns: {df.shape[1]}")
print(f"\nDROPPED (with reason):")
print(f"  order_profit_per_order → DUPLICATE of benefit_per_order")
print(f"\nKEPT (user request — evaluate in modeling phase):")
print(f"  sales               → originally flagged as leakage (price × qty)")
print(f"  order_item_total    → originally flagged as leakage (qty-linked)")
print(f"  order_item_discount → originally flagged as leakage (price × rate × qty)")
print(f"  sales_per_customer  → originally flagged as leakage (price × qty × factor)")
print(f"  product_image       → originally flagged as zero predictive value")
print(f"  product_status      → originally flagged as zero predictive value")


# =============================================================================
# STEP 3: MISSING VALUES HANDLE KARO
# =============================================================================

print("\n" + "=" * 60)
print("STEP 3: MISSING VALUES HANDLE KARO")
print("=" * 60)

missing_before = df.isnull().sum()
missing_cols = missing_before[missing_before > 0]

if len(missing_cols) == 0:
    print("Koi missing values nahi — sab columns clean hain!")
else:
    print(f"Missing values found:\n{missing_cols}")

if 'Customer Zipcode' in df.columns:
    missing_zip = df['Customer Zipcode'].isnull().sum()
    if missing_zip > 0:
        zipcode_mode = df['Customer Zipcode'].mode()[0]
        df['Customer Zipcode'] = df['Customer Zipcode'].fillna(zipcode_mode)
        print(f"  Customer Zipcode: {missing_zip} missing → mode ({zipcode_mode}) se fill kiya")

print(f"\nMissing values after cleaning: {df.isnull().sum().sum()} total")


# =============================================================================
# STEP 4: DUPLICATE ROWS CHECK
# =============================================================================

print("\n" + "=" * 60)
print("STEP 4: DUPLICATE ROWS CHECK")
print("=" * 60)

duplicates = df.duplicated().sum()
print(f"Duplicate rows found: {duplicates}")

if duplicates > 0:
    df.drop_duplicates(inplace=True)
    print(f"Duplicates drop kiye. New shape: {df.shape}")
else:
    print("Koi duplicate rows nahi — dataset clean hai.")


# =============================================================================
# STEP 5: DATETIME COLUMNS PARSE KARO
# =============================================================================

print("\n" + "=" * 60)
print("STEP 5: DATETIME COLUMNS PARSE KARO")
print("=" * 60)

df['order date (DateOrders)']    = pd.to_datetime(df['order date (DateOrders)'],    format='mixed', dayfirst=False)
df['shipping date (DateOrders)'] = pd.to_datetime(df['shipping date (DateOrders)'], format='mixed', dayfirst=False)

print(f"  order date range: {df['order date (DateOrders)'].min()} → {df['order date (DateOrders)'].max()}")
print(f"  shipping date range: {df['shipping date (DateOrders)'].min()} → {df['shipping date (DateOrders)'].max()}")

# Temporal features
df['order_year']      = df['order date (DateOrders)'].dt.year
df['order_month']     = df['order date (DateOrders)'].dt.month
df['order_day']       = df['order date (DateOrders)'].dt.day
df['order_weekday']   = df['order date (DateOrders)'].dt.dayofweek
df['order_hour']      = df['order date (DateOrders)'].dt.hour
df['order_quarter']   = df['order date (DateOrders)'].dt.quarter

df['shipping_year']   = df['shipping date (DateOrders)'].dt.year
df['shipping_month']  = df['shipping date (DateOrders)'].dt.month

df['processing_delay_days'] = (
    df['shipping date (DateOrders)'] - df['order date (DateOrders)']
).dt.days

df.drop(columns=['order date (DateOrders)', 'shipping date (DateOrders)'], inplace=True)
print("\nTemporal features extracted and datetime columns dropped.")


# =============================================================================
# FIX-4: DROP PARTIAL 2018 DATA
# Original code mein yeh step missing tha — sirf Jan 2018 (2,123 rows = 1.2%)
# hai jo YoY analysis ko distort karta hai aur incomplete seasonal pattern deta hai.

# 2018 ke bache-kuche data ko dataset se nikaala (drop kiya) ja raha hai.
# Code mein 2018 ke saal ka data delete kiya ja raha hai.
# Use pata chalta hai ke sirf 2,123 rows hain, jo ke poore data ka bohot chota hissa (1.2%) hain
# aur is mein sirf January 2018 ka data hai.

# Agli line (df[df['order_year'] != 2018]) se woh 2018 ka poora data filter karke nikaal deta hai,
# aake data ab sirf January 2015 se September 2017 tak ka hi bache (jo ke mukammal saal hain).
# Q k : YoY (Year-over-Year) Analysis kharab ho rahi thi:
# Q k : Seasonal Patterns adhoore reh rahe thay

# =============================================================================

print("\n" + "=" * 60)
print("STEP 5b: 2018 PARTIAL YEAR DATA DROP (FIX-4)")
print("=" * 60)

rows_2018 = (df['order_year'] == 2018).sum()
print(f"  2018 rows found: {rows_2018:,} (only January — incomplete year)")
df = df[df['order_year'] != 2018].reset_index(drop=True)
print(f"  2018 data dropped. Dataset now covers Jan 2015 – Sep 2017 fully.")
print(f"  New shape: {df.shape[0]:,} rows")



# STEP 6: DATA TYPES FIX KARO
# =============================================================================

print("\n" + "=" * 60)
print("STEP 7: DATA TYPES FIX KARO")
print("=" * 60)

if 'Customer Zipcode' in df.columns:
    df['Customer Zipcode'] = df['Customer Zipcode'].astype('Int64')
    print("  Customer Zipcode → Int64")

categorical_cols = [
    'Type', 'Delivery Status', 'Category Name', 'Customer City',
    'Customer Country', 'Customer Segment', 'Customer State',
    'Department Name', 'Market', 'Order City', 'Order Country',
    'Order Region', 'Order State', 'Order Status',
    'Product Name', 'Shipping Mode'
]

converted = 0
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')
        converted += 1

print(f"  {converted} columns → category dtype")
print(f"\n  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")


# =============================================================================
# STEP 7: REDUNDANT COLUMNS FINAL DROP
# =============================================================================

print("\n" + "=" * 60)
print("STEP 8: REMAINING REDUNDANT COLUMNS DROP")
print("=" * 60)

redundant_cols = [
    'Customer Id',
    'Department Id',
    'Category Id',
    'Order Id',
    'Order Item Cardprod Id',
]

existing_redundant = [c for c in redundant_cols if c in df.columns]
df.drop(columns=existing_redundant, inplace=True)
print(f"Dropped: {existing_redundant}")
print(f"Final columns remaining: {df.shape[1]}")


# =============================================================================
# STEP 8: COLUMN NAMES STANDARDIZE
# =============================================================================

print("\n" + "=" * 60)
print("STEP 9: COLUMN NAMES STANDARDIZE")
print("=" * 60)

df.columns = (df.columns
              .str.strip()
              .str.lower()
              .str.replace(' ', '_', regex=False)
              .str.replace(r'[^\w]', '_', regex=True)
              .str.replace(r'_+', '_', regex=True)
              .str.strip('_'))

print("Column names → lowercase underscore format")
print("\nFinal columns list:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2}. {col}")


# =============================================================================
# STEP 9: FINAL SUMMARY + SAVE
# =============================================================================

print("\n" + "=" * 60)
print("STEP 10: FINAL SUMMARY")
print("=" * 60)

target_col = 'order_item_quantity'

print(f"\n  Total rows:    {df.shape[0]:,}")
print(f"  Total columns: {df.shape[1]}")
print(f"  Missing values: {df.isnull().sum().sum()}")
print(f"  Duplicate rows: {df.duplicated().sum()}")

print(f"\n  Target variable ({target_col}) distribution:")
total = len(df)
for qty, count in df[target_col].value_counts().sort_index().items():
    print(f"    Quantity {qty}: {count:>6,} rows ({count/total*100:.1f}%)")

print(f"\n  Data types:")
print(df.dtypes.value_counts().to_string())

print(f"\n  ✅ REMOVED:")
print(f"     order_profit_per_order → 100% duplicate of benefit_per_order")
print(f"     2018 partial data      → only Jan 2018, dropped for clean time series")
print(f"\n  ⚠  KEPT (user request — review before modeling):")
print(f"     sales               → potential leakage: price × quantity")
print(f"     order_item_total    → potential leakage: quantity-linked")
print(f"     order_item_discount → potential leakage: price × rate × qty")
print(f"     sales_per_customer  → potential leakage: price × qty × factor")
print(f"     product_image       → likely zero predictive value (URL string)")
print(f"     product_status      → likely zero predictive value (constant/near-constant)")

output_path = os.path.join(DATA_DIR, 'Retail_demand_forecasting_CLEANED.csv')
df.to_csv(output_path, index=False, encoding='utf-8')
print(f"\n  ✅ Cleaned dataset saved: '{output_path}'")
print(f"     Size: {os.path.getsize(output_path) / 1024**2:.1f} MB")

print("\n" + "=" * 60)
print("DATA CLEANING COMPLETE ✅")
print("Next step: EDA — use only non-leakage features")
print("=" * 60)
