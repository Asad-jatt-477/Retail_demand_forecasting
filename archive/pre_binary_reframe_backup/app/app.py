# =============================================================================
#  RETAIL DEMAND FORECASTING — STREAMLIT PREDICTION APP (v2)
#  Target : order_item_quantity (1-5)
#  Models : Random Forest (winner) + XGBoost — both leakage-safe, size-capped
#
#  WHAT CHANGED FROM v1
#  ─────────────────────────────────────────────────────────────────────────
#  • REMOVED 6 input fields that asked the user for information that does not
#    exist yet when an order is placed: processing delay (days), delivery
#    status, order status, realized profit / profit ratio. These were
#    post-order / post-shipment values in v1 — asking a user to type them in
#    before the order happens made the form logically impossible to fill in
#    honestly for a real incoming order. See PROJECT_CHANGELOG.md.
#  • Loads the v2 pipelines (models/v2/), which were retrained without
#    those 6 columns, so the form now matches exactly what the model uses.
#  • Random Forest pipeline is now ~54 MB (was 2.6 GB in v1) — loads in
#    under a second instead of risking an out-of-memory crash.
#  • Product / Department are now picked by NAME from a dropdown; the app
#    looks up the correct internal group-id from group_id_mapping.json
#    (built from the same historical data the model was trained on) instead
#    of asking the user for a raw numeric code.
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS (professional dark theme) ────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
  .stApp { background: radial-gradient(circle at 15% 0%, #141a26 0%, #0b0e14 55%); }
  [data-testid="stSidebar"] { background: #10141f !important; border-right: 1px solid #232a3a; }
  [data-testid="stSidebar"] * { color: #cbd5e1; }

  .hero {
      background: linear-gradient(135deg, #17223a 0%, #0f1420 70%);
      border: 1px solid #263048; border-radius: 18px;
      padding: 32px 40px; margin-bottom: 26px; position: relative; overflow: hidden;
  }
  .hero::before {
      content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
      background: linear-gradient(90deg, #4f8ef7, #8b5cf6, #10b981);
  }
  .hero h1 { color: #f1f5f9; font-size: 1.9rem; font-weight: 700; margin: 0 0 6px 0; }
  .hero p  { color: #93a2bd; font-size: 0.95rem; margin: 0; line-height: 1.55; }

  .badge {
      display: inline-block; padding: 4px 12px; border-radius: 999px;
      font-size: 0.72rem; font-weight: 600; letter-spacing: .03em; margin-right: 8px;
  }
  .badge-fixed { background: rgba(16,185,129,.15); color: #34d399; border: 1px solid rgba(16,185,129,.35); }
  .badge-info  { background: rgba(79,142,247,.15); color: #60a5fa; border: 1px solid rgba(79,142,247,.35); }

  .card {
      background: #131826; border: 1px solid #232a3a; border-radius: 14px;
      padding: 22px 24px; margin-bottom: 18px;
  }
  .card h3 { color: #e2e8f0; font-size: 1.05rem; margin: 0 0 14px 0; font-weight: 600; }

  .metric-box {
      background: #0f1420; border: 1px solid #232a3a; border-radius: 12px;
      padding: 16px 18px; text-align: center;
  }
  .metric-box .v { font-size: 1.6rem; font-weight: 700; color: #f1f5f9; font-family: 'JetBrains Mono', monospace; }
  .metric-box .l { font-size: 0.75rem; color: #7c8aa5; margin-top: 4px; }

  .pred-banner {
      background: linear-gradient(135deg, #142a1f 0%, #0f1420 100%);
      border: 1px solid #1f5c3d; border-radius: 16px; padding: 26px 30px; text-align: center;
  }
  .pred-banner .qty { font-size: 3rem; font-weight: 800; color: #34d399; font-family: 'JetBrains Mono', monospace; }
  .pred-banner .lbl { color: #93a2bd; font-size: 0.9rem; margin-top: 4px; }

  .footnote { color: #5c6b85; font-size: 0.78rem; line-height: 1.6; margin-top: 6px; }

  .stButton>button {
      background: linear-gradient(135deg, #4f8ef7, #6366f1); color: white; border: none;
      border-radius: 10px; padding: 10px 26px; font-weight: 600; width: 100%;
  }
  .stButton>button:hover { opacity: 0.9; }

  h1, h2, h3, h4, p, label, .stMarkdown { color: #e2e8f0 !important; }
  [data-testid="stMetricValue"] { color: #f1f5f9 !important; }
</style>
""", unsafe_allow_html=True)

# ─── PATHS (relative to the project root, not the current working directory) ──
APP_DIR = os.path.dirname(os.path.abspath(__file__))     # .../app
ROOT_DIR = os.path.dirname(APP_DIR)                        # project root
MODELS_DIR = os.path.join(ROOT_DIR, 'models', 'v2')
MAP_PATH = os.path.join(MODELS_DIR, 'group_id_mapping.json')
METRICS_PATH = os.path.join(ROOT_DIR, 'reports', 'metrics_v2.json')


# ─── LEAKAGE-SAFE ENCODER (must be importable for unpickling) ────────────────
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold


class KFoldTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols, n_splits=5, smoothing=10, random_state=42):
        self.cols = cols
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.random_state = random_state

    def _smoothed_means(self, group_col, y):
        agg = pd.DataFrame({'g': group_col.values, 't': np.asarray(y)})
        agg = agg.groupby('g')['t'].agg(['mean', 'count'])
        global_mean = float(np.mean(y))
        smoothed = (agg['mean'] * agg['count'] + global_mean * self.smoothing) \
            / (agg['count'] + self.smoothing)
        return smoothed, global_mean

    def fit(self, X, y):
        X = X.reset_index(drop=True)
        y = pd.Series(np.asarray(y)).reset_index(drop=True)
        self.mapping_ = {}
        self.global_mean_ = float(y.mean())
        for col in self.cols:
            smoothed, _ = self._smoothed_means(X[col], y)
            self.mapping_[col] = smoothed
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            X[col] = X[col].map(self.mapping_[col]).fillna(self.global_mean_)
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


# ─── CACHED LOADERS ───────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open(os.path.join(MODELS_DIR, 'random_forest_pipeline.pkl'), 'rb') as f:
        rf = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'xgboost_pipeline.pkl'), 'rb') as f:
        xgb_bundle = pickle.load(f)
    return rf, xgb_bundle['pipeline'], xgb_bundle['label_encoder']


@st.cache_data
def load_mappings():
    with open(MAP_PATH) as f:
        m = json.load(f)
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    return m['product_name_to_group'], m['department_name_to_group'], metrics


rf_pipeline, xgb_pipeline, label_enc = load_models()
PRODUCT_MAP, DEPT_MAP, METRICS = load_mappings()
FEATURE_ORDER = METRICS['features_used']
CLASSES = [1, 2, 3, 4, 5]

MARKETS = ['Pacific Asia', 'USCA', 'Africa', 'Europe', 'LATAM']
REGIONS = ['South Asia', 'West Asia', 'Oceania', 'Eastern Asia', 'West of USA',
           'US Center', 'West Africa', 'Central Africa', 'North Africa',
           'Western Europe', 'Northern Europe', 'Central America', 'Caribbean',
           'South America', 'East Africa', 'Southern Europe', 'East of USA',
           'Canada', 'Southern Africa', 'Southeast Asia', 'Central Asia',
           'Eastern Europe', 'South of USA']
SEGMENTS = ['Consumer', 'Corporate', 'Home Office']
SHIP_MODES = {'Standard Class': 1, 'Second Class': 2, 'First Class': 3, 'Same Day': 4}

# ─── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <span class="badge badge-fixed">✓ Leakage-fixed</span>
  <span class="badge badge-info">18 pre-order features</span>
  <h1>📦 Retail Demand Forecasting</h1>
  <p>Predicts the quantity-class (1–5) of an <b>incoming</b> order using only information
  that is actually known at the moment the order is placed — no shipping outcome,
  delivery status, or realized profit is used as input.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Model")
    model_choice = st.selectbox("Prediction model", ["Random Forest (recommended)", "XGBoost"])
    st.markdown("---")
    st.markdown("### 📊 Test-set performance")
    key = 'random_forest' if 'Random Forest' in model_choice else 'xgboost'
    m = METRICS[key]
    st.markdown(f"""
    <div class="metric-box" style="margin-bottom:10px;">
      <div class="v">{m['f1_macro']:.3f}</div><div class="l">F1-Macro</div>
    </div>
    <div class="metric-box" style="margin-bottom:10px;">
      <div class="v">{m['accuracy']:.3f}</div><div class="l">Accuracy</div>
    </div>
    <div class="metric-box">
      <div class="v">{m['model_size_mb']:.0f} MB</div><div class="l">Model size on disk</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(
        '<p class="footnote">Baseline (random-guess) F1-Macro: '
        f"{METRICS['baseline']['f1_macro']:.3f}. Metrics come from "
        "metrics_v2.json, produced by 10_model_training.py on a held-out test set.</p>",
        unsafe_allow_html=True,
    )

# ─── INPUT FORM ───────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card"><h3>🛒 Order details</h3>', unsafe_allow_html=True)
    product_name = st.selectbox("Product", sorted(PRODUCT_MAP.keys()))
    product_price = st.number_input("Product price ($)", min_value=1.0, max_value=2000.0,
                                     value=100.0, step=1.0)
    discount_amount = st.number_input("Discount amount on this item ($, 0 if none)",
                                       min_value=0.0, max_value=500.0, value=0.0, step=1.0)
    order_date = st.date_input("Order date")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><h3>🚚 Shipping choice</h3>', unsafe_allow_html=True)
    shipping_mode = st.selectbox("Shipping mode selected by customer", list(SHIP_MODES.keys()))
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card"><h3>🏢 Customer & market</h3>', unsafe_allow_html=True)
    department_name = st.selectbox("Department", sorted(DEPT_MAP.keys()))
    customer_segment = st.selectbox("Customer segment", SEGMENTS)
    market = st.selectbox("Market", MARKETS)
    order_region = st.selectbox("Order region", REGIONS)
    st.markdown('</div>', unsafe_allow_html=True)

# ─── BUILD FEATURE ROW ────────────────────────────────────────────────────────
def build_feature_row():
    order_month = order_date.month
    order_weekday = order_date.weekday()
    is_weekend = int(order_weekday >= 5)
    is_q1 = int(order_date.month in [1, 2, 3])
    is_q4 = int(order_date.month in [10, 11, 12])

    if discount_amount <= 0:
        discount_intensity = 0
    elif discount_amount <= 5:
        discount_intensity = 1
    elif discount_amount <= 15:
        discount_intensity = 2
    else:
        discount_intensity = 3

    row = {
        'product_price_log': np.log1p(product_price),
        'discount_intensity': discount_intensity,
        'order_month': order_month,
        'order_weekday': order_weekday,
        'shipping_year': order_date.year,
        'shipping_urgency_score': SHIP_MODES[shipping_mode],
        'is_q1_restocking': is_q1,
        'is_q4_slowdown': is_q4,
        'is_weekend': is_weekend,
        'customer_segment_Consumer': int(customer_segment == 'Consumer'),
        'customer_segment_Corporate': int(customer_segment == 'Corporate'),
        'market_Europe': int(market == 'Europe'),
        'market_LATAM': int(market == 'LATAM'),
        'market_Pacific Asia': int(market == 'Pacific Asia'),
        'order_region_Oceania': int(order_region == 'Oceania'),
        'order_region_Western Europe': int(order_region == 'Western Europe'),
        'product_group': PRODUCT_MAP[product_name],
        'dept_group': DEPT_MAP[department_name],
    }
    return pd.DataFrame([row])[FEATURE_ORDER]


st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button("🔮 Predict order quantity")

if predict_clicked:
    X_input = build_feature_row()
    if 'Random Forest' in model_choice:
        pred = rf_pipeline.predict(X_input)[0]
        proba = rf_pipeline.predict_proba(X_input)[0]
        classes_order = list(rf_pipeline.named_steps['rf'].classes_)
    else:
        pred_enc = xgb_pipeline.predict(X_input)[0]
        pred = label_enc.inverse_transform([pred_enc])[0]
        proba = xgb_pipeline.predict_proba(X_input)[0]
        classes_order = list(label_enc.inverse_transform(xgb_pipeline.named_steps['xgb'].classes_))

    proba_map = {c: p for c, p in zip(classes_order, proba)}
    proba_sorted = [proba_map.get(c, 0.0) for c in CLASSES]

    st.markdown(f"""
    <div class="pred-banner">
      <div class="qty">Qty = {pred}</div>
      <div class="lbl">Predicted order-item quantity class</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card" style="margin-top:18px;"><h3>Class probabilities</h3>', unsafe_allow_html=True)
    prob_df = pd.DataFrame({'Quantity class': [f'Qty={c}' for c in CLASSES],
                             'Probability': proba_sorted}).set_index('Quantity class')
    st.bar_chart(prob_df)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        '<p class="footnote">Note: this is a demand-planning aid, not a guarantee. '
        'Per the test-set report, minority classes (Qty=3, Qty=4) are still the hardest '
        'to predict — treat mid-range predictions with more caution than Qty=1 predictions.</p>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<p class="footnote">Fill in the order details on the left and click '
        '"Predict order quantity".</p>', unsafe_allow_html=True,
    )
