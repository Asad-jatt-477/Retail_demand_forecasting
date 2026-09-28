# =============================================================================
#  RETAIL DEMAND FORECASTING — STREAMLIT PREDICTION APP
#  Target: order_item_quantity (1-5)
#  Models: Random Forest, Extra Trees, XGBoost, LightGBM (tuned best model)
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main background */
    .stApp {
        background: #0f1117;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #161b27 !important;
        border-right: 1px solid #2a3040;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1a2332 0%, #0f1117 60%);
        border: 1px solid #2a3040;
        border-radius: 16px;
        padding: 28px 36px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #4f8ef7, #7c4dff, #00c9a7);
    }
    .main-header h1 {
        color: #e8eaf0;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0 0 6px 0;
        letter-spacing: -0.3px;
    }
    .main-header p {
        color: #6b7a99;
        font-size: 0.92rem;
        margin: 0;
    }
    .header-badge {
        display: inline-block;
        background: #1e3a5f;
        color: #4f8ef7;
        border: 1px solid #2d5a9e;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 8px;
        margin-top: 10px;
        letter-spacing: 0.5px;
    }

    /* Section cards */
    .section-card {
        background: #161b27;
        border: 1px solid #2a3040;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 20px;
    }
    .section-title {
        color: #8b9dc3;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 16px;
        padding-bottom: 10px;
        border-bottom: 1px solid #2a3040;
    }

    /* Result card */
    .result-card {
        background: linear-gradient(135deg, #0d2137 0%, #091929 100%);
        border: 1px solid #1e4a7a;
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        margin-bottom: 20px;
    }
    .result-qty {
        font-family: 'JetBrains Mono', monospace;
        font-size: 5rem;
        font-weight: 700;
        color: #4f8ef7;
        line-height: 1;
        margin: 8px 0;
    }
    .result-label {
        color: #6b7a99;
        font-size: 0.85rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        font-weight: 500;
    }
    .result-confidence {
        color: #00c9a7;
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 10px;
    }

    /* Probability bars */
    .prob-row {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 10px;
    }
    .prob-label {
        color: #8b9dc3;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        width: 45px;
        flex-shrink: 0;
    }
    .prob-bar-bg {
        flex: 1;
        background: #1e2535;
        border-radius: 4px;
        height: 8px;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, #4f8ef7, #7c4dff);
        transition: width 0.5s ease;
    }
    .prob-bar-fill.highest {
        background: linear-gradient(90deg, #00c9a7, #4f8ef7);
    }
    .prob-pct {
        color: #e8eaf0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        width: 42px;
        text-align: right;
        flex-shrink: 0;
    }

    /* Info pills */
    .info-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #1a2235;
        border: 1px solid #2a3a55;
        border-radius: 8px;
        padding: 6px 14px;
        color: #8b9dc3;
        font-size: 0.82rem;
        margin: 4px 4px 4px 0;
    }
    .info-pill strong {
        color: #c9d4f0;
    }

    /* Warning / error */
    .warn-box {
        background: #2a1a0a;
        border: 1px solid #7a4400;
        border-radius: 10px;
        padding: 14px 18px;
        color: #e8a050;
        font-size: 0.85rem;
        margin-bottom: 16px;
    }
    .success-box {
        background: #0a2a1a;
        border: 1px solid #00754a;
        border-radius: 10px;
        padding: 14px 18px;
        color: #00c9a7;
        font-size: 0.85rem;
        margin-bottom: 16px;
    }

    /* Streamlit widget overrides */
    .stSelectbox label, .stNumberInput label, .stSlider label {
        color: #8b9dc3 !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
    }
    div[data-baseweb="select"] > div {
        background: #1e2535 !important;
        border-color: #2a3a55 !important;
        color: #c9d4f0 !important;
    }
    input[type="number"] {
        background: #1e2535 !important;
        border-color: #2a3a55 !important;
        color: #c9d4f0 !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #3b6fd4, #5b4dcc);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 32px;
        font-size: 0.95rem;
        font-weight: 600;
        width: 100%;
        letter-spacing: 0.3px;
        transition: all 0.2s ease;
        font-family: 'Inter', sans-serif;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #4f8ef7, #7c4dff);
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(79, 142, 247, 0.3);
    }
    hr {
        border-color: #2a3040;
    }
    h3 {
        color: #c9d4f0 !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── MODEL LOADING ────────────────────────────────────────────────────────────

@st.cache_resource
def load_models_and_features():
    """Load all saved models, scaler, and feature list."""
    base_path = "saved_models"

    # Feature list
    feature_path = os.path.join(base_path, "feature_list.json")
    if not os.path.exists(feature_path):
        return None, None, None, "❌ feature_list.json not found in 'saved_models/' folder."

    with open(feature_path, "r") as f:
        features = json.load(f)

    # Scaler
    scaler = None
    scaler_path = os.path.join(base_path, "scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)

    # Models — find available ones
    model_files = {
        "LightGBM (Tuned ★)": "LightGBM_tuned.pkl",
        "XGBoost (Tuned ★)":  "XGBoost_tuned.pkl",
        "Random Forest (Tuned ★)": "Random_Forest_tuned.pkl",
        "Extra Trees (Tuned ★)":   "Extra_Trees_tuned.pkl",
        "LightGBM (Default)":      "LightGBM_default.pkl",
        "XGBoost (Default)":       "XGBoost_default.pkl",
        "Random Forest (Default)": "Random_Forest_default.pkl",
        "Extra Trees (Default)":   "Extra_Trees_default.pkl",
    }

    loaded_models = {}
    for display_name, filename in model_files.items():
        path = os.path.join(base_path, filename)
        if os.path.exists(path):
            try:
                loaded_models[display_name] = joblib.load(path)
            except Exception:
                pass

    if not loaded_models:
        return None, None, None, "❌ No model .pkl files found in 'saved_models/' folder."

    return loaded_models, scaler, features, None


models, scaler, FEATURES, load_error = load_models_and_features()


# ─── PREDICTION FUNCTION ──────────────────────────────────────────────────────

def predict_quantity(input_data: dict, model, feature_list: list, model_name: str) -> dict:
    """Predict order_item_quantity given input features."""
    df_in = pd.DataFrame([input_data])

    # Fill missing features with 0
    for feat in feature_list:
        if feat not in df_in.columns:
            df_in[feat] = 0

    df_in = df_in[feature_list]

    # XGBoost needs 0-indexed labels
    is_xgb = "XGBoost" in model_name

    pred = model.predict(df_in)[0]
    proba = model.predict_proba(df_in)[0]

    if is_xgb:
        pred = int(pred) + 1  # 0-4 → 1-5
        classes = [1, 2, 3, 4, 5]
    else:
        classes = list(model.classes_)

    return {
        "predicted_quantity": int(pred),
        "confidence": round(float(max(proba)) * 100, 2),
        "class_probabilities": {
            f"Qty_{c}": round(float(p) * 100, 2)
            for c, p in zip(classes, proba)
        }
    }


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding: 0 0 20px 0;">
        <div style="color:#4f8ef7; font-size:1.4rem; font-weight:700; letter-spacing:-0.5px;">📦 Demand<br>Forecaster</div>
        <div style="color:#6b7a99; font-size:0.78rem; margin-top:6px;">Retail Quantity Predictor</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    if load_error:
        st.error(load_error)
        st.stop()

    # Model selector
    st.markdown('<div class="section-title">🤖 MODEL</div>', unsafe_allow_html=True)
    selected_model_name = st.selectbox(
        "Choose model",
        list(models.keys()),
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Feature info
    st.markdown('<div class="section-title">📊 DATASET INFO</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-pill">Features: <strong>{len(FEATURES)}</strong></div>
    <div class="info-pill">Target: <strong>Qty 1–5</strong></div>
    <div class="info-pill">Task: <strong>Multiclass</strong></div>
    <div class="info-pill">Metric: <strong>Macro F1</strong></div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="color:#3d4f6b; font-size:0.75rem; line-height:1.6;">
    Fill all input fields on the right, then click <strong style="color:#4f8ef7">Predict Quantity</strong> to get a forecast.
    </div>
    """, unsafe_allow_html=True)


# ─── MAIN HEADER ──────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>📦 Retail Demand Forecasting</h1>
    <p>Enter order details below to predict <strong>Order Item Quantity (1–5)</strong></p>
    <span class="header-badge">MULTICLASS CLASSIFICATION</span>
    <span class="header-badge">ML POWERED</span>
    <span class="header-badge">REAL-TIME</span>
</div>
""", unsafe_allow_html=True)


# ─── INPUT FORM ───────────────────────────────────────────────────────────────

st.markdown("### 📝 Order Input Features")
st.markdown(
    "<div style='color:#6b7a99; font-size:0.85rem; margin-bottom:20px;'>"
    "All fields correspond to the final selected features from your Feature Selection phase.</div>",
    unsafe_allow_html=True
)

# We'll organize inputs into logical groups matching the pipeline phases

col_left, col_right = st.columns([3, 2])

with col_left:

    # ── GROUP 1: Price & Discount ──────────────────────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">💰 Price & Discount Features</div>', unsafe_allow_html=True)

    pc1, pc2 = st.columns(2)
    with pc1:
        product_price = st.number_input(
            "Product Price ($)", min_value=0.0, max_value=2000.0,
            value=50.0, step=1.0,
            help="Original product unit price"
        )
        order_item_discount_rate = st.slider(
            "Discount Rate", min_value=0.0, max_value=1.0,
            value=0.05, step=0.01,
            help="Fraction discounted (0 = no discount, 1 = 100% off)"
        )
    with pc2:
        product_price_log = np.log1p(product_price)
        st.metric("product_price_log (auto)", f"{product_price_log:.4f}")

        effective_unit_price = product_price * (1 - order_item_discount_rate)
        st.metric("effective_unit_price (auto)", f"${effective_unit_price:.2f}")

    pc3, pc4 = st.columns(2)
    with pc3:
        unit_discount_amount = product_price * order_item_discount_rate
        price_median_ref = 59.99  # approximate from data
        is_bulk_friendly_price = int(product_price < price_median_ref)
        st.metric("unit_discount_amount (auto)", f"${unit_discount_amount:.2f}")
    with pc4:
        # price_tier: 0=budget, 1=mid, 2=premium, 3=luxury (qcut 4 bins)
        if product_price < 25:
            price_tier = 0
        elif product_price < 60:
            price_tier = 1
        elif product_price < 120:
            price_tier = 2
        else:
            price_tier = 3
        tier_names = {0: "Budget", 1: "Mid", 2: "Premium", 3: "Luxury"}
        st.metric("price_tier (auto)", f"{price_tier} — {tier_names[price_tier]}")

    # discount_tier
    if order_item_discount_rate == 0:
        discount_tier = 0
    elif order_item_discount_rate <= 0.10:
        discount_tier = 1
    elif order_item_discount_rate <= 0.20:
        discount_tier = 2
    else:
        discount_tier = 3

    st.markdown('</div>', unsafe_allow_html=True)

    # ── GROUP 2: Temporal Features ─────────────────────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🕐 Temporal & Seasonality Features</div>', unsafe_allow_html=True)

    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        order_month = st.selectbox(
            "Order Month", list(range(1, 13)),
            index=5,
            format_func=lambda m: ["Jan","Feb","Mar","Apr","May","Jun",
                                   "Jul","Aug","Sep","Oct","Nov","Dec"][m-1]
        )
        order_quarter = ((order_month - 1) // 3) + 1
    with tc2:
        order_weekday = st.selectbox(
            "Order Weekday (0=Mon)", list(range(7)),
            index=2,
            format_func=lambda d: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][d]
        )
        is_weekend = int(order_weekday >= 5)
    with tc3:
        order_hour = st.slider("Order Hour (0–23)", 0, 23, 14)
        order_day = st.slider("Order Day of Month", 1, 31, 15)

    # Cyclic encodings (auto-computed)
    month_sin = np.sin(2 * np.pi * order_month / 12)
    month_cos = np.cos(2 * np.pi * order_month / 12)
    weekday_sin = np.sin(2 * np.pi * order_weekday / 7)
    weekday_cos = np.cos(2 * np.pi * order_weekday / 7)
    hour_sin = np.sin(2 * np.pi * order_hour / 24)
    hour_cos = np.cos(2 * np.pi * order_hour / 24)

    # High demand season: Q4 (Oct–Dec) and Q1 partial
    is_high_demand_season = int(order_quarter in [4, 1])

    # Year trend (2015=0, 2016=1, 2017=2)
    order_year = st.selectbox("Order Year", [2015, 2016, 2017], index=1)
    year_map = {2015: 0, 2016: 1, 2017: 2}
    year_trend = year_map.get(order_year, 1)

    tc_info = st.columns(3)
    tc_info[0].metric("Quarter (auto)", f"Q{order_quarter}")
    tc_info[1].metric("Weekend (auto)", "Yes" if is_weekend else "No")
    tc_info[2].metric("High Season (auto)", "Yes" if is_high_demand_season else "No")

    st.markdown('</div>', unsafe_allow_html=True)

    # ── GROUP 3: Department & Category ────────────────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🏬 Department & Category Features</div>', unsafe_allow_html=True)

    dc1, dc2 = st.columns(2)
    with dc1:
        dept_name_raw = st.selectbox(
            "Department",
            ["Fan Shop", "Apparel", "Golf", "Footwear", "Outdoors",
             "Technology", "Fitness", "Book Shop", "Health and Beauty",
             "Pet Shop", "Discs Shop", "Toy", "Music"]
        )
        # Bulk categories (from EDA insight)
        bulk_depts = {"Fan Shop", "Apparel", "Footwear", "Outdoors", "Fitness"}
        dept_is_bulk_category = int(dept_name_raw in bulk_depts)

        # Target encoding approximate medians (from EDA/FE phase)
        dept_te_map = {
            "Fan Shop": 2.1, "Apparel": 1.9, "Golf": 1.7,
            "Footwear": 2.0, "Outdoors": 1.8, "Technology": 1.6,
            "Fitness": 1.9, "Book Shop": 1.5, "Health and Beauty": 1.7,
            "Pet Shop": 1.6, "Discs Shop": 1.5, "Toy": 1.8, "Music": 1.5
        }
        dept_target_enc = dept_te_map.get(dept_name_raw, 1.7)

    with dc2:
        category_name_raw = st.selectbox(
            "Product Category",
            ["Cleats", "Men's Footwear", "Women's Apparel", "Men's Apparel",
             "Sporting Goods", "Cardio Equipment", "Computers",
             "Cameras", "Children's Clothing", "Electronics",
             "Fishing", "Books", "Accessories"]
        )
        cat_te_map = {
            "Cleats": 2.2, "Men's Footwear": 2.0, "Women's Apparel": 1.9,
            "Men's Apparel": 1.9, "Sporting Goods": 1.8, "Cardio Equipment": 1.7,
            "Computers": 1.5, "Cameras": 1.6, "Children's Clothing": 1.9,
            "Electronics": 1.6, "Fishing": 1.8, "Books": 1.5, "Accessories": 1.7
        }
        category_target_enc = cat_te_map.get(category_name_raw, 1.7)

        relative_price_in_category = st.number_input(
            "Relative Price in Category",
            min_value=0.1, max_value=5.0, value=1.0, step=0.1,
            help="product_price / mean_price_in_category (1.0 = average)"
        )

    # Product target encoding
    product_target_enc = st.number_input(
        "Product Target Encoding (avg qty for this product)",
        min_value=1.0, max_value=5.0, value=1.8, step=0.1,
        help="Historical average order quantity for this product. Check from your FE output."
    )

    dc_info = st.columns(2)
    dc_info[0].metric("dept_is_bulk_category (auto)", "Yes" if dept_is_bulk_category else "No")
    dc_info[1].metric("dept_target_enc (auto)", f"{dept_target_enc:.2f}")

    st.markdown('</div>', unsafe_allow_html=True)

    # ── GROUP 4: Geography & Market ───────────────────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🌍 Geographic & Market Features</div>', unsafe_allow_html=True)

    gc1, gc2 = st.columns(2)
    with gc1:
        market_raw = st.selectbox(
            "Market",
            ["LATAM", "Europe", "Pacific Asia", "USCA", "Africa"]
        )
        market_te_map = {
            "LATAM": 1.9, "Europe": 1.8, "Pacific Asia": 1.8,
            "USCA": 1.7, "Africa": 1.6
        }
        market_target_enc = market_te_map.get(market_raw, 1.8)

        region_raw = st.selectbox(
            "Order Region",
            ["Western Europe", "Central America", "Oceania", "Eastern Asia",
             "South Asia", "West of USA", "East of USA", "Central Asia",
             "Southern Europe", "North Africa", "Eastern Europe",
             "Southeast Asia", "Caribbean", "South America"]
        )
        region_te_map = {
            "Western Europe": 1.85, "Central America": 1.90, "Oceania": 1.75,
            "Eastern Asia": 1.80, "South Asia": 1.70, "West of USA": 1.75,
            "East of USA": 1.72, "Central Asia": 1.68, "Southern Europe": 1.80,
            "North Africa": 1.65, "Eastern Europe": 1.78, "Southeast Asia": 1.75,
            "Caribbean": 1.88, "South America": 1.82
        }
        region_target_enc = region_te_map.get(region_raw, 1.78)

    with gc2:
        customer_segment_raw = st.selectbox(
            "Customer Segment",
            ["Consumer", "Corporate", "Home Office"]
        )
        seg_map = {"Consumer": 0, "Corporate": 1, "Home Office": 2}
        customer_segment_enc = seg_map[customer_segment_raw]

        longitude = st.number_input(
            "Longitude", min_value=-180.0, max_value=180.0,
            value=-90.0, step=0.5,
            help="Geographic longitude of the order location"
        )

    gc_info = st.columns(2)
    gc_info[0].metric("market_target_enc (auto)", f"{market_target_enc:.2f}")
    gc_info[1].metric("region_target_enc (auto)", f"{region_target_enc:.2f}")

    st.markdown('</div>', unsafe_allow_html=True)

    # ── GROUP 5: Fulfillment & Shipping ───────────────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🚚 Fulfillment & Shipping Features</div>', unsafe_allow_html=True)

    sc1, sc2 = st.columns(2)
    with sc1:
        days_for_shipping_real = st.slider(
            "Actual Shipping Days", 0, 14, 4,
            help="Real days taken to ship"
        )
        days_for_shipment_scheduled = st.slider(
            "Scheduled Shipping Days", 1, 10, 4,
            help="Originally scheduled shipping days"
        )
        shipping_delay = days_for_shipping_real - days_for_shipment_scheduled
        shipping_efficiency_ratio = days_for_shipping_real / (days_for_shipment_scheduled + 1)
        is_fast_delivery = int(days_for_shipping_real <= 3)

    with sc2:
        shipping_mode_raw = st.selectbox(
            "Shipping Mode",
            ["Standard Class", "Second Class", "First Class", "Same Day"]
        )
        ship_map = {"Standard Class": 0, "Second Class": 1,
                    "First Class": 2, "Same Day": 3}
        shipping_mode_enc = ship_map[shipping_mode_raw]

        order_status_raw = st.selectbox(
            "Order Status",
            ["COMPLETE", "PENDING", "CLOSED", "PENDING_PAYMENT",
             "CANCELED", "PROCESSING", "SUSPECTED_FRAUD", "ON_HOLD",
             "PAYMENT_REVIEW"]
        )
        status_risk_map = {
            "COMPLETE": 0, "CLOSED": 0, "PENDING": 1,
            "PENDING_PAYMENT": 2, "PROCESSING": 1,
            "CANCELED": 3, "SUSPECTED_FRAUD": 4,
            "ON_HOLD": 2, "PAYMENT_REVIEW": 2
        }
        order_status_risk = status_risk_map.get(order_status_raw, 1)

        payment_type_raw = st.selectbox(
            "Payment Type",
            ["DEBIT", "TRANSFER", "CASH", "PAYMENT"]
        )
        pay_map = {"DEBIT": 0, "TRANSFER": 1, "CASH": 2, "PAYMENT": 3}
        payment_type_enc = pay_map[payment_type_raw]

    sc_info = st.columns(3)
    sc_info[0].metric("shipping_delay (auto)", f"{shipping_delay:+d} days")
    sc_info[1].metric("efficiency_ratio (auto)", f"{shipping_efficiency_ratio:.2f}")
    sc_info[2].metric("is_fast_delivery (auto)", "Yes" if is_fast_delivery else "No")

    st.markdown('</div>', unsafe_allow_html=True)

    # ── GROUP 6: Profitability ─────────────────────────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📈 Profitability Features</div>', unsafe_allow_html=True)

    prc1, prc2 = st.columns(2)
    with prc1:
        benefit_per_order = st.number_input(
            "Benefit Per Order ($)", min_value=-500.0, max_value=500.0,
            value=15.0, step=1.0,
            help="Profit/loss per order (can be negative)"
        )
        order_item_profit_ratio = st.slider(
            "Order Item Profit Ratio", min_value=-1.0, max_value=1.0,
            value=0.20, step=0.01,
            help="Profit ratio: profit / revenue (negative = loss)"
        )

    with prc2:
        # Yeo-Johnson transforms (approximate: for typical ranges)
        from scipy import stats
        try:
            bpo_yj, _ = stats.yeojohnson([benefit_per_order])
            bpo_yj_val = float(bpo_yj[0])
        except Exception:
            bpo_yj_val = benefit_per_order
        try:
            ratio_yj, _ = stats.yeojohnson([order_item_profit_ratio])
            ratio_yj_val = float(ratio_yj[0])
        except Exception:
            ratio_yj_val = order_item_profit_ratio

        is_profitable = int(benefit_per_order > 0)
        is_high_margin = int(order_item_profit_ratio > 0.2)
        is_loss_order = int(benefit_per_order < 0)

        if benefit_per_order < -50:
            profit_tier = 0
        elif benefit_per_order < 0:
            profit_tier = 1
        elif benefit_per_order < 50:
            profit_tier = 2
        else:
            profit_tier = 3

    pr_info = st.columns(3)
    pr_info[0].metric("is_profitable (auto)", "Yes" if is_profitable else "No")
    pr_info[1].metric("is_high_margin (auto)", "Yes" if is_high_margin else "No")
    pr_info[2].metric("is_loss_order (auto)", "Yes" if is_loss_order else "No")

    st.markdown('</div>', unsafe_allow_html=True)

    # ── GROUP 7: Processing Delay ──────────────────────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">⏱ Processing & Delivery</div>', unsafe_allow_html=True)

    proc1, proc2 = st.columns(2)
    with proc1:
        processing_delay_days = st.slider(
            "Processing Delay (days from order to ship)", 0, 30, 3,
            help="days between order date and shipping date"
        )
        delivery_status_raw = st.selectbox(
            "Delivery Status",
            ["Shipping on time", "Late delivery", "Advance shipping",
             "Shipping canceled"]
        )
        del_status_map = {
            "Shipping on time": 0, "Late delivery": 1,
            "Advance shipping": -1, "Shipping canceled": 2
        }
        # stored as encoded value for model if it appears in FEATURES
        delivery_status_enc = del_status_map.get(delivery_status_raw, 0)

    with proc2:
        shipping_year = st.selectbox("Shipping Year", [2015, 2016, 2017], index=1)
        shipping_month = st.selectbox(
            "Shipping Month", list(range(1, 13)),
            index=5,
            format_func=lambda m: ["Jan","Feb","Mar","Apr","May","Jun",
                                   "Jul","Aug","Sep","Oct","Nov","Dec"][m-1]
        )

    st.markdown('</div>', unsafe_allow_html=True)


# ─── PREDICT BUTTON + RESULT ──────────────────────────────────────────────────

with col_right:
    st.markdown("### 🎯 Prediction")

    # Build feature dict — must match exactly what model was trained on
    # This is a comprehensive dict of ALL possible features
    input_dict = {
        # Price features
        "product_price":                product_price,
        "product_price_log":            product_price_log,
        "order_item_product_price":     product_price,     # same column
        "order_item_discount_rate":     order_item_discount_rate,
        "effective_unit_price":         effective_unit_price,
        "unit_discount_amount":         unit_discount_amount,
        "price_tier":                   price_tier,
        "discount_tier":                discount_tier,
        "is_bulk_friendly_price":       is_bulk_friendly_price,
        "price_x_discount":             product_price_log * order_item_discount_rate,
        "dept_bulk_x_price_tier":       dept_is_bulk_category * price_tier,
        "relative_price_in_category":   relative_price_in_category,
        "shipping_x_price":             shipping_mode_enc * price_tier,

        # Temporal
        "order_month":                  order_month,
        "order_quarter":                order_quarter,
        "order_weekday":                order_weekday,
        "order_hour":                   order_hour,
        "order_day":                    order_day,
        "order_year":                   order_year,
        "year_trend":                   year_trend,
        "is_weekend":                   is_weekend,
        "is_high_demand_season":        is_high_demand_season,
        "month_sin":                    month_sin,
        "month_cos":                    month_cos,
        "weekday_sin":                  weekday_sin,
        "weekday_cos":                  weekday_cos,
        "hour_sin":                     hour_sin,
        "hour_cos":                     hour_cos,
        "season_x_dept_bulk":           is_high_demand_season * dept_is_bulk_category,

        # Department & Category
        "dept_is_bulk_category":        dept_is_bulk_category,
        "dept_target_enc":              dept_target_enc,
        "category_target_enc":          category_target_enc,
        "product_target_enc":           product_target_enc,
        "relative_price_in_category":   relative_price_in_category,

        # Geography & Market
        "market_target_enc":            market_target_enc,
        "region_target_enc":            region_target_enc,
        "customer_segment_enc":         customer_segment_enc,
        "longitude":                    longitude,

        # Fulfillment
        "days_for_shipping_real":       days_for_shipping_real,
        "days_for_shipment_scheduled":  days_for_shipment_scheduled,
        "shipping_delay":               shipping_delay,
        "shipping_efficiency_ratio":    shipping_efficiency_ratio,
        "is_fast_delivery":             is_fast_delivery,
        "shipping_mode_enc":            shipping_mode_enc,
        "order_status_risk":            order_status_risk,
        "payment_type_enc":             payment_type_enc,
        "processing_delay_days":        processing_delay_days,
        "shipping_year":                shipping_year,
        "shipping_month":               shipping_month,

        # Profitability
        "benefit_per_order":            benefit_per_order,
        "benefit_per_order_yj":         bpo_yj_val,
        "order_item_profit_ratio":      order_item_profit_ratio,
        "order_item_profit_ratio_yj":   ratio_yj_val,
        "is_profitable":                is_profitable,
        "is_high_margin":               is_high_margin,
        "is_loss_order":                is_loss_order,
        "profit_tier":                  profit_tier,
    }

    # ── PREDICT BUTTON ─────────────────────────────────────────────────────────
    predict_btn = st.button("🔮 Predict Quantity", use_container_width=True)

    if predict_btn:
        selected_model = models[selected_model_name]

        with st.spinner("Running prediction..."):
            try:
                result = predict_quantity(
                    input_dict,
                    selected_model,
                    FEATURES,
                    selected_model_name
                )

                qty = result["predicted_quantity"]
                conf = result["confidence"]
                probs = result["class_probabilities"]

                # ── RESULT DISPLAY ─────────────────────────────────────────────
                qty_colors = {1: "#4f8ef7", 2: "#7c4dff", 3: "#00c9a7",
                              4: "#f7934c", 5: "#f74f4f"}
                qty_color = qty_colors.get(qty, "#4f8ef7")

                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">PREDICTED ORDER QUANTITY</div>
                    <div class="result-qty" style="color:{qty_color};">{qty}</div>
                    <div class="result-label">items per order</div>
                    <div class="result-confidence">✓ {conf:.1f}% confidence</div>
                </div>
                """, unsafe_allow_html=True)

                # ── PROBABILITY BARS ───────────────────────────────────────────
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">📊 CLASS PROBABILITIES</div>',
                            unsafe_allow_html=True)

                max_prob = max(probs.values())
                for cls_label, prob_val in sorted(probs.items()):
                    is_highest = (prob_val == max_prob)
                    fill_class = "highest" if is_highest else ""
                    bar_html = f"""
                    <div class="prob-row">
                        <div class="prob-label">{cls_label}</div>
                        <div class="prob-bar-bg">
                            <div class="prob-bar-fill {fill_class}"
                                 style="width:{prob_val:.1f}%"></div>
                        </div>
                        <div class="prob-pct">{prob_val:.1f}%</div>
                    </div>
                    """
                    st.markdown(bar_html, unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

                # ── BUSINESS INTERPRETATION ────────────────────────────────────
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">💡 BUSINESS INSIGHT</div>',
                            unsafe_allow_html=True)

                interpretations = {
                    1: ("🔵", "Single-unit purchase",
                        "Customer likely buying for personal use. Low urgency, standard fulfillment."),
                    2: ("🟣", "Small bulk purchase",
                        "Likely a repeat customer or small business buyer. Consider bundling offers."),
                    3: ("🟢", "Medium quantity order",
                        "Strong demand signal. Ensure adequate stock levels for this product."),
                    4: ("🟠", "High quantity order",
                        "Bulk buyer detected. May indicate B2B customer or promotional response."),
                    5: ("🔴", "Maximum quantity order",
                        "Very high demand. Priority fulfillment recommended. Check inventory immediately."),
                }

                icon, title, desc = interpretations.get(qty, ("📦", "Order", ""))
                st.markdown(f"""
                <div style="padding: 4px 0;">
                    <div style="color:#e8eaf0; font-size:1.05rem; font-weight:600; margin-bottom:8px;">
                        {icon} {title}
                    </div>
                    <div style="color:#8b9dc3; font-size:0.87rem; line-height:1.6;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

                # ── MODEL INFO ────────────────────────────────────────────────
                st.markdown(f"""
                <div class="info-pill">Model: <strong>{selected_model_name}</strong></div>
                <div class="info-pill">Features used: <strong>{len(FEATURES)}</strong></div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.markdown(f"""
                <div class="warn-box">
                    ⚠️ Prediction failed: {str(e)}<br><br>
                    Make sure your <code>saved_models/</code> folder contains the correct .pkl files
                    generated from your Phase 9 modeling notebook.
                </div>
                """, unsafe_allow_html=True)

    else:
        # Placeholder before prediction
        st.markdown("""
        <div style="background:#161b27; border:2px dashed #2a3a55; border-radius:16px;
                    padding:48px 24px; text-align:center; margin-bottom:20px;">
            <div style="font-size:3rem; margin-bottom:12px;">📦</div>
            <div style="color:#6b7a99; font-size:0.9rem; line-height:1.8;">
                Fill in the order details on the left<br>
                and click <strong style="color:#4f8ef7">Predict Quantity</strong><br>
                to see the forecast.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Quick reference
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📋 TARGET CLASSES</div>', unsafe_allow_html=True)
        qty_info = [
            ("1", "#4f8ef7", "Single unit"),
            ("2", "#7c4dff", "Small bulk"),
            ("3", "#00c9a7", "Medium bulk"),
            ("4", "#f7934c", "High quantity"),
            ("5", "#f74f4f", "Max quantity"),
        ]
        for qty_val, color, label in qty_info:
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:12px; margin:8px 0;">
                <div style="background:{color}20; border:1px solid {color};
                            border-radius:8px; padding:4px 14px;
                            font-family:'JetBrains Mono',monospace;
                            color:{color}; font-weight:700; font-size:1rem;">
                    {qty_val}
                </div>
                <div style="color:#8b9dc3; font-size:0.85rem;">{label}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:center;
            color:#3d4f6b; font-size:0.78rem; padding: 4px 0 16px 0;">
    <div>Retail Demand Forecasting Pipeline — Phase 9 Prediction Interface</div>
    <div>Target: <code style="color:#4f8ef7">order_item_quantity</code> (1–5)</div>
</div>
""", unsafe_allow_html=True)
