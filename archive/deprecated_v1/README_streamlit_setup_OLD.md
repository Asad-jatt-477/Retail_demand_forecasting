# 📦 Retail Demand Forecasting — Streamlit App Setup

## ✅ Step-by-Step: App Kaise Run Karein

---

### STEP 1 — Folder Structure Banayein

Apne project folder mein yeh structure honi chahiye:

```
your_project_folder/
│
├── streamlit_app.py          ← (ye file jo aapko mili hai)
│
└── saved_models/             ← (Phase 9 se generate hoye files)
    ├── feature_list.json
    ├── scaler.pkl
    ├── LightGBM_tuned.pkl         ← best model
    ├── XGBoost_tuned.pkl
    ├── Random_Forest_tuned.pkl
    ├── Extra_Trees_tuned.pkl
    ├── LightGBM_default.pkl
    ├── XGBoost_default.pkl
    ├── Random_Forest_default.pkl
    └── Extra_Trees_default.pkl
```

> ⚠️ **Important:** `saved_models/` folder aur `streamlit_app.py` **same directory** mein hone chahiye.

---

### STEP 2 — Libraries Install Karein

```bash
pip install streamlit pandas numpy scikit-learn xgboost lightgbm joblib scipy
```

---

### STEP 3 — App Run Karein

```bash
streamlit run streamlit_app.py
```

Browser mein automatically khul jayega:
```
http://localhost:8501
```

---

## 🗂️ Agar saved_models/ Folder Nahi Hai?

Aapko pehle **Phase 9 Modeling Notebook** run karni hogi:

```bash
jupyter notebook 9th_Modeling_phase.ipynb
```

Notebook run karne ke baad yeh automatically ban jayengi:
- `saved_models/feature_list.json`
- `saved_models/scaler.pkl`
- `saved_models/LightGBM_tuned.pkl` (ya jo best model tha)
- Baaki default model files

---

## 🎯 App Features

| Feature | Details |
|---------|---------|
| **Input Form** | Saare final selected features input kar sakte hain |
| **Model Selector** | Sidebar mein 8 models mein se choose karein (tuned + default) |
| **Auto-Computed Fields** | Cyclic encodings, log transforms, interaction features automatically compute hote hain |
| **Prediction** | Quantity 1–5 predict hoti hai with confidence % |
| **Probability Bars** | Har class ki probability visual bars mein dikhti hai |
| **Business Insight** | Prediction ka business meaning bhi explain hota hai |

---

## 💡 Inputs Jo Automatically Compute Hote Hain

Yeh sab manually enter karne ki zaroorat nahi — app khud calculate karta hai:

- `product_price_log` → `log1p(product_price)`
- `effective_unit_price` → `price × (1 - discount_rate)`
- `unit_discount_amount` → `price × discount_rate`
- `price_tier` → budget / mid / premium / luxury
- `month_sin`, `month_cos` → cyclic encoding
- `weekday_sin`, `weekday_cos` → cyclic encoding
- `hour_sin`, `hour_cos` → cyclic encoding
- `is_weekend` → weekday >= 5
- `order_quarter` → month se derive
- `is_high_demand_season` → Q4/Q1
- `shipping_delay` → real - scheduled
- `shipping_efficiency_ratio` → real / (scheduled + 1)
- `is_fast_delivery` → real <= 3 days
- `dept_is_bulk_category` → department ke basis par
- `benefit_per_order_yj` → Yeo-Johnson transform
- `order_item_profit_ratio_yj` → Yeo-Johnson transform
- `is_profitable`, `is_high_margin`, `is_loss_order` → thresholds se
- `profit_tier` → 4 tiers
- `price_x_discount` → interaction feature
- `season_x_dept_bulk` → interaction feature
- `dept_bulk_x_price_tier` → interaction feature
- `shipping_x_price` → interaction feature

---

## ❓ Common Issues

**Error: "feature_list.json not found"**
→ `saved_models/` folder same directory mein rakhein

**Error: "No model .pkl files found"**
→ Phase 9 notebook pehle run karein, ya pkl files correct folder mein rakhein

**Prediction wrong lag rahi hai**
→ `product_target_enc` aur `relative_price_in_category` carefully enter karein — ye aapke actual data se depend karte hain

---

*Generated for: Retail Demand Forecasting Project — Phase 9 Prediction Interface*
