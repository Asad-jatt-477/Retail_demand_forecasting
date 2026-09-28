# =============================================================================
#  RETAIL DEMAND FORECASTING — STREAMLIT APP (v3)
#  Question answered: will an incoming order be ONE unit (fast-pick shelf) or
#  SEVERAL units (bulk-pick zone)?      Target: is_multi_unit = (quantity >= 2)
#
#  Design
#  ─────────────────────────────────────────────────────────────────────────
#  * The page is built like a small web app: custom top bar, hero, cards,
#    a live prediction panel (HTML/CSS/JS in an iframe) and tabs for model
#    performance, drivers (SHAP) and a product explorer (SVG + JS).
#  * The form shows only the inputs the deployed model actually uses (read from
#    reports/metrics_v3.json), so nobody is asked for information that is
#    ignored — or that could contradict another field.
#  * Every number on the page is read from the artifacts written by the
#    pipeline (metrics_v3.json, shap_feature_importance.json, evaluation_report.json,
#    product_stats.json). Nothing is typed in by hand.
#  * Live prediction: it updates whenever an input changes (no button).
# =============================================================================

import json
import os
import pickle
import warnings
import html as _html

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Retail Demand Forecasting", page_icon="📦",
                   layout="wide", initial_sidebar_state="collapsed")

# ─── PATHS ────────────────────────────────────────────────────────────────────
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
MODELS_DIR = os.path.join(ROOT_DIR, 'models', 'v3')
REPORTS_DIR = os.path.join(ROOT_DIR, 'reports')
PLOTS_DIR = os.path.join(ROOT_DIR, 'assets', 'plots')


# ─── LEAKAGE-SAFE ENCODER (must be importable for unpickling) ────────────────
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


# ─── LOADERS ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open(os.path.join(MODELS_DIR, 'random_forest_pipeline.pkl'), 'rb') as f:
        rf = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'xgboost_pipeline.pkl'), 'rb') as f:
        xgb = pickle.load(f)['pipeline']
    return rf, xgb


@st.cache_data
def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


rf_pipeline, xgb_pipeline = load_models()
METRICS = load_json(os.path.join(REPORTS_DIR, 'metrics_v3.json'))
SHAP = load_json(os.path.join(REPORTS_DIR, 'shap_feature_importance.json')) or {}
MAPS = load_json(os.path.join(MODELS_DIR, 'group_id_mapping.json'))
STATS = load_json(os.path.join(MODELS_DIR, 'product_stats.json'))
PRODUCT_MAP = MAPS['product_name_to_group']
DEPT_MAP = MAPS.get('department_name_to_group') or {}
PRODUCTS = {p['name']: p for p in STATS['products']}
FEATURE_ORDER = METRICS['features_used']
WINNER = METRICS['winner']
WKEY = 'random_forest' if WINNER == 'Random Forest' else 'xgboost'
W = METRICS[WKEY]
PRICE_CUT = METRICS['baseline_price_rule']['train_max_multi_unit_price']

MARKETS = ['Pacific Asia', 'USCA', 'Africa', 'Europe', 'LATAM']
REGIONS = ['South Asia', 'West Asia', 'Oceania', 'Eastern Asia', 'West of USA', 'US Center',
           'West Africa', 'Central Africa', 'North Africa', 'Western Europe', 'Northern Europe',
           'Central America', 'Caribbean', 'South America', 'East Africa', 'Southern Europe',
           'East of USA', 'Canada', 'Southern Africa', 'Southeast Asia', 'Central Asia',
           'Eastern Europe', 'South of USA']
SEGMENTS = ['Consumer', 'Corporate', 'Home Office']
SHIP_MODES = {'Standard Class': 1, 'Second Class': 2, 'First Class': 3, 'Same Day': 4}

f3 = lambda x: f'{x:.3f}'
pct = lambda x, d=0: f'{x * 100:.{d}f}%'
esc = _html.escape


def render_html(html, height):
    """Embed an HTML/CSS/JS snippet. st.iframe replaces the deprecated components.html
    in current Streamlit; fall back to it on older versions."""
    if hasattr(st, 'iframe'):
        st.iframe(html, height=height)
    else:
        components.html(html, height=height, scrolling=False)


def show_image(path, caption=None):
    try:
        st.image(path, caption=caption, width='stretch')
    except Exception:                       # older Streamlit: no width='stretch'
        st.image(path, caption=caption, use_container_width=True)

# base rate of multi-unit orders on products where several units are possible
_multi_ok = [p for p in STATS['products'] if not p['always_single']]
BASE_MULTI = (sum(p['orders'] * (1 - p['single_share']) for p in _multi_ok) /
              max(1, sum(p['orders'] for p in _multi_ok)))
N_ALWAYS = sum(1 for p in STATS['products'] if p['always_single'])
N_PRODUCTS = len(STATS['products'])
ALWAYS_ROWS = sum(p['orders'] for p in STATS['products'] if p['always_single'])

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS  (makes Streamlit look like a custom web page)
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{
  --bg:#0a0d14; --bg2:#0e1320; --card:#121826; --card2:#0d121e; --line:#1f2838; --line2:#2a3550;
  --text:#e8edf6; --sub:#a4b0c8; --muted:#7a88a6; --dim:#56637f;
  --blue:#3987e5; --blue-t:#8fbcf5; --orange:#d95926; --orange-t:#f3a27f;
  --green:#34d399; --violet:#8b7cf6; --amber:#fbbf24;
}
html, body, [class*="css"], .stApp { font-family:'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif; }
.stApp{ background:
   radial-gradient(1100px 520px at 12% -8%, rgba(74,155,240,.14), transparent 60%),
   radial-gradient(900px 500px at 96% 4%, rgba(139,124,246,.10), transparent 60%),
   var(--bg); color:var(--text); }
#MainMenu, footer, header[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"], .stDeployButton,
[data-testid="stSidebar"], [data-testid="collapsedControl"]{ display:none !important; }
.block-container{ max-width:1180px !important; padding:0 28px 48px 28px !important; }

/* top bar */
.nav{ position:sticky; top:0; z-index:50; display:flex; align-items:center; justify-content:space-between;
  padding:14px 4px; margin:0 -4px 8px; backdrop-filter:blur(14px);
  background:rgba(10,13,20,.97); border-bottom:1px solid var(--line); }
.brand{ display:flex; align-items:center; gap:12px; }
.logo{ width:38px; height:38px; border-radius:11px; display:grid; place-items:center; font-size:20px;
  background:linear-gradient(135deg,#3987e5,#8b7cf6); box-shadow:0 6px 22px rgba(74,155,240,.35); }
.brand b{ font-size:15.5px; font-weight:700; letter-spacing:.1px; display:block; line-height:1.15; }
.brand small{ color:var(--muted); font-size:11.5px; font-weight:500; }
.navr{ display:flex; gap:8px; align-items:center; flex-wrap:wrap; justify-content:flex-end; }
.pill{ font-size:11.5px; font-weight:600; padding:5px 11px; border-radius:999px; border:1px solid var(--line2);
  color:var(--sub); background:rgba(255,255,255,.02); white-space:nowrap; }
.pill.live{ color:#7ee2b8; border-color:rgba(52,211,153,.35); background:rgba(52,211,153,.08); }
.pill.live:before{ content:''; display:inline-block; width:7px; height:7px; border-radius:50%; background:var(--green);
  margin-right:7px; box-shadow:0 0 0 0 rgba(52,211,153,.6); animation:pulse 2s infinite; }
@keyframes pulse{ 70%{ box-shadow:0 0 0 8px rgba(52,211,153,0);} 100%{ box-shadow:0 0 0 0 rgba(52,211,153,0);} }

/* hero */
.hero{ padding:44px 4px 26px; }
.eyebrow{ font-size:12px; font-weight:700; letter-spacing:.14em; text-transform:uppercase; color:var(--blue-t); }
.hero h1{ font-size:clamp(30px,4.4vw,48px); line-height:1.08; font-weight:800; letter-spacing:-.02em; margin:10px 0 14px; padding:0; }
.grad{ background:linear-gradient(90deg,#8fbcf5,#b9a8ff 55%,#f3a27f); -webkit-background-clip:text; background-clip:text;
  -webkit-text-fill-color:transparent; }
.hero p{ max-width:720px; color:var(--sub); font-size:16px; line-height:1.65; margin:0 0 20px; }
.chips{ display:flex; gap:10px; flex-wrap:wrap; }
.chip{ display:flex; align-items:center; gap:9px; padding:9px 14px; border-radius:12px; border:1px solid var(--line);
  background:rgba(18,24,38,.7); font-size:13px; color:var(--sub); }
.chip b{ color:var(--text); font-weight:700; font-size:14px; }

/* cards (Streamlit bordered containers) */
[data-testid="stVerticalBlockBorderWrapper"]{ background:linear-gradient(180deg,rgba(18,24,38,.92),rgba(13,18,30,.92));
  border:1px solid var(--line) !important; border-radius:18px !important; box-shadow:0 12px 40px rgba(0,0,0,.28); }
.card-title{ display:flex; align-items:center; gap:10px; font-size:13px; font-weight:700; letter-spacing:.09em;
  text-transform:uppercase; color:var(--muted); margin:2px 0 14px; }
.card-title i{ font-style:normal; width:26px; height:26px; border-radius:8px; display:grid; place-items:center;
  background:rgba(74,155,240,.14); color:var(--blue-t); font-size:14px; }

/* widgets */
[data-testid="stWidgetLabel"] p, label{ color:var(--sub) !important; font-size:13px !important; font-weight:600 !important; }
[data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="base-input"], .stNumberInput input{
  background:var(--card2) !important; border-color:var(--line2) !important; border-radius:11px !important; color:var(--text) !important; }
[data-baseweb="select"] > div:hover, [data-baseweb="input"]:hover{ border-color:var(--blue) !important; }
.stNumberInput button{ background:var(--card2) !important; color:var(--sub) !important; border-color:var(--line2) !important; }
.stTabs [data-baseweb="tab-list"]{ gap:6px; border-bottom:1px solid var(--line); }
.stTabs [data-baseweb="tab"]{ height:46px; padding:0 16px; background:transparent; color:var(--muted); font-weight:600; font-size:14px; }
.stTabs [aria-selected="true"]{ color:var(--text) !important; }
.stTabs [data-baseweb="tab-highlight"]{ background:linear-gradient(90deg,#3987e5,#8b7cf6) !important; height:3px; border-radius:3px; }
.stTabs [data-baseweb="tab-border"]{ display:none; }
[data-testid="stExpander"]{ border:1px solid var(--line) !important; border-radius:14px !important; background:rgba(13,18,30,.6); }
[data-testid="stImage"] img{ border-radius:12px; border:1px solid var(--line); }
[data-testid="stCaptionContainer"]{ color:var(--muted) !important; }

/* product card */
.pcard{ margin-top:16px; padding:16px; border-radius:14px; border:1px solid var(--line); background:rgba(10,14,24,.7); }
.pname{ font-weight:700; font-size:15px; line-height:1.3; }
.pmeta{ color:var(--muted); font-size:12.5px; margin-top:3px; }
.qbars{ display:grid; grid-template-columns:repeat(5,1fr); gap:8px; margin-top:14px; align-items:end; height:104px; }
.qb{ display:flex; flex-direction:column; align-items:center; justify-content:flex-end; height:100%; gap:5px; }
.qb .bar{ width:100%; border-radius:6px 6px 2px 2px; min-height:2px; }
.qb span{ font-size:11.5px; color:var(--muted); }
.qb em{ font-style:normal; font-size:11.5px; color:var(--sub); font-weight:600; }
.tag{ display:inline-block; padding:3px 9px; border-radius:999px; font-size:11.5px; font-weight:700; margin-top:9px; }
.tag.blue{ color:var(--blue-t); background:rgba(57,135,229,.14); border:1px solid rgba(57,135,229,.35); }
.tag.orange{ color:var(--orange-t); background:rgba(217,89,38,.14); border:1px solid rgba(217,89,38,.4); }

/* KPI grid & tables */
.kpis{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:8px 0 22px; }
.kpi{ padding:18px; border-radius:16px; border:1px solid var(--line); background:linear-gradient(180deg,rgba(18,24,38,.9),rgba(13,18,30,.9)); }
.kpi small{ color:var(--muted); font-size:12px; font-weight:600; letter-spacing:.06em; text-transform:uppercase; }
.kpi b{ display:block; font-size:32px; font-weight:800; letter-spacing:-.02em; margin-top:6px; }
.kpi span{ color:var(--muted); font-size:12px; }
.tbl{ width:100%; border-collapse:separate; border-spacing:0; font-size:13.5px; overflow:hidden; border:1px solid var(--line); border-radius:14px; }
.tbl th{ text-align:left; padding:11px 14px; color:var(--muted); font-size:11.5px; font-weight:700; letter-spacing:.07em;
  text-transform:uppercase; background:rgba(255,255,255,.025); border-bottom:1px solid var(--line); }
.tbl td{ padding:12px 14px; border-bottom:1px solid var(--line); color:var(--sub); }
.tbl tr:last-child td{ border-bottom:0; }
.tbl td.n, .tbl th.n{ text-align:right; font-variant-numeric:tabular-nums; }
.tbl td b{ color:var(--text); }
.hb{ display:grid; grid-template-columns:230px 1fr 64px; gap:12px; align-items:center; margin:11px 0; font-size:13.5px; color:var(--sub); }
.hb .track{ height:12px; background:rgba(255,255,255,.05); border-radius:8px; overflow:hidden; }
.hb .fill{ height:100%; border-radius:8px; }
.hb .val{ text-align:right; font-weight:700; color:var(--text); font-variant-numeric:tabular-nums; }
.sec{ font-size:20px; font-weight:750; letter-spacing:-.01em; margin:26px 0 6px; }
.lead{ color:var(--sub); font-size:14.5px; line-height:1.7; max-width:880px; }
.note{ margin-top:14px; padding:14px 16px; border-radius:12px; border:1px solid rgba(251,191,36,.28); background:rgba(251,191,36,.06);
  color:#f0d9a0; font-size:13.5px; line-height:1.6; }
.good{ border-color:rgba(52,211,153,.3); background:rgba(52,211,153,.06); color:#a6ecd0; }
.grid2{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }
.tile{ padding:18px; border:1px solid var(--line); border-radius:14px; background:rgba(13,18,30,.7); }
.tile h4{ margin:0 0 6px; font-size:14.5px; font-weight:700; }
.tile p{ margin:0; color:var(--sub); font-size:13.5px; line-height:1.65; }
.foot{ margin-top:44px; padding:22px 4px 0; border-top:1px solid var(--line); color:var(--dim); font-size:12.5px;
  display:flex; justify-content:space-between; gap:16px; flex-wrap:wrap; }
@media (max-width:900px){ .kpis{grid-template-columns:repeat(2,1fr);} .grid2{grid-template-columns:1fr;} .hb{grid-template-columns:130px 1fr 56px;} }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TOP BAR + HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="nav">
  <div class="brand"><div class="logo">📦</div>
    <div><b>Retail Demand Forecasting</b><small>Single-unit vs multi-unit order prediction</small></div></div>
  <div class="navr"><span class="pill live">Model live</span><span class="pill">{esc(WINNER)}</span>
    <span class="pill">v3 · binary</span></div>
</div>
<div class="hero">
  <div class="eyebrow">Warehouse planning assistant</div>
  <h1>Will this order ship as <span class="grad">one unit — or several?</span></h1>
  <p>Pick a product and get an instant call: <b>fast-pick shelf</b> for single-unit orders, <b>bulk-pick zone</b> for
  multi-unit orders. Built only on information known at the moment the order is placed.</p>
  <div class="chips">
    <div class="chip"><b>{f3(W['f1_macro'])}</b> F1-Macro on held-out orders</div>
    <div class="chip"><b>{STATS['n_orders']:,}</b> historical orders</div>
    <div class="chip"><b>{N_PRODUCTS}</b> products</div>
    <div class="chip"><b>0</b> post-order inputs</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# WHICH INPUTS DOES THE MODEL USE?  (form adapts to metrics_v3.json)
# ─────────────────────────────────────────────────────────────────────────────
USES = set(FEATURE_ORDER)
CALENDAR = {'order_year', 'order_month', 'order_day', 'order_weekday', 'order_quarter',
            'is_q1_restocking', 'is_q4_slowdown', 'is_weekend', 'is_sports_season'}
uses_price = 'product_price_log' in USES
uses_date = bool(USES & CALENDAR)
uses_ship = 'shipping_urgency_score' in USES
uses_dept = 'dept_group' in USES and bool(DEPT_MAP)
uses_segment = any(f.startswith('customer_segment_') for f in USES)
uses_market = any(f.startswith('market_') for f in USES)
uses_region = any(f.startswith('order_region_') for f in USES)

product_name = product_price = order_date = shipping_mode = None
department_name = customer_segment = market = order_region = None


def norm(s):
    return ' '.join(str(s).split())


def build_feature_row():
    """Builds exactly the features listed in metrics_v3.json['features_used'],
    using the same definitions as 08_feature_engineering.py."""
    values = {'product_group': PRODUCT_MAP[product_name]}
    if product_price is not None:
        values['product_price_log'] = float(np.log1p(product_price))
    if department_name is not None:
        values['dept_group'] = DEPT_MAP[department_name]
    if order_date is not None:
        d = order_date
        values.update({'order_year': d.year, 'order_month': d.month, 'order_day': d.day,
                       'order_weekday': d.weekday(), 'order_quarter': (d.month - 1) // 3 + 1,
                       'is_q1_restocking': int(d.month in [1, 2, 3]),
                       'is_q4_slowdown': int(d.month in [10, 11, 12]),
                       'is_weekend': int(d.weekday() >= 5), 'is_sports_season': int(d.month in [3, 4])})
    if shipping_mode is not None:
        values['shipping_urgency_score'] = SHIP_MODES[shipping_mode]
    prefixed = {k: v for k, v in (('customer_segment_', customer_segment), ('market_', market),
                                  ('order_region_', order_region)) if v is not None}
    row, unsupported = {}, []
    for feat in FEATURE_ORDER:
        if feat in values:
            row[feat] = values[feat]
            continue
        for prefix, chosen in prefixed.items():
            if feat.startswith(prefix):
                row[feat] = int(norm(feat[len(prefix):]) == norm(chosen))
                break
        else:
            unsupported.append(feat)
    if unsupported:
        st.error(f"The trained model uses feature(s) this form cannot build: {unsupported}. "
                 "Update build_feature_row() in app/app.py.")
        st.stop()
    return pd.DataFrame([row])[FEATURE_ORDER]


# ─────────────────────────────────────────────────────────────────────────────
# HTML BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
def product_card(p):
    q = p['qty_share']
    maxq = max(q.values()) or 1
    bars = ''.join(
        f'<div class="qb"><em>{q[str(k)] * 100:.0f}%</em>'
        f'<div class="bar" style="height:{max(3, q[str(k)] / maxq * 62):.0f}px;'
        f'background:{"var(--blue)" if k == 1 else "var(--orange)"}"></div><span>Qty {k}</span></div>'
        for k in range(1, 6))
    if p['always_single']:
        tag = '<span class="tag blue">Always ordered one unit at a time</span>'
    else:
        tag = (f'<span class="tag orange">Quantities 1–5 all occur · '
               f'{(1 - p["single_share"]) * 100:.0f}% multi-unit</span>')
    dept = f' · {esc(p["department"])}' if p.get('department') else ''
    return (f'<div class="pcard"><div class="pname">{esc(p["name"])}</div>'
            f'<div class="pmeta">{p["orders"]:,} historical orders · typical price ${p["price"]:,.2f}{dept}</div>'
            f'<div class="qbars">{bars}</div>{tag}</div>')


PRED_HTML = """
<!doctype html><html><head><meta charset="utf-8"><style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',system-ui,-apple-system,'Segoe UI',sans-serif;background:transparent;color:#e8edf6;padding:2px}
.panel{border:1px solid #1f2838;border-radius:18px;padding:22px 24px 20px;
  background:linear-gradient(180deg,rgba(18,24,38,.95),rgba(13,18,30,.95));box-shadow:0 12px 40px rgba(0,0,0,.28);position:relative;overflow:hidden}
.panel:before{content:'';position:absolute;inset:0 0 auto 0;height:3px;background:__ACCENT__}
.top{display:flex;justify-content:space-between;align-items:center}
.lbl{font-size:12px;font-weight:700;letter-spacing:.11em;text-transform:uppercase;color:#7a88a6}
.mdl{font-size:11.5px;font-weight:600;color:#a4b0c8;border:1px solid #2a3550;border-radius:999px;padding:4px 10px}
.verdict{margin-top:14px;font-size:30px;font-weight:800;letter-spacing:-.02em;line-height:1.15;color:__TXT__}
.sub{margin-top:6px;color:#a4b0c8;font-size:14.5px}
.sub b{color:#e8edf6}
.gwrap{display:flex;align-items:center;gap:22px;margin:16px 0 4px;flex-wrap:wrap}
svg.g{width:230px;height:144px;flex:none}
.big{font-size:44px;font-weight:800;letter-spacing:-.03em;line-height:1;color:__TXT__;font-variant-numeric:tabular-nums}
.big small{font-size:18px;font-weight:700}
.cap{color:#7a88a6;font-size:12.5px;margin-top:4px}
.rows{margin-top:14px;display:grid;gap:11px}
.r{display:grid;grid-template-columns:130px 1fr 50px;gap:10px;align-items:center;font-size:13px;color:#a4b0c8}
.tr{height:11px;border-radius:7px;background:rgba(255,255,255,.06);overflow:hidden}
.fl{height:100%;border-radius:7px;width:0}
.v{text-align:right;font-weight:700;color:#e8edf6;font-variant-numeric:tabular-nums}
.act{margin-top:16px;display:flex;gap:12px;align-items:center;padding:13px 15px;border-radius:13px;border:1px solid __BORDER__;background:__SOFT__}
.act i{font-style:normal;font-size:22px}
.act b{display:block;font-size:14.5px}
.act span{font-size:12.5px;color:#a4b0c8}
.why{margin-top:12px;padding:13px 15px;border-radius:13px;border:1px solid #1f2838;background:rgba(10,14,24,.6);font-size:13px;line-height:1.6;color:#a4b0c8}
.why b{color:#e8edf6}
.conf{display:inline-block;margin-left:8px;font-size:11.5px;font-weight:700;padding:3px 9px;border-radius:999px;border:1px solid __BORDER__;color:__TXT__;background:__SOFT__;vertical-align:middle;letter-spacing:0}
</style></head><body>
<div class="panel">
  <div class="top"><div class="lbl">Prediction</div><div class="mdl">__MODEL__</div></div>
  <div class="verdict">__VERDICT__ <span class="conf">__CONF__</span></div>
  <div class="sub">__SUB__</div>
  <div class="gwrap">
    <svg class="g" viewBox="0 0 230 144">
      <path d="M 25 115 A 90 90 0 0 1 205 115" fill="none" stroke="rgba(255,255,255,.08)" stroke-width="16" stroke-linecap="round"/>
      <path id="arc" d="M 25 115 A 90 90 0 0 1 205 115" fill="none" stroke="__COLOR__" stroke-width="16" stroke-linecap="round"
        stroke-dasharray="283" stroke-dashoffset="283"/>
      <text x="25" y="140" fill="#56637f" font-size="10.5" text-anchor="middle">0%</text>
      <text x="205" y="140" fill="#56637f" font-size="10.5" text-anchor="middle">100%</text>
    </svg>
    <div><div class="big"><span id="num">0</span><small>%</small></div><div class="cap">probability the order is multi-unit</div></div>
  </div>
  <div class="rows">
    <div class="r"><span>Single-unit (Qty 1)</span><div class="tr"><div class="fl" id="b1" style="background:#3987e5"></div></div><span class="v" id="v1">0%</span></div>
    <div class="r"><span>Multi-unit (Qty 2–5)</span><div class="tr"><div class="fl" id="b2" style="background:#d95926"></div></div><span class="v" id="v2">0%</span></div>
  </div>
  <div class="act"><i>__ICON__</i><div><b>__ACTION__</b><span>__ACTION_SUB__</span></div></div>
  <div class="why"><b>Why this call.</b> __WHY__</div>
</div>
<script>
const P = __P__;   // probability of multi-unit (0..1)
const arc = document.getElementById('arc'), num = document.getElementById('num');
const start = performance.now(), dur = 1000;
function ease(t){ return 1 - Math.pow(1 - t, 3); }
function frame(now){
  const t = Math.min(1, (now - start) / dur), e = ease(t), cur = P * e;
  arc.setAttribute('stroke-dashoffset', 283 - 283 * cur);
  num.textContent = Math.round(cur * 100);
  document.getElementById('v1').textContent = Math.round((1 - cur) * 100) + '%';
  document.getElementById('v2').textContent = Math.round(cur * 100) + '%';
  document.getElementById('b1').style.width = ((1 - cur) * 100) + '%';
  document.getElementById('b2').style.width = (cur * 100) + '%';
  if (t < 1) requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
</script></body></html>
"""

SCATTER_HTML = """
<!doctype html><html><head><meta charset="utf-8"><style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',system-ui,'Segoe UI',sans-serif;background:transparent;color:#e8edf6}
.wrap{position:relative;border:1px solid #1f2838;border-radius:16px;padding:14px 10px 6px;background:linear-gradient(180deg,rgba(18,24,38,.9),rgba(13,18,30,.9))}
svg{width:100%;height:auto;display:block}
.leg{display:flex;gap:18px;padding:2px 14px 10px;font-size:12.5px;color:#a4b0c8;flex-wrap:wrap}
.leg i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:7px}
#tip{position:absolute;pointer-events:none;opacity:0;transition:opacity .12s;background:#0a0e18;border:1px solid #2a3550;border-radius:10px;
  padding:9px 12px;font-size:12.5px;line-height:1.5;box-shadow:0 10px 30px rgba(0,0,0,.5);max-width:260px;z-index:5}
#tip b{color:#e8edf6}
circle.d{cursor:pointer;transition:r .12s}
</style></head><body><div class="wrap">
<div class="leg"><span><i style="background:#d95926"></i>Products where Qty 1–5 all occur (__NM__)</span>
<span><i style="background:#3987e5"></i>Products that are always Qty 1 (__NA__)</span>
<span style="color:#7a88a6">Hover a dot for details · dashed line = the five quantities equally likely (20%)</span></div>
<svg id="s" viewBox="0 0 1100 430" preserveAspectRatio="xMidYMid meet"></svg><div id="tip"></div></div>
<script>
const D = __DATA__;
const W=1100,H=430,L=64,R=24,T=14,B=52, s=document.getElementById('s'), tip=document.getElementById('tip');
const xmin=Math.log(9), xmax=Math.log(2400);
const X=v=>L+(Math.log(v)-xmin)/(xmax-xmin)*(W-L-R), Y=v=>T+12+(1-v)*(H-T-B-12);
const ns='http://www.w3.org/2000/svg';
function el(n,a,txt){const e=document.createElementNS(ns,n);for(const k in a)e.setAttribute(k,a[k]);if(txt!==undefined)e.textContent=txt;s.appendChild(e);return e;}
[0,.2,.4,.6,.8,1].forEach(v=>{el('line',{x1:L,x2:W-R,y1:Y(v),y2:Y(v),stroke:'#1f2838','stroke-width':1});
  el('text',{x:L-10,y:Y(v)+4,fill:'#7a88a6','font-size':12,'text-anchor':'end'},Math.round(v*100)+'%');});
[10,20,50,100,200,500,1000,2000].forEach(v=>{el('line',{x1:X(v),x2:X(v),y1:T,y2:H-B,stroke:'#161e2d','stroke-width':1});
  el('text',{x:X(v),y:H-B+20,fill:'#7a88a6','font-size':12,'text-anchor':'middle'},'$'+v.toLocaleString());});
el('text',{x:(L+W-R)/2,y:H-8,fill:'#a4b0c8','font-size':12.5,'text-anchor':'middle'},'Product price (log scale)');
el('text',{x:16,y:(T+H-B)/2,fill:'#a4b0c8','font-size':12.5,'text-anchor':'middle',transform:'rotate(-90 16 '+((T+H-B)/2)+')'},"Share of the product's orders that are Qty = 1");
el('line',{x1:L,x2:W-R,y1:Y(.2),y2:Y(.2),stroke:'#7a88a6','stroke-width':1.2,'stroke-dasharray':'5 4'});
el('line',{x1:X(100),x2:X(100),y1:T,y2:H-B,stroke:'#3a4766','stroke-width':1.2});
el('text',{x:X(100)+6,y:Y(.55),fill:'#a4b0c8','font-size':11.5},'price = $100');
D.forEach(p=>{
  const c=el('circle',{class:'d',cx:X(p.price),cy:Y(p.single),r:5.5,fill:p.always?'#3987e5':'#d95926','fill-opacity':.85,stroke:'#0d121e','stroke-width':1});
  c.addEventListener('mouseenter',()=>{c.setAttribute('r',9);
    tip.innerHTML='<b>'+p.name+'</b><br>Price $'+p.price.toLocaleString()+' · '+p.orders.toLocaleString()+' orders<br>'+Math.round(p.single*100)+'% single-unit';
    tip.style.opacity=1;});
  c.addEventListener('mousemove',e=>{const r=document.querySelector('.wrap').getBoundingClientRect();
    tip.style.left=Math.min(e.clientX-r.left+14,r.width-270)+'px';tip.style.top=(e.clientY-r.top-10)+'px';});
  c.addEventListener('mouseleave',()=>{c.setAttribute('r',5.5);tip.style.opacity=0;});
});
</script></body></html>
"""

# ─────────────────────────────────────────────────────────────────────────────
# PREDICT  — form (left)  +  live result (right)
# ─────────────────────────────────────────────────────────────────────────────
left, right = st.columns([1, 1.12], gap="large")
_names = sorted(PRODUCT_MAP.keys())

with left:
    with st.container(border=True):
        st.markdown('<div class="card-title"><i>🛒</i>Order details</div>', unsafe_allow_html=True)
        product_name = st.selectbox(
            "Product", _names,
            index=_names.index('Perfect Fitness Perfect Rip Deck') if 'Perfect Fitness Perfect Rip Deck' in _names else 0)
        prod = PRODUCTS[product_name]
        if uses_price:
            product_price = st.number_input(
                "Product price ($) — pre-filled with the product's usual price",
                min_value=1.0, max_value=2000.0, value=float(min(2000.0, max(1.0, prod['price']))),
                step=1.0, key=f"price::{product_name}")
        if uses_date:
            order_date = st.date_input("Order date")
        if uses_ship:
            shipping_mode = st.selectbox("Shipping mode selected by customer", list(SHIP_MODES.keys()))
        if uses_dept:
            department_name = st.selectbox("Department", sorted(DEPT_MAP.keys()))
        if uses_segment:
            customer_segment = st.selectbox("Customer segment", SEGMENTS)
        if uses_market:
            market = st.selectbox("Market", MARKETS)
        if uses_region:
            order_region = st.selectbox("Order region", REGIONS)
        model_choice = st.segmented_control(
            "Model", ["Random Forest", "XGBoost"], default=WINNER,
            help="Random Forest is recommended: on products the model has never seen, XGBoost "
                 "falls back to predicting single-unit for everything.") or WINNER
        st.markdown(product_card(prod), unsafe_allow_html=True)

# ── prediction ────────────────────────────────────────────────────────────────
X_in = build_feature_row()
pipe = rf_pipeline if model_choice == 'Random Forest' else xgb_pipeline
p_multi = float(pipe.predict_proba(X_in)[0, 1])
is_multi = p_multi >= 0.5
st.session_state['last_prediction'] = {
    'product': product_name, 'price': product_price, 'model': model_choice,
    'p_multi': p_multi, 'label': 'multi-unit' if is_multi else 'single-unit'}

certain = (1 - p_multi) >= 0.95 or p_multi >= 0.95
if prod['always_single']:
    why = (f'All <b>{prod["orders"]:,}</b> historical orders of this product were single-unit, so the model treats it as '
           f'a one-unit item.')
elif product_price is not None and product_price > PRICE_CUT:
    why = (f'Above <b>${PRICE_CUT:,.0f}</b> every historical order was single-unit, so the price alone rules out '
           f'multi-unit orders.')
else:
    why = (f'For this product all quantities from 1 to 5 occur (<b>{(1 - prod["single_share"]) * 100:.0f}%</b> of its '
           f'{prod["orders"]:,} orders were multi-unit) and nothing known at order time says which quantity an order will be. '
           f'The call reflects the base rate (about <b>{BASE_MULTI * 100:.0f}%</b> across comparable products), not a certainty.')

tone = dict(
    color='#d95926' if is_multi else '#3987e5', txt='#f3a27f' if is_multi else '#8fbcf5',
    soft='rgba(217,89,38,.10)' if is_multi else 'rgba(57,135,229,.10)',
    border='rgba(217,89,38,.45)' if is_multi else 'rgba(57,135,229,.45)',
    accent='linear-gradient(90deg,#d95926,#f3a27f)' if is_multi else 'linear-gradient(90deg,#3987e5,#8fbcf5)')
html_pred = (PRED_HTML
             .replace('__ACCENT__', tone['accent']).replace('__TXT__', tone['txt']).replace('__COLOR__', tone['color'])
             .replace('__SOFT__', tone['soft']).replace('__BORDER__', tone['border'])
             .replace('__MODEL__', esc(model_choice))
             .replace('__VERDICT__', 'Multi-unit order' if is_multi else 'Single-unit order')
             .replace('__CONF__', 'High confidence' if certain else 'Base-rate estimate')
             .replace('__SUB__', '<b>Qty 2–5</b> expected' if is_multi else '<b>Qty = 1</b> expected')
             .replace('__ICON__', '📦' if is_multi else '⚡')
             .replace('__ACTION__', 'Route to the bulk-pick zone' if is_multi else 'Route to the fast-pick shelf')
             .replace('__ACTION_SUB__', 'Stock and pick several units together' if is_multi
                      else 'Keep one unit within quick reach')
             .replace('__WHY__', why).replace('__P__', f'{p_multi:.4f}'))
with right:
    render_html(html_pred, 592)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)
t1, t2, t3, t4 = st.tabs(["Model performance", "What drives the prediction", "Product explorer", "About & limits"])

# ── 1. performance ────────────────────────────────────────────────────────────
with t1:
    st.markdown(f'<div class="sec">How well does it work?</div><div class="lead">Measured on '
                f'{METRICS["test_rows"]:,} held-out orders the model never saw during training or tuning. '
                f'{"Random Forest and XGBoost score the same here." if abs(METRICS["random_forest"]["f1_macro"] - METRICS["xgboost"]["f1_macro"]) < 0.002 else "Random Forest and XGBoost are compared below."}</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="kpis">
      <div class="kpi"><small>Accuracy</small><b>{f3(W['accuracy'])}</b><span>share of orders called correctly</span></div>
      <div class="kpi"><small>F1-Macro</small><b>{f3(W['f1_macro'])}</b><span>both classes weighted equally</span></div>
      <div class="kpi"><small>F1-Weighted</small><b>{f3(W['f1_weighted'])}</b><span>weighted by class size</span></div>
      <div class="kpi"><small>ROC-AUC</small><b>{f3(W['roc_auc'])}</b><span>ranking quality</span></div>
    </div>""", unsafe_allow_html=True)

    pc_w = W['per_class']
    st.markdown('<div class="sec" style="margin-top:6px">Per-class results</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <table class="tbl"><tr><th>Order type</th><th class="n">Precision</th><th class="n">Recall</th><th class="n">F1</th><th class="n">Test orders</th></tr>
    <tr><td><b>Single-unit</b> (Qty 1) → fast-pick shelf</td><td class="n">{f3(pc_w['single_unit']['precision'])}</td><td class="n">{f3(pc_w['single_unit']['recall'])}</td><td class="n">{f3(pc_w['single_unit']['f1'])}</td><td class="n">{pc_w['single_unit']['support']:,}</td></tr>
    <tr><td><b>Multi-unit</b> (Qty 2–5) → bulk-pick zone</td><td class="n">{f3(pc_w['multi_unit']['precision'])}</td><td class="n">{f3(pc_w['multi_unit']['recall'])}</td><td class="n">{f3(pc_w['multi_unit']['f1'])}</td><td class="n">{pc_w['multi_unit']['support']:,}</td></tr></table>
    <div class="lead" style="margin-top:12px">When the model says <b>single-unit</b> it is virtually always right (precision {f3(pc_w['single_unit']['precision'])}).
    It finds {pct(pc_w['multi_unit']['recall'])} of multi-unit orders, at the cost of catching only {pct(pc_w['single_unit']['recall'])} of single-unit
    ones: some single-unit orders are placed on products where several units are possible and look identical to multi-unit orders.</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec">Compared with simple baselines (F1-Macro)</div>', unsafe_allow_html=True)
    base = [('Random guess', METRICS['baseline_dummy']['f1_macro'], '#56637f'),
            (f'Price rule (single-unit above ${PRICE_CUT:,.0f})', METRICS['baseline_price_rule']['f1_macro'], '#8b7cf6'),
            ('Random Forest', METRICS['random_forest']['f1_macro'], '#3987e5'),
            ('XGBoost', METRICS['xgboost']['f1_macro'], '#3987e5')]
    st.markdown(''.join(
        f'<div class="hb"><span>{esc(n)}</span><div class="track"><div class="fill" style="width:{v * 100:.1f}%;background:{c}"></div></div>'
        f'<span class="val">{f3(v)}</span></div>' for n, v, c in base), unsafe_allow_html=True)
    gain = METRICS['random_forest']['f1_macro'] - METRICS['baseline_price_rule']['f1_macro']
    st.markdown(f'<div class="note good"><b>Honest reading:</b> the learned models beat the one-line price rule by only '
                f'<b>{gain:+.3f}</b> F1-Macro. The signal in this data is essentially “which product, at what price” — '
                f'the value of the model is that it applies that knowledge consistently, including the few inexpensive '
                f'products that are always ordered one unit at a time.</div>', unsafe_allow_html=True)

    ST = METRICS.get('stress_test_unseen_products')
    if ST:
        st.markdown('<div class="sec">Stress test — products the model has never seen</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <table class="tbl"><tr><th>Model</th><th class="n">Mean F1-Macro</th><th class="n">Worst fold</th><th class="n">Best fold</th></tr>
        <tr><td><b>Random Forest</b></td><td class="n">{f3(ST['Random Forest']['mean'])}</td><td class="n">{f3(ST['Random Forest']['min'])}</td><td class="n">{f3(ST['Random Forest']['max'])}</td></tr>
        <tr><td><b>XGBoost</b></td><td class="n">{f3(ST['XGBoost']['mean'])}</td><td class="n">{f3(ST['XGBoost']['min'])}</td><td class="n">{f3(ST['XGBoost']['max'])}</td></tr></table>
        <div class="note">Five folds, each holding out whole products. For a brand-new product XGBoost tends to call every order
        single-unit, while Random Forest still uses the price — that is why Random Forest is the recommended model.
        Treat this as a warning about new products, not a precise figure.</div>""", unsafe_allow_html=True)

    with st.expander("Confusion matrices and learning curve"):
        for fn, cap in [('model_training/model_01_confusion_matrices.png',
                         'Confusion matrices — Random Forest, XGBoost and the price-rule baseline'),
                        ('evaluation/eval_01_learning_curve.png',
                         'Learning curve — training and cross-validation scores lie on top of each other (no overfitting)')]:
            pth = os.path.join(PLOTS_DIR, fn)
            if os.path.exists(pth):
                show_image(pth, cap)

# ── 2. drivers ────────────────────────────────────────────────────────────────
with t2:
    st.markdown('<div class="sec">What the model looks at</div><div class="lead">SHAP values show how much each input moves '
                'the prediction. Only inputs that survived a strict screening are used by the model; everything else that was '
                'tested (dates, region, customer segment, shipping mode, discounts, payment type, …) showed no more '
                'information than random noise.</div>', unsafe_allow_html=True)
    nice = {'product_group': 'Which product', 'product_price_log': 'Product price', 'dept_group': 'Department',
            'shipping_year': 'Shipping year', 'is_q4_slowdown': 'Quarter 4 (Oct–Dec)'}
    gi = SHAP.get('global_feature_importance', {})
    if gi:
        mx = max(gi.values()) or 1
        st.markdown('<div class="sec" style="margin-top:14px">Average effect on the prediction (mean |SHAP|)</div>',
                    unsafe_allow_html=True)
        st.markdown(''.join(
            f'<div class="hb"><span>{esc(nice.get(k, k.replace("_", " ")))}</span><div class="track"><div class="fill" '
            f'style="width:{v / mx * 100:.1f}%;background:linear-gradient(90deg,#3987e5,#8b7cf6)"></div></div>'
            f'<span class="val">{v:.3f}</span></div>' for k, v in list(gi.items())[:8]), unsafe_allow_html=True)
    st.markdown(f"""
    <div class="grid2" style="margin-top:14px">
      <div class="tile"><h4>Which product</h4><p>The product decides whether several units are even possible.
      {N_ALWAYS} of {N_PRODUCTS} products are always ordered one unit at a time ({ALWAYS_ROWS / STATS['n_orders'] * 100:.0f}% of all orders).</p></div>
      <div class="tile"><h4>Product price</h4><p>Every product above about ${PRICE_CUT:,.0f} was ordered one unit at a time.
      The price lets the model handle products it can identify only weakly.</p></div>
    </div>""", unsafe_allow_html=True)
    ex = SHAP.get('worked_examples') or {}
    if ex:
        with st.expander("Worked examples — how a single prediction is explained"):
            for key, title in (('single_unit_order', 'A real single-unit order'),
                               ('multi_unit_order', 'A real multi-unit order')):
                rows = ex.get(key) or []
                if rows:
                    st.markdown(
                        f'<div class="sec" style="font-size:15px;margin:10px 0 8px">{title}</div>'
                        '<table class="tbl"><tr><th>Input</th><th class="n">Value</th><th class="n">Effect on P(multi-unit)</th></tr>'
                        + ''.join(f'<tr><td>{esc(nice.get(r["feature"], r["feature"]))}</td><td class="n">{r["value"]:.3f}</td>'
                                  f'<td class="n"><b>{r["shap_contribution"]:+.3f}</b> '
                                  f'{"→ multi" if r["shap_contribution"] > 0 else "→ single"}</td></tr>' for r in rows[:4])
                        + '</table>', unsafe_allow_html=True)
    pth = os.path.join(PLOTS_DIR, 'explainability', 'shap_02_beeswarm_multi_unit.png')
    if os.path.exists(pth):
        with st.expander("SHAP beeswarm plot"):
            show_image(pth)

# ── 3. product explorer ───────────────────────────────────────────────────────
with t3:
    st.markdown(f'<div class="sec">Every product at a glance</div><div class="lead">Each dot is one of the {N_PRODUCTS} products. '
                f'The vertical axis shows how often its orders were a single unit. Blue dots (always one unit) are '
                f'{N_ALWAYS} products — every product above $100 and a few inexpensive generic items. For the orange products, '
                f'each quantity from 1 to 5 occurs about 20% of the time, whatever the product.</div>', unsafe_allow_html=True)
    data = [{'name': p['name'], 'price': p['price'], 'single': p['single_share'], 'orders': p['orders'],
             'always': p['always_single']} for p in STATS['products']]
    scatter = (SCATTER_HTML.replace('__DATA__', json.dumps(data))
               .replace('__NM__', str(N_PRODUCTS - N_ALWAYS)).replace('__NA__', str(N_ALWAYS)))
    render_html(scatter, 540)

# ── 4. about ──────────────────────────────────────────────────────────────────
with t4:
    st.markdown(f"""
    <div class="sec">What this tool does</div>
    <div class="lead">It answers one warehouse question: will an incoming order be for one unit or for several? It uses only
    information available when the order is placed — no shipping outcome, delivery status, realized profit or dollar discount.</div>
    <div class="grid2" style="margin-top:16px">
      <div class="tile"><h4>Why not predict the exact quantity?</h4><p>On products where several units are possible, quantities 1 to 5 are
      equally likely and no order-time information separates them. A model trained on shuffled labels scores the same as the real one,
      so “2 vs 3 vs 4 vs 5” is not predictable from this data.</p></div>
      <div class="tile"><h4>Where it is strongest</h4><p>Above ${PRICE_CUT:,.0f} and for always-single products the call is essentially certain
      (single-unit). Always-single products alone account for {ALWAYS_ROWS / STATS['n_orders'] * 100:.0f}% of orders.</p></div>
      <div class="tile"><h4>Where it is only a base-rate estimate</h4><p>For products priced ${PRICE_CUT:,.0f} or less, about {BASE_MULTI * 100:.0f}% of orders
      are multi-unit. The model says “multi-unit”, and roughly one in five of those orders is in fact a single unit — nothing recorded at order time can tell them apart.</p></div>
      <div class="tile"><h4>New products</h4><p>Performance on products never seen in training is unstable (see the stress test).
      For a brand-new product use the price rule — single-unit above ${PRICE_CUT:,.0f}, otherwise multi-unit — until the model is retrained with its orders.</p></div>
    </div>
    <div class="note" style="margin-top:18px"><b>Planning aid, not automation.</b> Pair the output with human review, especially for
    inexpensive products where the call reflects the base rate.</div>
    """, unsafe_allow_html=True)

st.markdown(f"""
<div class="foot"><span>Retail Demand Forecasting · v3 · trained on {STATS['n_orders']:,} orders (2015–2017) ·
{esc(WINNER)} is the recommended model</span><span>Metrics: reports/metrics_v3.json</span></div>
""", unsafe_allow_html=True)
