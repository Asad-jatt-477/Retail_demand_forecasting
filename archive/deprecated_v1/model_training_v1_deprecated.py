# =============================================================================
#  RETAIL DEMAND FORECASTING — MODELING PHASE
#  Phase 10: Random Forest + XGBoost (Multi-Class Classification)
#  Input  : Retail_demand_FS.csv          (178,396 rows × 23 columns)
#  Output : Trained models + evaluation report + plots
#  Target : order_item_quantity  (Classes: 1, 2, 3, 4, 5)
# =============================================================================
#
#  PROBLEM TYPE
#  ─────────────────────────────────────────────────────────────────────────────
#  Yeh ek MULTI-CLASS CLASSIFICATION problem hai (5 classes: qty = 1,2,3,4,5),
#  regression nahi. Har order-item ke liye hum predict kar rahe hain "kitni
#  quantity order hogi" — ek discrete category ke roop mein, continuous number
#  ke roop mein nahi. Isi liye ROC curves, PR curves, confusion matrix sab
#  multi-class (One-vs-Rest) tarz mein banengey.
#
#  REAL-WORLD RETAIL ANGLE
#  ─────────────────────────────────────────────────────────────────────────────
#  Yeh model warehouse/inventory planning ke liye use hoga: agar hum pehle se
#  predict kar lein ke ek incoming order kis quantity-class mein girega, to
#  • Single-unit orders (qty=1, 54% data) → fast-pick shelving
#  • Bulk orders (qty=4,5) → pallet-level / bulk-zone picking
#  Misclassification ki cost SYMMETRIC nahi hai: ek bulk order (qty=5) ko
#  qty=1 samajh lena → stockout + delayed shipment (bahut costly).
#  Isi wajah se hum sirf "accuracy" par nahi, balke F1-MACRO + per-class
#  Recall par focus karein gey — taake minority bulk-classes bhi sahi
#  pakdi jayein, sirf majority class (qty=1) achi predict ho ke overall
#  accuracy high na dikhe.
#
#  ISSUES CARRIED OVER FROM FEATURE ENGINEERING / FEATURE SELECTION
#  (jo humne us waqt promise kiya tha "Modeling phase mein fix karein gey")
#  ─────────────────────────────────────────────────────────────────────────────
#  1. DATA LEAKAGE — product_avg_qty & dept_avg_qty Group B mein FULL dataset
#     par target-encode hue thay (groupby(target).mean() saare data par).
#     Fix: in dono columns ko sirf training-fold data se, K-Fold ke andar,
#     dobara (leak-free) encode karein gey — sklearn Pipeline ke andar, taake
#     hyperparameter tuning ke har fold mein bhi koi leakage na ho.
#  2. CLASS IMBALANCE — qty=1 is 54.4% of data, baaki 4 classes ~11.4% each.
#     Fix: class_weight='balanced' (Random Forest) + balanced sample_weight
#     (XGBoost) — taake model sirf majority class ki taraf bias na ho.
#
#  MODELING PLAN
#  ─────────────────────────────────────────────────────────────────────────────
#  STEP 0  : Load FS dataset
#  STEP 1  : Fix Data Leakage — recover safe group-identity from leaky columns
#  STEP 2  : Stratified Train/Test Split
#  STEP 3  : Baseline (DummyClassifier) — F1-Macro floor jo beat karna hai
#  STEP 4  : Leakage-Safe K-Fold Target Encoder (custom transformer)
#  STEP 5  : Build Pipelines — Random Forest & XGBoost (class-imbalance aware)
#  STEP 6  : Hyperparameter Tuning — RandomizedSearchCV (StratifiedKFold-5, F1-Macro)
#  STEP 7  : Final Fit + Test-Set Evaluation (classification report, confusion matrix)
#  STEP 8  : ROC Curves (multi-class, One-vs-Rest)
#  STEP 9  : Precision-Recall Curves (multi-class, One-vs-Rest)
#  STEP 10 : Feature Importance — both models
#  STEP 11 : Model Comparison + Business Recommendation
#  STEP 12 : Save trained models
#
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import pickle
import time

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score,
    roc_curve, auc, precision_recall_curve, average_precision_score,
)
from xgboost import XGBClassifier

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

RANDOM_STATE = 42
TARGET = 'order_item_quantity'
CLASSES = [1, 2, 3, 4, 5]

PLOTS_DIR = 'Modeling_Plots'
MODELS_DIR = 'Saved_Models'
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
print(f"📁 Output folders ready: '{PLOTS_DIR}/' , '{MODELS_DIR}/'")

# === RUN-SPEED CONFIG ========================================================
# Aapke machine ke hisaab se yeh values adjust kar sakte hain.
# N_ITER  = RandomizedSearchCV kitne random hyperparameter combos try karega
# CV_FOLDS = kitne cross-validation folds (tuning ke andar)
# Bigger values = better tuned model, lekin zyada time lagega.
N_ITER   = 25
CV_FOLDS = 5
# =============================================================================


# =============================================================================
# STEP 0: LOAD DATASET
# =============================================================================
print("\n" + "="*70)
print("STEP 0: LOAD DATASET")
print("="*70)

df = pd.read_csv('Retail_demand_FS.csv')
print(f"\n  Dataset loaded : {df.shape[0]:,} rows × {df.shape[1]} columns")

class_dist = df[TARGET].value_counts().sort_index()
print(f"\n  Target distribution (order_item_quantity):")
for cls, cnt in class_dist.items():
    pct = cnt / len(df) * 100
    bar = "█" * int(pct / 2)
    print(f"    Qty={cls}  {cnt:>7,}  ({pct:5.1f}%)  {bar}")
print(f"\n  ⚠️  Class imbalance confirmed: Qty=1 alone is "
      f"{class_dist[1]/len(df)*100:.1f}% of all orders.")


# =============================================================================
# STEP 1: FIX DATA LEAKAGE — RECOVER SAFE GROUP IDENTITY
# =============================================================================
print("\n" + "="*70)
print("STEP 1: FIX DATA LEAKAGE (product_avg_qty, dept_avg_qty)")
print("="*70)
print("""
  PROBLEM:
  product_avg_qty aur dept_avg_qty Feature Engineering phase mein poore
  dataset par target-encode hue thay (df.groupby('product_name')[target].mean()).
  Iska matlab har row ke andar, indirectly, TEST SET ke targets ka asar bhi
  shamil hai — yeh CLASSIC DATA LEAKAGE hai.

  CHALLENGE:
  Original categorical columns (product_name, department_name) Feature
  Engineering ke baad hi DROP ho chuke hain — ab raw category wapas nahi mil
  sakti.

  SOLUTION:
  Target encoding ek DETERMINISTIC 1:1 mapping hai — har product ki ek hi
  fixed avg_qty value hai. Iska matlab product_avg_qty ki har UNIQUE VALUE
  khud ek "product ID" ka kaam de rahi hai (group identity), sirf uski
  LEAKED VALUE use nahi karni. Hum neeche verify bhi karte hain ke yeh
  grouping bilkul exact hai (round-trip mean check).
""")

# Verify exact group integrity before relying on it
check_prod = df.groupby('product_avg_qty')[TARGET].mean()
max_diff_prod = (check_prod - check_prod.index.to_series()).abs().max()
check_dept = df.groupby('dept_avg_qty')[TARGET].mean()
max_diff_dept = (check_dept - check_dept.index.to_series()).abs().max()
print(f"  Integrity check → product_avg_qty max diff: {max_diff_prod:.2e}  "
      f"(should be ~0)")
print(f"  Integrity check → dept_avg_qty    max diff: {max_diff_dept:.2e}  "
      f"(should be ~0)")
assert max_diff_prod < 1e-6 and max_diff_dept < 1e-6, \
    "Group identity assumption violated — do not proceed with this shortcut."

# Recover GROUP IDENTITY only (not the leaked value itself)
df['product_group'] = pd.factorize(df['product_avg_qty'])[0]
df['dept_group']    = pd.factorize(df['dept_avg_qty'])[0]

n_prod_groups = df['product_group'].nunique()
n_dept_groups = df['dept_group'].nunique()
print(f"\n  ✓ Recovered {n_prod_groups} product groups, {n_dept_groups} dept groups")
print(f"    (Group MEMBERSHIP kept, LEAKED VALUE dropped — re-encoding will")
print(f"     happen safely, inside the pipeline, using training data only.)")

# Drop the leaked numeric columns — they will be safely re-built in the pipeline
df.drop(columns=['product_avg_qty', 'dept_avg_qty'], inplace=True)

LEAKY_GROUP_COLS = ['product_group', 'dept_group']
print(f"\n  Dropped leaked columns: product_avg_qty, dept_avg_qty")
print(f"  Will be safely re-encoded from: {LEAKY_GROUP_COLS}")


# =============================================================================
# STEP 2: STRATIFIED TRAIN / TEST SPLIT
# =============================================================================
print("\n" + "="*70)
print("STEP 2: STRATIFIED TRAIN / TEST SPLIT")
print("="*70)
print("""
  WHY STRATIFIED: Qty=1 54% vs baaki classes ~11% each — agar random split
  karein to chance hai test set mein minority classes (qty=4,5) ka proportion
  bigad jaye. stratify=y se guarantee milta hai ke train aur test set dono
  mein class-ratio EXACTLY same rahega jaisa poore dataset mein hai.
""")

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)

print(f"  Train set : {X_train.shape[0]:,} rows")
print(f"  Test  set : {X_test.shape[0]:,} rows")
print(f"\n  Class ratio check (Train vs Test):")
train_pct = y_train.value_counts(normalize=True).sort_index() * 100
test_pct  = y_test.value_counts(normalize=True).sort_index() * 100
for cls in CLASSES:
    print(f"    Qty={cls}  Train={train_pct[cls]:5.1f}%   Test={test_pct[cls]:5.1f}%")

# XGBoost requires 0-indexed contiguous class labels (1..5 → 0..4)
label_enc = LabelEncoder()
label_enc.fit(CLASSES)
y_train_xgb = label_enc.transform(y_train)
y_test_xgb  = label_enc.transform(y_test)


# =============================================================================
# STEP 3: BASELINE MODEL — DummyClassifier
# =============================================================================
print("\n" + "="*70)
print("STEP 3: BASELINE MODEL (DummyClassifier)")
print("="*70)
print("""
  Baseline yeh batata hai ke "kuch na seekhne wala" model kya score deta hai.
  Hamare dono real models (RF, XGBoost) ko isse SIGNIFICANTLY behtar hona
  chahiye, warna unka koi business value nahi.
""")

dummy = DummyClassifier(strategy='stratified', random_state=RANDOM_STATE)
dummy.fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)
dummy_f1 = f1_score(y_test, dummy_pred, average='macro')
dummy_acc = accuracy_score(y_test, dummy_pred)
print(f"  Baseline (stratified-random guess)  →  F1-Macro: {dummy_f1:.4f}  |  "
      f"Accuracy: {dummy_acc:.4f}")
print(f"  TARGET FOR OUR MODELS: F1-Macro > 0.50 (minimum), > 0.65 (ideal)")


# =============================================================================
# STEP 4: LEAKAGE-SAFE K-FOLD TARGET ENCODER (custom transformer)
# =============================================================================
print("\n" + "="*70)
print("STEP 4: BUILD LEAKAGE-SAFE TARGET ENCODER")
print("="*70)

class KFoldTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Group-identity columns (product_group, dept_group) ko unki HISTORICAL
    AVERAGE target-value se encode karta hai — bilkul wahi idea jo Feature
    Engineering mein tha, lekin ab LEAKAGE-FREE tareeqe se:

    • fit(X, y)        → poore X par group→mean(target) mapping seekhta hai.
                          Yeh mapping baad mein TEST/VALIDATION set par
                          transform() ke liye use hoga (test ka koi bhi
                          target is mapping mein shamil nahi hota).

    • fit_transform()  → SIRF training set par chalta hai. Andar StratifiedKFold
                          use karke OUT-OF-FOLD encoding banata hai: har row
                          ki encoded value sirf UN folds se aati hai jin mein
                          woh row khud shamil NAHI thi. Isse training set ke
                          andar bhi row apna khud ka target indirectly nahi
                          dekh pati — chhota lekin real wala leakage bhi avoid
                          ho jata hai.

    • transform(X)     → naye/unseen data (test set, production) ke liye —
                          fit() se seekhi hui mapping use hoti hai. Naye/rare
                          group jo training mein dikhe hi nahi, unko global
                          mean mil jata hai (fallback).

    SMOOTHING: Bayesian smoothing — chhote groups (kam orders waale products)
    ki estimate ko global mean ki taraf khinchta hai, taake noisy/overfit
    averages na banein.
    """

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
        X_orig = X.reset_index(drop=True).copy()
        y_ser = pd.Series(np.asarray(y)).reset_index(drop=True)

        # 1) Learn the "production" mapping on the FULL training set
        #    (this is what later transform() calls — e.g. on the test set — will use)
        self.fit(X_orig, y_ser)

        # 2) Build OUT-OF-FOLD encoding for the training rows themselves
        X_out = X_orig.copy()
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                               random_state=self.random_state)
        for col in self.cols:
            oof = np.zeros(len(X_orig))
            for tr_idx, val_idx in skf.split(X_orig, y_ser):
                smoothed, gmean = self._smoothed_means(
                    X_orig[col].iloc[tr_idx], y_ser.iloc[tr_idx]
                )
                oof[val_idx] = X_orig[col].iloc[val_idx].map(smoothed).fillna(gmean).values
            X_out[col] = oof
        return X_out


print(f"""
  ✓ KFoldTargetEncoder class ready.
    cols to encode : {LEAKY_GROUP_COLS}
    n_splits       : 5 (out-of-fold, training set ke andar bhi leak-free)
    smoothing      : 10 (chhote groups global mean ki taraf khinchengey)

  Yeh transformer dono models (RF, XGBoost) ki Pipeline ka PEHLA step banega,
  taake RandomizedSearchCV ke HAR fold mein bhi encoding sirf us fold ke
  training-portion se ho — hyperparameter tuning ke dauran bhi koi leakage
  nahi hogi.
""")


# =============================================================================
# STEP 5: CLASS IMBALANCE STRATEGY
# =============================================================================
print("\n" + "="*70)
print("STEP 5: CLASS IMBALANCE HANDLING")
print("="*70)
print("""
  Random Forest → class_weight='balanced'
    sklearn khud, har .fit() call par (har CV fold ke training-subset par),
    class frequencies dekh kar weights recompute karta hai: rare classes
    (qty=4,5) ko zyada weight, common class (qty=1) ko kam weight.

  XGBoost → sample_weight (compute_sample_weight 'balanced')
    XGBoost ke multi-class objective mein direct class_weight parameter
    nahi hota, isliye hum sklearn ka compute_sample_weight use karke
    per-row weight banatay hain aur Pipeline.fit() ko fit_params ke
    through pass karte hain. sklearn ki CV machinery is weight-array ko
    automatically har fold ke train-indices ke hisaab se slice kar deti hai.
""")

sample_weights_train = compute_sample_weight(class_weight='balanced', y=y_train)
print(f"  Sample weights computed for {len(sample_weights_train):,} training rows.")
print(f"  (Used only by the XGBoost pipeline; Random Forest handles this internally.)")


# =============================================================================
# STEP 6: BUILD PIPELINES — RANDOM FOREST & XGBOOST
# =============================================================================
print("\n" + "="*70)
print("STEP 6: BUILD MODELING PIPELINES")
print("="*70)

rf_pipeline = Pipeline([
    ('target_enc', KFoldTargetEncoder(cols=LEAKY_GROUP_COLS, n_splits=5,
                                       smoothing=10, random_state=RANDOM_STATE)),
    ('rf', RandomForestClassifier(class_weight='balanced',
                                   random_state=RANDOM_STATE, n_jobs=-1)),
])

xgb_pipeline = Pipeline([
    ('target_enc', KFoldTargetEncoder(cols=LEAKY_GROUP_COLS, n_splits=5,
                                       smoothing=10, random_state=RANDOM_STATE)),
    ('xgb', XGBClassifier(objective='multi:softprob', num_class=len(CLASSES),
                           eval_metric='mlogloss', random_state=RANDOM_STATE,
                           n_jobs=-1)),
])

print("  ✓ Random Forest pipeline : target_enc → RandomForestClassifier(class_weight='balanced')")
print("  ✓ XGBoost      pipeline : target_enc → XGBClassifier(sample_weight at fit time)")

# Hyperparameter search spaces — tree-based models, non-normal data (confirmed in EDA/FE)
rf_param_dist = {
    'rf__n_estimators':      [200, 300, 400, 500, 600],
    'rf__max_depth':         [8, 12, 16, 20, None],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf':  [1, 2, 4],         
    'rf__max_features':      ['sqrt', 'log2'],
}

xgb_param_dist = {
    'xgb__n_estimators':     [200, 300, 400, 500],
    'xgb__max_depth':        [3, 4, 5, 6, 8],
    'xgb__learning_rate':    [0.01, 0.03, 0.05, 0.1],
    'xgb__subsample':        [0.7, 0.8, 0.9, 1.0],
    'xgb__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'xgb__min_child_weight': [1, 3, 5],
}


# =============================================================================
# STEP 7: HYPERPARAMETER TUNING — RandomizedSearchCV
# =============================================================================
print("\n" + "="*70)
print(f"STEP 7: HYPERPARAMETER TUNING (RandomizedSearchCV, n_iter={N_ITER}, "
      f"cv={CV_FOLDS}, scoring=F1-Macro)")
print("="*70)
print("""
  scoring='f1_macro' chuna gaya hai (na ke 'accuracy') kyunke dataset
  imbalanced hai — accuracy sirf qty=1 sahi predict karke bhi high dikh
  sakti hai jabke business-critical bulk classes (qty=4,5) bilkul miss ho
  rahi hon. F1-Macro har class ko BARABAR importance deta hai.
""")

cv_strategy = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# ── Random Forest tuning ──────────────────────────────────────────────────────
print("\n  → Tuning Random Forest ...")
t0 = time.time()
rf_search = RandomizedSearchCV(
    rf_pipeline, param_distributions=rf_param_dist, n_iter=N_ITER,
    scoring='f1_macro', cv=cv_strategy, random_state=RANDOM_STATE,
    n_jobs=-1, verbose=1, refit=True,
)
rf_search.fit(X_train, y_train)
print(f"  ✓ Random Forest tuned in {time.time()-t0:.1f}s")
print(f"    Best CV F1-Macro : {rf_search.best_score_:.4f}")
print(f"    Best params      : {rf_search.best_params_}")

# ── XGBoost tuning ─────────────────────────────────────────────────────────────
print("\n  → Tuning XGBoost ...")
t0 = time.time()
xgb_search = RandomizedSearchCV(
    xgb_pipeline, param_distributions=xgb_param_dist, n_iter=N_ITER,
    scoring='f1_macro', cv=cv_strategy, random_state=RANDOM_STATE,
    n_jobs=-1, verbose=1, refit=True,
)
xgb_search.fit(X_train, y_train_xgb, xgb__sample_weight=sample_weights_train)
print(f"  ✓ XGBoost tuned in {time.time()-t0:.1f}s")
print(f"    Best CV F1-Macro : {xgb_search.best_score_:.4f}")
print(f"    Best params      : {xgb_search.best_params_}")

best_rf  = rf_search.best_estimator_
best_xgb = xgb_search.best_estimator_


# =============================================================================
# STEP 8: TEST-SET EVALUATION
# =============================================================================
print("\n" + "="*70)
print("STEP 8: TEST-SET EVALUATION")
print("="*70)

rf_pred  = best_rf.predict(X_test)
xgb_pred_enc = best_xgb.predict(X_test)
xgb_pred = label_enc.inverse_transform(xgb_pred_enc)          # back to 1..5

rf_proba  = best_rf.predict_proba(X_test)                     # columns already in 1..5 order (rf.classes_)
xgb_proba = best_xgb.predict_proba(X_test)                    # columns in 0..4 order → need realignment

# Make sure probability column order is [class1, class2, class3, class4, class5] for BOTH models
rf_class_order = list(best_rf.named_steps['rf'].classes_)
rf_proba = rf_proba[:, [rf_class_order.index(c) for c in CLASSES]]

xgb_class_order = list(label_enc.inverse_transform(best_xgb.named_steps['xgb'].classes_))
xgb_proba = xgb_proba[:, [xgb_class_order.index(c) for c in CLASSES]]

results = {}
for name, pred in [('Random Forest', rf_pred), ('XGBoost', xgb_pred)]:
    f1m = f1_score(y_test, pred, average='macro')
    f1w = f1_score(y_test, pred, average='weighted')
    acc = accuracy_score(y_test, pred)
    results[name] = {'f1_macro': f1m, 'f1_weighted': f1w, 'accuracy': acc}
    print(f"\n  ── {name} ──")
    print(f"    Accuracy     : {acc:.4f}")
    print(f"    F1-Macro     : {f1m:.4f}")
    print(f"    F1-Weighted  : {f1w:.4f}")
    print(f"\n{classification_report(y_test, pred, target_names=[f'Qty={c}' for c in CLASSES])}")

# ── Confusion Matrices ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, (name, pred) in zip(axes, [('Random Forest', rf_pred), ('XGBoost', xgb_pred)]):
    cm = confusion_matrix(y_test, pred, labels=CLASSES)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    sns.heatmap(cm_pct, annot=cm, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=[f'Qty={c}' for c in CLASSES],
                yticklabels=[f'Qty={c}' for c in CLASSES],
                annot_kws={'fontsize': 11, 'fontweight': 'bold'})
    ax.set_title(f'{name} — Confusion Matrix\n(counts shown, color = row %)', fontsize=11)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'model_01_confusion_matrices.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f"\n  Saved: {PLOTS_DIR}/model_01_confusion_matrices.png")


# =============================================================================
# STEP 9: ROC CURVES (Multi-Class, One-vs-Rest)
# =============================================================================
print("\n" + "="*70)
print("STEP 9: ROC CURVES (One-vs-Rest)")
print("="*70)
print("""
  Multi-class problem mein ek single ROC curve nahi banta. Har class ke liye
  alag "is class vs baaki sab" (One-vs-Rest) ROC curve banta hai — class ka
  predicted probability vs threshold sweep. AUC jitna 1.0 ke kareeb, model
  us class ko utna behtar pehchanta hai.
""")

y_test_bin = label_binarize(y_test, classes=CLASSES)

fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
for ax, (name, proba) in zip(axes, [('Random Forest', rf_proba), ('XGBoost', xgb_proba)]):
    for i, cls in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=qty_colors[cls], lw=2,
                label=f'Qty={cls} (AUC={roc_auc:.3f})')
    # micro-average
    fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), proba.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    ax.plot(fpr_micro, tpr_micro, color=palette['primary'], lw=2.5, linestyle='--',
            label=f'Micro-avg (AUC={auc_micro:.3f})')
    ax.plot([0, 1], [0, 1], color='grey', lw=1, linestyle=':')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{name} — ROC Curves (One-vs-Rest)', fontsize=11)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'model_02_roc_curves.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f"  Saved: {PLOTS_DIR}/model_02_roc_curves.png")


# =============================================================================
# STEP 10: PRECISION-RECALL CURVES (Multi-Class, One-vs-Rest)
# =============================================================================
print("\n" + "="*70)
print("STEP 10: PRECISION-RECALL CURVES (One-vs-Rest)")
print("="*70)
print("""
  Imbalanced data mein PR curves ROC se zyada informative hote hain —
  khaaskar minority classes (qty=4,5) ke liye, kyunke PR curve sirf
  POSITIVE class par focus karta hai (jabke ROC negative class ki
  abundance se thoda "optimistic" ho sakta hai).
""")

fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
for ax, (name, proba) in zip(axes, [('Random Forest', rf_proba), ('XGBoost', xgb_proba)]):
    for i, cls in enumerate(CLASSES):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], proba[:, i])
        ap = average_precision_score(y_test_bin[:, i], proba[:, i])
        ax.plot(recall, precision, color=qty_colors[cls], lw=2,
                label=f'Qty={cls} (AP={ap:.3f})')
    baseline = y_test_bin.sum(axis=0) / len(y_test_bin)  # per-class positive rate, just for reference
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'{name} — Precision-Recall Curves (One-vs-Rest)', fontsize=11)
    ax.legend(loc='lower left', fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'model_03_pr_curves.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f"  Saved: {PLOTS_DIR}/model_03_pr_curves.png")


# =============================================================================
# STEP 11: FEATURE IMPORTANCE
# =============================================================================
print("\n" + "="*70)
print("STEP 11: FEATURE IMPORTANCE")
print("="*70)

feature_names = list(X_train.columns)

rf_importances  = best_rf.named_steps['rf'].feature_importances_
xgb_importances = best_xgb.named_steps['xgb'].feature_importances_

fig, axes = plt.subplots(1, 2, figsize=(15, 7))
for ax, (name, importances) in zip(axes, [('Random Forest', rf_importances),
                                            ('XGBoost', xgb_importances)]):
    imp_series = pd.Series(importances, index=feature_names).sort_values(ascending=True)
    colors = [palette['teal'] if 'group' not in f else palette['warn'] for f in imp_series.index]
    ax.barh(imp_series.index, imp_series.values, color=colors, edgecolor='white')
    ax.set_title(f'{name} — Feature Importance', fontsize=11)
    ax.set_xlabel('Importance')
    ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'model_04_feature_importance.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f"  Saved: {PLOTS_DIR}/model_04_feature_importance.png")
print(f"\n  Note (orange bars) : product_group / dept_group — yeh humare leakage-fixed")
print(f"  target-encoded features hain, ab bhi top predictors hain (jaisa FE phase mein tha)")
print(f"  lekin ab unki value har CV fold mein sirf training data se aati hai.")


# =============================================================================
# STEP 12: MODEL COMPARISON + BUSINESS RECOMMENDATION
# =============================================================================
print("\n" + "="*70)
print("STEP 12: FINAL MODEL COMPARISON")
print("="*70)

comparison_df = pd.DataFrame(results).T
comparison_df.loc['Baseline (Dummy)'] = {'f1_macro': dummy_f1, 'f1_weighted': np.nan, 'accuracy': dummy_acc}
comparison_df = comparison_df[['accuracy', 'f1_macro', 'f1_weighted']].round(4)
print(f"\n{comparison_df.to_string()}\n")

best_model_name = 'Random Forest' if results['Random Forest']['f1_macro'] >= results['XGBoost']['f1_macro'] else 'XGBoost'
best_model_obj  = best_rf if best_model_name == 'Random Forest' else best_xgb

print(f"""
  WINNER (by F1-Macro)  : {best_model_name}
  F1-Macro improvement vs baseline : +{results[best_model_name]['f1_macro'] - dummy_f1:.4f}

  BUSINESS RECOMMENDATION:
  • {best_model_name} ko production demand-forecasting pipeline mein deploy karein.
  • Per-class recall (Qty=4, Qty=5) ko monitor karte rahein — agar yeh kam
    rahe to bulk-order stockouts ka risk badh jata hai. Confusion matrix
    aur PR-curve (AP score) regularly track karein.
  • Model ko quarterly retrain karein — naye products/departments aane par
    target-encoding mapping (KFoldTargetEncoder) automatically update ho
    jayegi jab pipeline.fit() dobara chalega.
""")


# =============================================================================
# STEP 13: SAVE TRAINED MODELS
# =============================================================================
print("="*70)
print("STEP 13: SAVE TRAINED MODELS")
print("="*70)

with open(os.path.join(MODELS_DIR, 'random_forest_pipeline.pkl'), 'wb') as f:
    pickle.dump(best_rf, f)
with open(os.path.join(MODELS_DIR, 'xgboost_pipeline.pkl'), 'wb') as f:
    pickle.dump({'pipeline': best_xgb, 'label_encoder': label_enc}, f)

print(f"  Saved: {MODELS_DIR}/random_forest_pipeline.pkl")
print(f"  Saved: {MODELS_DIR}/xgboost_pipeline.pkl  (includes LabelEncoder for 0..4 ↔ 1..5)")

print("\n" + "="*70)
print("MODELING PHASE COMPLETE ✓")
print("="*70)
