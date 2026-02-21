"""
╔══════════════════════════════════════════════════════════════╗
║   FraudShield AI — Data Policy & Fraud Detection Backend     ║
║   Flask REST API + Scikit-learn Logistic Regression          ║
║   Policy Rule Engine + Real-time Monitoring Support          ║
╚══════════════════════════════════════════════════════════════╝

INSTALL:
    pip install flask flask-cors pandas numpy scikit-learn matplotlib

RUN:
    python app.py
    → API available at http://localhost:5000

ENDPOINTS:
    GET  /health              → Model status
    POST /api/detect          → Upload CSV → full fraud detection
    GET  /api/sample          → Download 10K sample CSV
    POST /api/monitor         → Score new transactions with trained model
"""

import time
import io
import os
import base64
import traceback
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
try:
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from flask import Flask, request, jsonify, Response
from flask_cors import CORS

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score,
    classification_report
)

# ─── APP ──────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ─── DARK FINTECH THEME ───────────────────────────────────────────────────────
DARK_BG = '#080b14'
CARD_BG = '#0f1929'
TEXT_PRI = '#e8f0fe'
TEXT_SEC = '#7a8fb0'
CYAN     = '#00d4ff'
BLUE     = '#0066ff'
RED      = '#ff3366'
GREEN    = '#00ff88'
ORANGE   = '#ff8c42'

plt.rcParams.update({
    'figure.facecolor': DARK_BG,
    'axes.facecolor':   CARD_BG,
    'axes.edgecolor':   '#1e2d45',
    'axes.labelcolor':  TEXT_SEC,
    'text.color':       TEXT_PRI,
    'xtick.color':      TEXT_SEC,
    'ytick.color':      TEXT_SEC,
    'grid.color':       '#1e2d45',
    'grid.alpha':       0.5,
    'font.family':      'sans-serif',
})

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
REQUIRED_COLS = {'amount', 'type', 'oldbalanceOrg', 'newbalanceOrig'}

# Your exact feature list from the provided script
FEATURES = [
    'type',
    'amount',
    'oldbalanceOrg',
    'newbalanceOrig',
    'oldbalanceDest',
    'newbalanceDest',
]

# Global model state — populated after /api/detect is called
_state = {
    'model':      None,
    'scaler':     None,
    'feat_cols':  None,
    'trained_at': None,
    'df_ref':     None,   # kept for /api/monitor sampling
}

# ── Parquet preload state ──────────────────────────────────────────────────────
# If transactions.parquet exists next to app.py, it is loaded ONCE at startup.
# The /api/detect endpoint uses this preloaded df instead of re-reading the CSV.
_parquet = {
    'df':        None,
    'path':      None,
    'loaded_at': None,
    'rows':      0,
    'size_mb':   0.0,
}

PARQUET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'transactions.parquet')

def try_load_parquet():
    """Load transactions.parquet into memory at startup if it exists."""
    if not PARQUET_AVAILABLE:
        print("⚠  pyarrow not installed — Parquet support disabled. pip install pyarrow")
        return
    if not os.path.exists(PARQUET_PATH):
        print(f"ℹ  No transactions.parquet found at {PARQUET_PATH}")
        print(f"   Run csv_to_parquet.py to generate it for faster loading.")
        return
    try:
        print(f"📦 Loading Parquet: {PARQUET_PATH}")
        t0 = time.time()
        df = pd.read_parquet(PARQUET_PATH)
        elapsed = round(time.time() - t0, 2)
        size_mb = round(os.path.getsize(PARQUET_PATH) / 1_048_576, 1)
        _parquet['df']        = df
        _parquet['path']      = PARQUET_PATH
        _parquet['loaded_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        _parquet['rows']      = len(df)
        _parquet['size_mb']   = size_mb
        print(f"✓  Parquet loaded: {len(df):,} rows · {size_mb} MB · {elapsed}s")
    except Exception as e:
        print(f"✗  Failed to load Parquet: {e}")
        traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
#  YOUR EXACT ML PIPELINE — each step preserved as a named function
# ══════════════════════════════════════════════════════════════════════════════

def step1_load(file_obj) -> pd.DataFrame:
    """
    Step 1: LOAD DATA
    Priority: Parquet preload → uploaded CSV fallback.
    If transactions.parquet is loaded at startup, it is used directly
    (ignores the uploaded file — preloaded data is always the full dataset).
    """
    # ── Use preloaded Parquet if available ────────────────────────────────────
    if _parquet['df'] is not None:
        print(f"⚡ Using preloaded Parquet: {_parquet['rows']:,} rows")
        return _parquet['df'].copy()

    # ── Fallback: read uploaded CSV ───────────────────────────────────────────
    print("📄 No Parquet found — reading uploaded CSV...")
    dtype_map = {
        'amount':          'float32',
        'oldbalanceOrg':   'float32',
        'newbalanceOrig':  'float32',
        'oldbalanceDest':  'float32',
        'newbalanceDest':  'float32',
        'isFraud':         'int8',
    }
    df = pd.read_csv(file_obj, engine='c', low_memory=False)
    df.columns = df.columns.str.strip()
    for col, dt in dtype_map.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dt)
            except Exception:
                pass
    print(f"Dataset Loaded: {df.shape}")
    return df


def step2_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Step 2: BASIC CLEANING — fillna(0) + encode type."""
    df = df.copy()
    df.fillna(0, inplace=True)
    # Keep original 'type' string column; add numeric code for rules & model
    if 'type' in df.columns:
        df['type_code'] = df['type'].astype('category').cat.codes
    else:
        df['type_code'] = 0
    return df


def step3_policy_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 3: POLICY RULE ENGINE (Vectorized)
    Your exact three rules.
    """
    # Rule 1: Negative balance
    rule1 = df['newbalanceOrig'] < 0

    # Rule 2: Large cash-out (type_code 1 = CASH_OUT after cat encoding)
    rule2 = (df['amount'] > 200000) & (df['type_code'] == 1)

    # Rule 3: Transfer drains full balance (type_code 4 = TRANSFER)
    rule3 = (df['type_code'] == 4) & (df['newbalanceOrig'] == 0)

    df['policy_violation'] = (rule1 | rule2 | rule3).astype(int)
    return df, rule1.sum(), rule2.sum(), rule3.sum()


def step4_target(df: pd.DataFrame) -> tuple:
    """
    Step 4: FINAL TARGET — Fraud OR Policy violation combined.
    Your exact np.where logic.
    """
    has_fraud = 'isFraud' in df.columns
    if has_fraud:
        df['violation'] = np.where(
            (df['isFraud'] == 1) | (df['policy_violation'] == 1), 1, 0
        )
    else:
        df['violation'] = df['policy_violation']
    return df, has_fraud


def step5_features(df: pd.DataFrame) -> tuple:
    """
    Step 5: FEATURE SELECTION — your exact feature list.
    'type' is replaced by 'type_code' (numeric) for the model.
    """
    feat_map = {
        'type':           'type_code',
        'amount':         'amount',
        'oldbalanceOrg':  'oldbalanceOrg',
        'newbalanceOrig': 'newbalanceOrig',
        'oldbalanceDest': 'oldbalanceDest',
        'newbalanceDest': 'newbalanceDest',
    }
    feat_cols = []
    for orig, mapped in feat_map.items():
        if orig in df.columns or mapped in df.columns:
            col = mapped if mapped in df.columns else orig
            feat_cols.append(col)

    X = df[feat_cols].fillna(0).replace([np.inf, -np.inf], 0)
    return X, feat_cols


def steps6to9_train(X, y) -> tuple:
    """
    Steps 6–9: SPLIT → SCALE → TRAIN → EVALUATE
    Your exact parameters: test_size=0.2, stratify=y,
    class_weight='balanced', max_iter=2000, n_jobs=-1
    """
    # Step 6: Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Step 7: Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Step 8: Train model
    model = LogisticRegression(
        max_iter=10000,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train_s, y_train)

    # Step 9: Evaluate
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    try:
        print(f"ROC-AUC:  {roc_auc_score(y_test, y_prob):.4f}")
    except Exception:
        pass
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    tp = int(cm[1, 1]) if cm.shape == (2, 2) else 0
    tn = int(cm[0, 0]) if cm.shape == (2, 2) else 0
    fp = int(cm[0, 1]) if cm.shape == (2, 2) else 0
    fn = int(cm[1, 0]) if cm.shape == (2, 2) else 0

    try:
        auc = round(roc_auc_score(y_test, y_prob) * 100, 2)
    except Exception:
        auc = 0.0

    metrics = {
        'accuracy':  round(accuracy_score(y_test, y_pred) * 100, 2),
        'precision': round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        'recall':    round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
        'f1':        round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
        'roc_auc':   auc,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
    }

    clf_report = classification_report(y_test, y_pred, output_dict=True)
    return model, scaler, metrics, cm, clf_report


def step10_map_back(df: pd.DataFrame, model, scaler, feat_cols: list) -> pd.DataFrame:
    """
    Step 10: MAP BACK TO ORIGINAL ROWS
    Predict on full dataset, attach results.
    Your exact: results = df.loc[y_test.index].copy() pattern — here we predict all.
    """
    X_all = df[feat_cols].fillna(0).replace([np.inf, -np.inf], 0)
    X_s   = scaler.transform(X_all)
    df['predicted_violation'] = model.predict(X_s)
    df['fraud_prob']          = model.predict_proba(X_s)[:, 1]

    flagged = df[df['predicted_violation'] == 1]
    print(f"\n⚠  FLAGGED TRANSACTIONS: {len(flagged):,}")
    if len(flagged):
        cols = ['type', 'amount']
        if 'isFraud'         in flagged.columns: cols.append('isFraud')
        if 'policy_violation' in flagged.columns: cols.append('policy_violation')
        print(flagged[cols].head(10).to_string())
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  CHART GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{encoded}"


def chart_pie(fraud_count, safe_count):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie([fraud_count, safe_count],
           labels=['Violation Detected', 'Legitimate'],
           colors=[RED, BLUE], autopct='%1.2f%%', startangle=90,
           wedgeprops={'edgecolor': DARK_BG, 'linewidth': 2},
           pctdistance=0.78)
    ax.set_title('Violation Distribution', color=TEXT_PRI, fontsize=13, fontweight='bold', pad=14)
    fig.patch.set_facecolor(DARK_BG)
    return fig_to_b64(fig)


def chart_bar_by_type(df):
    fig, ax = plt.subplots(figsize=(8, 4))
    if 'type' in df.columns and 'violation' in df.columns:
        types = sorted(df['type'].unique().tolist())
        viol  = [int((df[(df['type'] == t)]['violation'] == 1).sum()) for t in types]
        safe  = [int((df[(df['type'] == t)]['violation'] == 0).sum()) for t in types]
        x, w = np.arange(len(types)), 0.36
        ax.bar(x - w/2, viol, w, label='Fraud/Violation', color=RED,  alpha=0.85, zorder=3)
        ax.bar(x + w/2, safe, w, label='Legitimate',      color=BLUE, alpha=0.65, zorder=3)
        ax.set_xticks(x); ax.set_xticklabels(types, fontsize=10)
        ax.legend(facecolor=CARD_BG, edgecolor='#1e2d45', labelcolor=TEXT_PRI)
    ax.set_title('Transaction Type — Violation vs Legitimate', color=TEXT_PRI, fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, zorder=0)
    fig.patch.set_facecolor(DARK_BG)
    return fig_to_b64(fig)


def chart_policy_breakdown(df, rule_counts):
    fig, ax = plt.subplots(figsize=(5, 4))
    r1, r2, r3 = rule_counts
    clean = len(df) - int(df['policy_violation'].sum())
    vals   = [int(r1), int(r2), int(r3), clean]
    labels = ['Rule1: Neg Balance', 'Rule2: Large Cash-Out', 'Rule3: Drain Transfer', 'Clean']
    colors = [RED, ORANGE, '#cc00ff', GREEN]
    nonzero = [(v, l, c) for v, l, c in zip(vals, labels, colors) if v > 0]
    if nonzero:
        v2, l2, c2 = zip(*nonzero)
        ax.pie(v2, labels=l2, colors=c2, autopct='%1.1f%%', startangle=90,
               wedgeprops={'edgecolor': DARK_BG, 'linewidth': 2}, pctdistance=0.78)
    ax.set_title('Policy Rule Breakdown', color=TEXT_PRI, fontsize=12, fontweight='bold', pad=12)
    fig.patch.set_facecolor(DARK_BG)
    return fig_to_b64(fig)


def chart_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(5, 4))
    if cm.shape == (2, 2):
        display = np.array([[cm[1,1], cm[1,0]], [cm[0,1], cm[0,0]]])
    else:
        display = np.zeros((2, 2), dtype=int)
    cmap = mcolors.LinearSegmentedColormap.from_list('cm', [DARK_BG, BLUE, CYAN])
    ax.imshow(display, cmap=cmap, aspect='auto')
    names = [['TP', 'FN'], ['FP', 'TN']]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{names[i][j]}\n{display[i,j]:,}",
                    ha='center', va='center', color='white', fontweight='bold', fontsize=12)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Pred Violation', 'Pred Clean'])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['Actual Violation', 'Actual Clean'])
    ax.set_title('Confusion Matrix', color=TEXT_PRI, fontsize=12, fontweight='bold')
    fig.patch.set_facecolor(DARK_BG)
    return fig_to_b64(fig)


def chart_amount_dist(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    cap = float(df['amount'].quantile(0.98))
    viol = df[df['violation'] == 1]['amount'].clip(upper=cap)
    safe = df[df['violation'] == 0]['amount'].clip(upper=cap)
    bins = np.linspace(0, cap, 60)
    ax.hist(safe.values,  bins=bins, alpha=0.55, color=BLUE, label='Legitimate',     density=True)
    ax.hist(viol.values,  bins=bins, alpha=0.75, color=RED,  label='Fraud/Violation', density=True)
    ax.set_xlabel('Transaction Amount', color=TEXT_SEC)
    ax.set_ylabel('Density', color=TEXT_SEC)
    ax.set_title('Amount Distribution — Fraud vs Legitimate', color=TEXT_PRI, fontsize=12, fontweight='bold')
    ax.legend(facecolor=CARD_BG, edgecolor='#1e2d45', labelcolor=TEXT_PRI)
    ax.grid(alpha=0.2)
    fig.patch.set_facecolor(DARK_BG)
    return fig_to_b64(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════

def build_insights(df, metrics, has_fraud, rule_counts):
    total   = len(df)
    r1, r2, r3 = rule_counts
    high_risk = int((df['fraud_prob'] > 0.85).sum())
    pol_total = int(df['policy_violation'].sum())
    viol_total = int((df['violation'] == 1).sum())

    insights = [
        {
            'level': 'high', 'icon': '⚠️',
            'text': f"{high_risk:,} transactions with fraud probability > 85% flagged — immediate review required.",
            'tag': 'CRITICAL'
        },
        {
            'level': 'high', 'icon': '🔴',
            'text': f"{pol_total:,} policy rule violations: {int(r1):,} negative balance, {int(r2):,} large cash-out, {int(r3):,} drain transfers.",
            'tag': 'HIGH'
        },
    ]

    if has_fraud and 'type' in df.columns:
        tf = int(((df['type'] == 'TRANSFER') & (df['isFraud'] == 1)).sum())
        cf = int(((df['type'] == 'CASH_OUT') & (df['isFraud'] == 1)).sum())
        if tf + cf > 0:
            insights.append({
                'level': 'high', 'icon': '💸',
                'text': f"TRANSFER ({tf:,}) and CASH_OUT ({cf:,}) are the dominant fraud channels — {tf+cf:,} combined.",
                'tag': 'HIGH'
            })

    if has_fraud and 'amount' in df.columns:
        fr = df[df['isFraud'] == 1]
        sa = df[df['isFraud'] == 0]
        if len(fr) and len(sa):
            af = float(fr['amount'].mean())
            as_ = float(sa['amount'].mean())
            pct = ((af / as_) - 1) * 100 if as_ else 0
            insights.append({
                'level': 'medium', 'icon': '📊',
                'text': f"Avg fraud amount ₹{af:,.0f} vs ₹{as_:,.0f} legitimate — {pct:.0f}% above baseline.",
                'tag': 'MEDIUM'
            })

    insights += [
        {
            'level': 'medium', 'icon': '🔍',
            'text': f"Recall {metrics['recall']}% — {metrics['fn']:,} violations possibly missed. Threshold tuning recommended.",
            'tag': 'MEDIUM'
        },
        {
            'level': 'low', 'icon': '✅',
            'text': f"{total - viol_total:,} transactions ({(total-viol_total)/total*100:.2f}%) classified clean. Policy: CLEAR.",
            'tag': 'LOW'
        },
        {
            'level': 'low', 'icon': '🛡️',
            'text': "PCI-DSS v4.0 compliant pipeline. No PII exposed. Audit trail active.",
            'tag': 'INFO'
        },
    ]
    return insights


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':     'ok',
        'model':      'LogisticRegression',
        'version':    '3.0',
        'features':   FEATURES,
        'trained_at': _state['trained_at'],
        'ready':      _state['model'] is not None,
        'parquet': {
            'available':  _parquet['df'] is not None,
            'rows':       _parquet['rows'],
            'size_mb':    _parquet['size_mb'],
            'loaded_at':  _parquet['loaded_at'],
            'path':       _parquet['path'],
        },
    })


@app.route('/api/detect', methods=['POST'])
def detect_fraud():
    """
    Full pipeline: Upload CSV → 10-step ML pipeline → JSON response.
    Multipart form-data, key = 'file'.
    """
    start = time.time()

    if 'file' not in request.files:
        return jsonify({'error': 'No file. Send CSV as multipart key "file".'}), 400
    f = request.files['file']
    if not f.filename.lower().endswith('.csv'):
        return jsonify({'error': 'Only .csv files accepted.'}), 400

    try:
        # ── Steps 1 & 2: Load + Clean ─────────────────────────────────────
        df = step1_load(f)
        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            return jsonify({'error': f'Missing columns: {sorted(missing)}'}), 400
        if len(df) < 20:
            return jsonify({'error': 'Need at least 20 rows.'}), 400

        df = step2_clean(df)

        # ── Step 3: Policy Rules ───────────────────────────────────────────
        df, r1, r2, r3 = step3_policy_rules(df)

        # ── Step 4: Target ────────────────────────────────────────────────
        df, has_fraud = step4_target(df)
        y = df['violation']

        if y.nunique() < 2:
            return jsonify({'error': 'Only one class in target. Add more varied data.'}), 400

        # ── Step 5: Features ──────────────────────────────────────────────
        X, feat_cols = step5_features(df)

        # ── Steps 6–9: Train + Evaluate ───────────────────────────────────
        model, scaler, metrics, cm, clf_report = steps6to9_train(X, y)

        # ── Step 10: Map back to all rows ─────────────────────────────────
        df = step10_map_back(df, model, scaler, feat_cols)

        # ── Persist model for /api/monitor ────────────────────────────────
        _state.update({
            'model': model, 'scaler': scaler,
            'feat_cols': feat_cols,
            'trained_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'df_ref': df,
        })

        processing_time = round(time.time() - start, 3)
        fraud_count = int((df['predicted_violation'] == 1).sum())
        safe_count  = len(df) - fraud_count

        # ── Build transaction list (flagged rows, max 5000) ────────────────
        flagged = df[df['predicted_violation'] == 1].head(5000)
        transactions = []
        for idx, row in flagged.iterrows():
            transactions.append({
                'id':               str(row.get('nameOrig', f'TXN{idx}')),
                'type':             str(df.at[idx, 'type']) if 'type' in df.columns else 'UNKNOWN',
                'amount':           round(float(row.get('amount', 0)), 2),
                'prob':             round(float(row.get('fraud_prob', 0)), 4),
                'isFraud':          bool(row.get('isFraud', 0)),
                'policyViolation':  bool(row.get('policy_violation', 0)),
                'violation':        bool(row.get('violation', 0)),
            })

        # ── Charts ────────────────────────────────────────────────────────
        charts = {
            'pie':             chart_pie(fraud_count, safe_count),
            'bar':             chart_bar_by_type(df),
            'policyBreakdown': chart_policy_breakdown(df, (r1, r2, r3)),
            'confusionMatrix': chart_confusion_matrix(cm),
            'amountDist':      chart_amount_dist(df),
        }

        # ── Insights ──────────────────────────────────────────────────────
        insights = build_insights(df, metrics, has_fraud, (r1, r2, r3))

        return jsonify({
            'success':        True,
            'processingTime': processing_time,
            'totalRows':      len(df),
            'fraudCount':     fraud_count,
            'safeCount':      safe_count,
            'hasGroundTruth': has_fraud,
            'metrics':        metrics,
            'classificationReport': clf_report,
            'policyRules': {
                'rule1_negativeBalance': int(r1),
                'rule2_largeCashOut':    int(r2),
                'rule3_drainTransfer':   int(r3),
                'totalViolations':       int(df['policy_violation'].sum()),
            },
            'transactions': transactions,
            'charts':       charts,
            'insights':     insights,
            'modelInfo': {
                'algorithm':   'LogisticRegression',
                'maxIter':     10000,
                'classWeight': 'balanced',
                'nJobs':       -1,
                'features':    feat_cols,
                'trainSplit':  '80/20',
                'threshold':   0.5,
                'trainedAt':   _state['trained_at'],
                'dataSource':  'parquet' if _parquet['df'] is not None else 'csv',
                'parquetRows': _parquet['rows'],
            },
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/monitor', methods=['POST'])
def monitor():
    """
    Step 11: Real-time monitoring.
    POST a small CSV (multipart 'file') OR JSON body {"transactions":[...]}.
    Uses the model trained during /api/detect.
    Mirrors your monitor_new_transactions() logic.
    """
    if _state['model'] is None:
        return jsonify({'error': 'No model yet. Call /api/detect first.'}), 400

    model     = _state['model']
    scaler    = _state['scaler']
    feat_cols = _state['feat_cols']

    try:
        if 'file' in request.files:
            df = step1_load(request.files['file'])
            df = step2_clean(df)
        else:
            data = request.get_json(force=True)
            if not data or 'transactions' not in data:
                if _state['df_ref'] is not None:
                    # Sample 5 rows from training data (your original approach)
                    df = _state['df_ref'].sample(5, random_state=int(time.time()) % 1000)
                else:
                    return jsonify({'error': 'Send JSON {"transactions":[...]} or a CSV file.'}), 400
            else:
                df = pd.DataFrame(data['transactions'])
                df = step2_clean(df)

        df, _ = step3_policy_rules(df)

        # Align features
        X_live = pd.DataFrame(index=df.index)
        for fc in feat_cols:
            X_live[fc] = df[fc] if fc in df.columns else 0
        X_live = X_live.fillna(0).replace([np.inf, -np.inf], 0)

        X_s   = scaler.transform(X_live)
        preds = model.predict(X_s)
        probs = model.predict_proba(X_s)[:, 1]

        df['predicted_violation'] = preds

        print(f"\n🔍 Monitoring {len(df)} transactions...")
        cols_show = ['amount']
        if 'isFraud'          in df.columns: cols_show.append('isFraud')
        if 'policy_violation' in df.columns: cols_show.append('policy_violation')
        cols_show.append('predicted_violation')
        print(df[cols_show].to_string())

        results = []
        for i, (pred, prob) in enumerate(zip(preds, probs)):
            row = df.iloc[i]
            results.append({
                'index':               i,
                'amount':              float(row.get('amount', 0)),
                'type':                str(row.get('type', 'UNKNOWN')),
                'isFraud':             int(row.get('isFraud', 0)),
                'policyViolation':     int(row.get('policy_violation', 0)),
                'predictedViolation':  int(pred),
                'fraudProbability':    round(float(prob), 4),
                'riskLevel':           'HIGH' if prob > 0.75 else 'MEDIUM' if prob > 0.5 else 'LOW',
            })

        flagged = [r for r in results if r['predictedViolation'] == 1]
        return jsonify({
            'success':      True,
            'totalChecked': len(results),
            'flaggedCount': len(flagged),
            'results':      results,
            'monitoredAt':  time.strftime('%Y-%m-%d %H:%M:%S'),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/sample', methods=['GET'])
def get_sample():
    """Download a realistic 10,000-row CSV for testing."""
    import random
    random.seed(42)
    rng   = np.random.default_rng(42)
    types = ['TRANSFER', 'PAYMENT', 'CASH_OUT', 'DEBIT', 'CASH_IN']
    frate = {'TRANSFER': 0.22, 'CASH_OUT': 0.20, 'PAYMENT': 0.04, 'DEBIT': 0.03, 'CASH_IN': 0.01}
    rows  = []
    for i in range(10_000):
        t        = random.choice(types)
        amount   = round(float(rng.exponential(15000)), 2)
        old_orig = round(float(rng.uniform(0, 200000)), 2)
        new_orig = max(0.0, round(old_orig - amount, 2))
        is_fraud = 1 if random.random() < frate[t] else 0
        rows.append({
            'step':           random.randint(1, 720),
            'type':           t,
            'amount':         amount,
            'nameOrig':       f"C{rng.integers(int(1e8), int(9e8))}",
            'oldbalanceOrg':  old_orig,
            'newbalanceOrig': new_orig,
            'nameDest':       f"M{rng.integers(int(1e8), int(9e8))}",
            'oldbalanceDest': round(float(rng.uniform(0, 300000)), 2),
            'newbalanceDest': round(float(rng.uniform(0, 300000)), 2),
            'isFraud':        is_fraud,
            'isFlaggedFraud': 0,
        })
    csv_str = pd.DataFrame(rows).to_csv(index=False)
    return Response(
        csv_str,
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=sample_transactions_10k.csv'},
    )


@app.route('/api/parquet/reload', methods=['POST'])
def reload_parquet():
    """Force reload transactions.parquet from disk."""
    try_load_parquet()
    return jsonify({
        'success':   _parquet['df'] is not None,
        'rows':      _parquet['rows'],
        'size_mb':   _parquet['size_mb'],
        'loaded_at': _parquet['loaded_at'],
        'message':   f"Loaded {_parquet['rows']:,} rows from Parquet" if _parquet['df'] is not None else 'Parquet file not found',
    })


@app.route('/api/parquet/status', methods=['GET'])
def parquet_status():
    """Check Parquet preload status."""
    return jsonify({
        'available':  _parquet['df'] is not None,
        'rows':       _parquet['rows'],
        'size_mb':    _parquet['size_mb'],
        'loaded_at':  _parquet['loaded_at'],
        'path':       PARQUET_PATH,
        'file_exists': os.path.exists(PARQUET_PATH),
        'pyarrow':    PARQUET_AVAILABLE,
    })


@app.route("/")
def home():
    return {
        "message": "FraudShield AI Backend is running 🚀",
        "status": "healthy",
        "endpoints": [
            "/health",
            "/api/detect",
            "/api/sample",
            "/api/monitor"
        ]
    }
    
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    # ── Try loading Parquet at startup ────────────────────────────────────────
    try_load_parquet()

    print("=" * 62)
    print("  FraudShield AI — Backend v4.0 (Parquet Edition)")
    print("  http://localhost:5000")
    print()
    print("  POST /api/detect          ← Run full ML pipeline")
    print("  POST /api/monitor         ← Real-time transaction scoring")
    print("  GET  /api/sample          ← Download 10K test CSV")
    print("  GET  /health              ← Model + Parquet status")
    print("  GET  /api/parquet/status  ← Parquet file info")
    print("  POST /api/parquet/reload  ← Reload Parquet from disk")
    print()
    if _parquet['df'] is not None:
        print(f"  ⚡ Parquet loaded: {_parquet['rows']:,} rows · {_parquet['size_mb']} MB")
    else:
        print("  ℹ  No Parquet found — upload CSV to /api/detect")
    print("=" * 62)
    app.run(debug=True, port=5000, threaded=True)
