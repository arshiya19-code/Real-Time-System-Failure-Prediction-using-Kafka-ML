"""
=============================================================================
PREDICTIVE SYSTEM FAILURE DETECTION USING LOG INTELLIGENCE
ML Pipeline v6.0 Balanced — 50:50 LSTM + Transformer Hybrid
=============================================================================
"""

import os
import gc
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score,
    average_precision_score, classification_report
)

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout,
    MultiHeadAttention, LayerNormalization,
    GlobalAveragePooling1D, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

DATASET_PATH = os.path.expanduser("~/realistic_log_project/dataset/structured_logs_balanced.csv")
RESULTS_DIR  = os.path.expanduser("~/realistic_log_project/results")
RANDOM_SEED  = 42
BATCH_SIZE   = 2048
EPOCHS       = 10
NOISE_SIGMA  = 0.15

os.makedirs(RESULTS_DIR, exist_ok=True)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

TELEMETRY_COLS = [
    "cpu_usage", "memory_usage", "disk_io",
    "response_time", "network_latency",
    "warning_count", "error_count"
]
LOG_COLS = ["log_level", "component", "event_type", "status"]

# ─────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────────

def load_data():
    print("[STEP 1] Loading Balanced dataset...")
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    # The dataset is already perfectly randomly 50:50.
    print(f"         Rows    : {len(df):,}")
    print(f"         Normal  : {(df.anomaly_label==0).sum():,} ({(df.anomaly_label==0).mean()*100:.1f}%)")
    print(f"         Anomaly : {(df.anomaly_label==1).sum():,} ({(df.anomaly_label==1).mean()*100:.1f}%)")
    return df

# ─────────────────────────────────────────────
# STEP 2 — FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df, use_telemetry=True, encoders=None, fit_encoders=True):
    data = df.copy()

    if fit_encoders:
        encoders = {}
        for col in LOG_COLS:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            encoders[col] = le
    else:
        for col in LOG_COLS:
            le = encoders[col]
            data[col] = data[col].astype(str).map(
                lambda x: le.transform([x])[0] if x in le.classes_ else 0)

    if use_telemetry:
        for col in TELEMETRY_COLS:
            cmin = data[col].min()
            cmax = data[col].max()
            data[col] = (data[col] - cmin) / (cmax - cmin + 1e-8)
        rng = np.random.default_rng(RANDOM_SEED)
        for col in TELEMETRY_COLS:
            noise = rng.normal(0, NOISE_SIGMA, size=len(data))
            data[col] = np.clip(data[col] + noise, 0, 1)
        feature_cols = LOG_COLS + TELEMETRY_COLS
    else:
        feature_cols = LOG_COLS

    X = data[feature_cols].values.astype(np.float32)
    y = data["anomaly_label"].values.astype(int)
    del data
    gc.collect()
    return X, y, feature_cols, encoders

# ─────────────────────────────────────────────
# STEP 3 — TRAIN/TEST SPLIT + CLASS WEIGHTS
# ─────────────────────────────────────────────

def prepare_splits(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

    print(f"         Train   : {len(X_train):,} | Test : {len(X_test):,}")
    print(f"         Anomaly in train : {y_train.sum():,} ({y_train.mean()*100:.1f}%)")

    cw = compute_class_weight("balanced", classes=np.array([0,1]), y=y_train)
    class_weight = {0: cw[0], 1: cw[1]}
    print(f"         Class weights   : Normal={cw[0]:.2f} | Anomaly={cw[1]:.2f}")

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test  = X_test.reshape(X_test.shape[0],  1, X_test.shape[1])

    return X_train, X_test, y_train, y_test, class_weight, scaler

# ─────────────────────────────────────────────
# STEP 4 — MODEL
# ─────────────────────────────────────────────

def build_model(input_shape):
    inputs = Input(shape=input_shape)

    x = LSTM(128, return_sequences=True)(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    attn1 = MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.2)(x, x)
    x = LayerNormalization(epsilon=1e-6)(attn1 + x)
    x = Dropout(0.4)(x)

    attn2 = MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.2)(x, x)
    x = LayerNormalization(epsilon=1e-6)(attn2 + x)

    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ─────────────────────────────────────────────
# STEP 5 — TRAIN + EVALUATE
# ─────────────────────────────────────────────

def train_and_evaluate(X_train, X_test, y_train, y_test,
                       class_weight, dataset_name,
                       scaler=None, encoders=None, feature_cols=None,
                       save_model=False):

    print(f"\n         Training on : {dataset_name}")
    print(f"         Input shape : {X_train.shape}")

    model = build_model((X_train.shape[1], X_train.shape[2]))

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=2, min_lr=1e-5, verbose=1)
    ]

    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )

    y_prob = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)
    roc  = roc_auc_score(y_test, y_prob)
    prc  = average_precision_score(y_test, y_prob)
    
    # NEW METRIC 7: False Positive Rate (FPR)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    results = {
        "Dataset":          dataset_name,
        "Accuracy":         round(acc,  4),
        "Precision":        round(prec, 4),
        "Recall":           round(rec,  4),
        "F1-Score":         round(f1,   4),
        "AUC-ROC":          round(roc,  4),
        "AUC-PRC":          round(prc,  4),
        "FPR":              round(fpr,  4),
        "Confusion Matrix": cm.tolist()
    }

    print(f"\n{'─'*52}")
    print(f"  RESULTS — {dataset_name}")
    print(f"{'─'*52}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  AUC-ROC   : {roc:.4f}")
    print(f"  AUC-PRC   : {prc:.4f}")
    print(f"  FPR (New) : {fpr:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"                   Pred Normal  Pred Anomaly")
    print(f"  Actual Normal     {cm[0][0]:>9}    {cm[0][1]:>9}")
    print(f"  Actual Anomaly    {cm[1][0]:>9}    {cm[1][1]:>9}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Normal','Anomaly'], zero_division=0)}")

    safe = dataset_name.replace(' ','_').replace('+','plus')
    out  = os.path.join(RESULTS_DIR, f"results_Balanced_{safe}.txt")
    with open(out, "w") as f:
        f.write(f"RESULTS — Balanced {dataset_name}\n{'='*50}\n")
        f.write(f"Accuracy  : {acc:.4f}\n")
        f.write(f"Precision : {prec:.4f}\n")
        f.write(f"Recall    : {rec:.4f}\n")
        f.write(f"F1-Score  : {f1:.4f}\n")
        f.write(f"AUC-ROC   : {roc:.4f}\n")
        f.write(f"AUC-PRC   : {prc:.4f}\n")
        f.write(f"FPR       : {fpr:.4f}\n")
        f.write(f"\nConfusion Matrix:\n{cm}\n")
        f.write(f"\nClassification Report:\n")
        f.write(classification_report(y_test, y_pred,
                target_names=["Normal","Anomaly"], zero_division=0))
    print(f"  Saved: {out}")

    # ── SAVE MODEL + ARTIFACTS (Logs+Telemetry only) ──
    if save_model:
        print(f"\n  Saving Balanced model and artifacts...")

        model_path   = os.path.join(RESULTS_DIR, "model_logs_telemetry_balanced.keras")
        scaler_path  = os.path.join(RESULTS_DIR, "scaler_logs_telemetry_balanced.pkl")
        encoder_path = os.path.join(RESULTS_DIR, "label_encoders_balanced.pkl")
        cols_path    = os.path.join(RESULTS_DIR, "feature_cols_balanced.pkl")

        model.save(model_path)
        with open(scaler_path,  "wb") as f: pickle.dump(scaler,       f)
        with open(encoder_path, "wb") as f: pickle.dump(encoders,     f)
        with open(cols_path,    "wb") as f: pickle.dump(feature_cols, f)

        print(f"  ✅ Model   saved : {model_path}")
        print(f"  ✅ Scaler  saved : {scaler_path}")
        print(f"  ✅ Encoders saved: {encoder_path}")
        print(f"  ✅ Columns saved : {cols_path}")

    del model
    gc.collect()
    tf.keras.backend.clear_session()
    return results

# ─────────────────────────────────────────────
# COMPARISON
# ─────────────────────────────────────────────

def print_and_save_comparison(r1, r2):
    metrics = ["Accuracy","Precision","Recall","F1-Score","AUC-ROC","AUC-PRC","FPR"]

    print("\n"+"="*70)
    print("  FINAL 50:50 COMPARISON — LSTM + Transformer Hybrid")
    print("  Logs Only (benchmark) vs Logs+Telemetry (our contribution)")
    print("="*70)
    print(f"  {'Metric':<14} {'Logs Only':>14} {'Logs+Telemetry':>16} {'Improvement':>14}")
    print("  "+"-"*60)
    for m in metrics:
        v1   = r1[m]
        v2   = r2[m]
        diff = round(v2-v1, 4)
        # For FPR, improvement means a negative diff (lower is better)
        if m == "FPR":
            flag = "✅" if diff < 0 else ("➖" if diff==0 else "❌")
        else:
            flag = "✅" if diff > 0 else ("➖" if diff==0 else "❌")
            
        sign = "+" if diff >= 0 else ""
        print(f"  {m:<14} {v1:>14.4f} {v2:>16.4f}   {flag} {sign}{diff:.4f}")
    print("="*70)

    cm1 = np.array(r1["Confusion Matrix"])
    cm2 = np.array(r2["Confusion Matrix"])
    print(f"\n  Confusion Matrix — Logs Only:")
    print(f"                   Pred Normal  Pred Anomaly")
    print(f"  Actual Normal     {cm1[0][0]:>9}    {cm1[0][1]:>9}")
    print(f"  Actual Anomaly    {cm1[1][0]:>9}    {cm1[1][1]:>9}")
    print(f"\n  Confusion Matrix — Logs + Telemetry:")
    print(f"                   Pred Normal  Pred Anomaly")
    print(f"  Actual Normal     {cm2[0][0]:>9}    {cm2[0][1]:>9}")
    print(f"  Actual Anomaly    {cm2[1][0]:>9}    {cm2[1][1]:>9}")

    comp = os.path.join(RESULTS_DIR, "comparison_final_balanced.txt")
    with open(comp, "w") as f:
        f.write("FINAL 50:50 METRIC COMPARISON — LSTM + Transformer Hybrid\n")
        f.write("Logs Only (benchmark) vs Logs+Telemetry (our contribution)\n")
        f.write("="*65+"\n")
        f.write(f"{'Metric':<14} {'Logs Only':>14} {'Logs+Telemetry':>16} {'Diff':>10}\n")
        f.write("-"*65+"\n")
        for m in metrics:
            diff = round(r2[m]-r1[m], 4)
            sign = "+" if diff >= 0 else ""
            f.write(f"{m:<14} {r1[m]:>14.4f} {r2[m]:>16.4f} {sign}{diff:>9.4f}\n")
        f.write(f"\nConfusion Matrix — Logs Only:\n{cm1}\n")
        f.write(f"\nConfusion Matrix — Logs+Telemetry:\n{cm2}\n")
    print(f"\n  Saved: {comp}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("="*65)
    print("  ML PIPELINE v6.1 — 50:50 BALANCED DATASET")
    print("  Evaluates 7 metrics including False Positive Rate (FPR)")
    print("="*65+"\n")

    df = load_data()

    # ── Dataset 1 — Logs Only ──
    print("\n"+"═"*52)
    print("  DATASET 1 — Logs Only")
    print("═"*52)
    X1, y1, cols1, enc1 = engineer_features(df, use_telemetry=False)
    X1_tr, X1_te, y1_tr, y1_te, cw1, sc1 = prepare_splits(X1, y1)
    results1 = train_and_evaluate(
        X1_tr, X1_te, y1_tr, y1_te, cw1,
        "Logs Only", save_model=False)
    del X1, y1, X1_tr, X1_te, y1_tr, y1_te
    gc.collect()

    # ── Dataset 2 — Logs + Telemetry ──
    print("\n"+"═"*52)
    print("  DATASET 2 — Logs + Telemetry")
    print("═"*52)
    X2, y2, cols2, enc2 = engineer_features(df, use_telemetry=True)
    X2_tr, X2_te, y2_tr, y2_te, cw2, sc2 = prepare_splits(X2, y2)
    results2 = train_and_evaluate(
        X2_tr, X2_te, y2_tr, y2_te, cw2,
        "Logs + Telemetry",
        scaler=sc2, encoders=enc2, feature_cols=cols2,
        save_model=True)
    del X2, y2, X2_tr, X2_te, y2_tr, y2_te
    gc.collect()

    print_and_save_comparison(results1, results2)
    print(f"\n  All results saved to : {RESULTS_DIR}")
    print("\n  PIPELINE COMPLETE")

if __name__ == "__main__":
    main()
