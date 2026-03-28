"""
=============================================================================
PREDICTIVE SYSTEM FAILURE DETECTION USING LOG INTELLIGENCE
Script B — Live Prediction Engine v3 (Final)

VERIFIED column order from live_logs.csv (head -2 confirmed):
    0=log_level, 1=ip, 2=session_id, 3=component, 4=event_type,
    5=message, 6=status, 7=anomaly_label, 8=cpu_usage, 9=memory_usage,
    10=disk_io, 11=response_time, 12=network_latency, 13=warning_count,
    14=error_count, 15=timestamp

Run in Terminal 2:
    python3 live_prediction_engine.py
=============================================================================
"""

import os
import time
import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from datetime import datetime
from collections import defaultdict
from tensorflow.keras.models import load_model

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

RESULTS_DIR   = os.path.expanduser("~/realistic_log_project/results")
LIVE_DIR      = os.path.expanduser("~/realistic_log_project/live")
LIVE_LOG_PATH = os.path.join(LIVE_DIR, "live_logs.csv")
PRED_PATH     = os.path.join(LIVE_DIR, "predictions.csv")
HEALING_PATH  = os.path.join(LIVE_DIR, "self_healing_log.csv")
HORIZON_PATH  = os.path.join(LIVE_DIR, "prediction_horizon.csv")

MODEL_PATH    = os.path.join(RESULTS_DIR, "model_logs_telemetry.keras")
SCALER_PATH   = os.path.join(RESULTS_DIR, "scaler_logs_telemetry.pkl")
ENCODER_PATH  = os.path.join(RESULTS_DIR, "label_encoders.pkl")
COLS_PATH     = os.path.join(RESULTS_DIR, "feature_cols.pkl")

RISK_THRESHOLD = 0.50
POLL_INTERVAL  = 0.3

# ─────────────────────────────────────────────
# VERIFIED COLUMN INDICES
# Confirmed from: head -2 ~/realistic_log_project/live/live_logs.csv
# ─────────────────────────────────────────────

I_LOG_LEVEL       = 0
I_IP              = 1
I_SESSION_ID      = 2
I_COMPONENT       = 3
I_EVENT_TYPE      = 4
I_MESSAGE         = 5
I_STATUS          = 6
I_ANOMALY_LABEL   = 7
I_CPU_USAGE       = 8
I_MEMORY_USAGE    = 9
I_DISK_IO         = 10
I_RESPONSE_TIME   = 11
I_NETWORK_LATENCY = 12
I_WARNING_COUNT   = 13
I_ERROR_COUNT     = 14
I_TIMESTAMP       = 15
TOTAL_COLS        = 16

TELEMETRY_COLS = ["cpu_usage","memory_usage","disk_io","response_time",
                  "network_latency","warning_count","error_count"]
LOG_COLS       = ["log_level","component","event_type","status"]

# ─────────────────────────────────────────────
# CORRECT MIN/MAX NORMALIZATION
# Matches exactly what ml_pipeline_v6 used during training
# ─────────────────────────────────────────────

FEATURE_RANGES = {
    "cpu_usage":       (0.0,   100.0),
    "memory_usage":    (0.0,   100.0),
    "disk_io":         (0.0,   500.0),
    "response_time":   (10.0,  5000.0),
    "network_latency": (1.0,   2000.0),
    "warning_count":   (0.0,   20.0),
    "error_count":     (0.0,   15.0),
}

TELEMETRY_INDICES = {
    "cpu_usage":       I_CPU_USAGE,
    "memory_usage":    I_MEMORY_USAGE,
    "disk_io":         I_DISK_IO,
    "response_time":   I_RESPONSE_TIME,
    "network_latency": I_NETWORK_LATENCY,
    "warning_count":   I_WARNING_COUNT,
    "error_count":     I_ERROR_COUNT,
}

LOG_INDICES = {
    "log_level":  I_LOG_LEVEL,
    "component":  I_COMPONENT,
    "event_type": I_EVENT_TYPE,
    "status":     I_STATUS,
}

# ─────────────────────────────────────────────
# ROOT CAUSE THRESHOLDS — MapReduce Job 4
# ─────────────────────────────────────────────

CPU_THRESH  = 70.0
MEM_THRESH  = 65.0
RESP_THRESH = 500.0
LAT_THRESH  = 200.0
DISK_THRESH = 80.0
ERR_THRESH  = 2.0

# ─────────────────────────────────────────────
# SELF HEALING ACTIONS
# ─────────────────────────────────────────────

HEALING = {
    "Resource Exhaustion" : "Restart overloaded service + reallocate CPU resources",
    "Memory Pressure"     : "Clear memory cache + terminate zombie processes",
    "Network Congestion"  : "Reroute traffic + throttle incoming connections",
    "Service Degradation" : "Scale up service replicas + reduce request load",
    "Disk Bottleneck"     : "Flush disk buffer + archive old log files",
    "Software Fault"      : "Rollback to last stable state + alert dev team",
    "Anomalous Pattern"   : "Flag for manual review + increase monitoring frequency",
}

os.makedirs(LIVE_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────

print("="*65)
print("  LIVE PREDICTION ENGINE v3")
print("  LSTM + Multi-Head Transformer Hybrid")
print("  Verified column indices from live_logs.csv")
print("="*65+"\n")

for label, path in [("Model",MODEL_PATH),("Scaler",SCALER_PATH),
                    ("Encoders",ENCODER_PATH),("Cols",COLS_PATH)]:
    while not os.path.exists(path):
        print(f"  Waiting for {label}...")
        time.sleep(5)

print("[INIT] Loading...")
model    = load_model(MODEL_PATH, compile=False)
with open(SCALER_PATH,  "rb") as f: scaler    = pickle.load(f)
with open(ENCODER_PATH, "rb") as f: encoders  = pickle.load(f)
with open(COLS_PATH,    "rb") as f: feat_cols = pickle.load(f)

print(f"  ✅ Model loaded")
print(f"  ✅ Scaler: {scaler.n_features_in_} features")
print(f"  ✅ Feature cols: {feat_cols}")

# Write headers
if not os.path.exists(PRED_PATH):
    open(PRED_PATH,"w").write(
        "timestamp,session_id,log_level,component,event_type,status,"
        "cpu_usage,memory_usage,disk_io,response_time,network_latency,"
        "error_count,risk_score,risk_level,primary_cause,prediction,actual_label\n")

if not os.path.exists(HEALING_PATH):
    open(HEALING_PATH,"w").write(
        "timestamp,session_id,severity,risk_score,component,"
        "primary_cause,action_taken,status\n")

if not os.path.exists(HORIZON_PATH):
    open(HORIZON_PATH,"w").write(
        "timestamp,session_id,alert_at_log,actual_failure_log,"
        "logs_early,seconds_early,component,primary_cause\n")

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def sf(v, d=0.0):
    try:    return float(v)
    except: return d

def get_root_cause(vals):
    """Raw values — correct indices"""
    cpu  = sf(vals[I_CPU_USAGE])
    mem  = sf(vals[I_MEMORY_USAGE])
    resp = sf(vals[I_RESPONSE_TIME])
    lat  = sf(vals[I_NETWORK_LATENCY])
    disk = sf(vals[I_DISK_IO])
    err  = sf(vals[I_ERROR_COUNT])
    if cpu  > CPU_THRESH:  return "Resource Exhaustion"
    if mem  > MEM_THRESH:  return "Memory Pressure"
    if resp > RESP_THRESH: return "Service Degradation"
    if lat  > LAT_THRESH:  return "Network Congestion"
    if disk > DISK_THRESH: return "Disk Bottleneck"
    if err  > ERR_THRESH:  return "Software Fault"
    return "Anomalous Pattern"

def get_risk_level(score):
    if score >= 0.70:   return "HIGH"
    elif score >= 0.40: return "MEDIUM"
    else:               return "LOW"

def preprocess(vals):
    """Correct min/max normalization per feature"""
    row = {}
    for col, idx in LOG_INDICES.items():
        le  = encoders[col]
        val = str(vals[idx]).strip()
        row[col] = int(le.transform([val])[0]) if val in le.classes_ else 0
    for col, idx in TELEMETRY_INDICES.items():
        val        = sf(vals[idx])
        mn, mx     = FEATURE_RANGES[col]
        row[col]   = float(np.clip((val - mn) / (mx - mn + 1e-8), 0, 1))
    X = np.array([[row[c] for c in feat_cols]], dtype=np.float32)
    X = scaler.transform(X)
    return X.reshape(1, 1, X.shape[1])

# ─────────────────────────────────────────────
# SESSION TRACKER
# ─────────────────────────────────────────────

sessions   = defaultdict(lambda: {"n":0,"fired":False,"al":None,"fl":None})
seen       = set()

def update_session(sid, score, actual_int):
    s = sessions[sid]
    s["n"] += 1
    if actual_int == 1 and s["fl"] is None:
        s["fl"] = s["n"]
    new_alert = False
    if score >= RISK_THRESHOLD and not s["fired"]:
        s["fired"] = True
        s["al"]    = s["n"]
        new_alert  = True
    return new_alert

def get_horizon(sid):
    s = sessions[sid]
    if s["al"] and s["fl"]:
        le = max(0, s["fl"] - s["al"])
        return s["al"], s["fl"], le, round(le * 0.3, 1)
    return None, None, 0, 0.0

# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

print(f"\n[WATCHING] {LIVE_LOG_PATH}")
print(f"[THRESHOLD] Risk >= {RISK_THRESHOLD} fires alert")
print(f"[OUTPUTS]")
print(f"  {PRED_PATH}")
print(f"  {HEALING_PATH}")
print(f"  {HORIZON_PATH}")
print(f"\n{'─'*65}")

last_line = 0   # no header in file — start from line 0
alerts    = 0
processed = 0

try:
    while True:
        if not os.path.exists(LIVE_LOG_PATH):
            print("  Waiting for streamer...")
            time.sleep(2)
            continue

        with open(LIVE_LOG_PATH, "r") as f:
            lines = f.readlines()

        if len(lines) <= last_line:
            time.sleep(POLL_INTERVAL)
            continue

        new_lines  = lines[last_line:]
        last_line  = len(lines)

        for line in new_lines:
            line = line.strip()
            if not line:
                continue
            try:
                vals = line.split(",")
                # Need at least 15 columns
                if len(vals) < 15:
                    continue
                while len(vals) < TOTAL_COLS:
                    vals.append("0")

                sess       = str(vals[I_SESSION_ID]).strip()
                comp       = str(vals[I_COMPONENT]).strip()
                loglvl     = str(vals[I_LOG_LEVEL]).strip()
                evtype     = str(vals[I_EVENT_TYPE]).strip()
                status_val = str(vals[I_STATUS]).strip()
                actual_int = int(sf(vals[I_ANOMALY_LABEL]))
                actual     = "ANOMALY" if actual_int == 1 else "NORMAL"
                cpu        = sf(vals[I_CPU_USAGE])
                mem        = sf(vals[I_MEMORY_USAGE])
                disk       = sf(vals[I_DISK_IO])
                resp       = sf(vals[I_RESPONSE_TIME])
                lat        = sf(vals[I_NETWORK_LATENCY])
                err        = sf(vals[I_ERROR_COUNT])

                X      = preprocess(vals)
                score  = float(model.predict(X, verbose=0)[0][0])
                level  = get_risk_level(score)
                cause  = get_root_cause(vals)
                pred   = "FAILURE" if score >= RISK_THRESHOLD else "NORMAL"
                ts     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                processed += 1

                # Write prediction
                with open(PRED_PATH, "a") as f:
                    f.write(f"{ts},{sess},{loglvl},{comp},{evtype},{status_val},"
                            f"{cpu:.1f},{mem:.1f},{disk:.1f},{resp:.1f},{lat:.1f},"
                            f"{err:.1f},{score:.4f},{level},{cause},{pred},{actual}\n")

                new_alert = update_session(sess, score, actual_int)

                # Horizon
                al, fl, le, se = get_horizon(sess)
                if al and fl and sess not in seen:
                    seen.add(sess)
                    with open(HORIZON_PATH, "a") as f:
                        f.write(f"{ts},{sess},{al},{fl},{le},{se},{comp},{cause}\n")

                # HIGH alert
                if level == "HIGH" and new_alert:
                    alerts += 1
                    action = HEALING.get(cause, "Flag for manual review")
                    sev    = "CRITICAL" if score >= 0.70 else "SEVERE"
                    print(f"🔴 [{ts}] ALERT #{alerts}")
                    print(f"   Session  : {sess}")
                    print(f"   Score    : {score:.4f} | {sev}")
                    print(f"   Cause    : {cause}")
                    print(f"   Action   : {action}")
                    print(f"   CPU={cpu:.1f}%  Mem={mem:.1f}%  "
                          f"Resp={resp:.0f}ms  Lat={lat:.0f}ms")
                    print(f"   Actual   : {actual}")
                    if le > 0:
                        print(f"   ⏱ Predicted {le} logs ({se}s) EARLY")
                    print(f"{'─'*65}")
                    with open(HEALING_PATH, "a") as f:
                        f.write(f"{ts},{sess},{sev},{score:.4f},"
                                f"{comp},{cause},{action},TRIGGERED\n")

                elif level == "MEDIUM":
                    print(f"🟡 [{ts}] WATCH | Score={score:.4f} | "
                          f"{comp} | {cause} | CPU={cpu:.1f}%")

                else:
                    if processed % 200 == 0:
                        print(f"🟢 [{ts}] OK | Processed={processed:,} | "
                              f"Alerts={alerts:,} | CPU={cpu:.1f}% | Resp={resp:.0f}ms")

            except Exception:
                continue

        time.sleep(POLL_INTERVAL)

except KeyboardInterrupt:
    print(f"\n{'='*65}")
    print(f"  [STOPPED]")
    print(f"  Processed : {processed:,} | Alerts : {alerts:,}")
    print(f"  Saved to  : {LIVE_DIR}")
    print(f"{'='*65}")
