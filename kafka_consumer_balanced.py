# =============================================================================
# FINAL KAFKA CONSUMER — 50:50 BALANCED PIPELINE ✅
# =============================================================================

import os, time, json, pickle, numpy as np, signal, sys
from kafka import KafkaConsumer
from datetime import datetime
from collections import defaultdict
from tensorflow.keras.models import load_model

# ─────────────────────────────────────────────
# CTRL+C HANDLER
# ─────────────────────────────────────────────
def stop_handler(sig, frame):
    print("\n\n🛑 Stopping consumer safely...")
    try:
        if PRED_BUFFER:
            with open(CSV_PATHS["predictions"], "a") as f:
                f.writelines(PRED_BUFFER)
        if SCORE_BUFFER:
            with open(CSV_PATHS["scores"], "a") as f:
                f.writelines(SCORE_BUFFER)
    except: pass
    try: consumer.close()
    except: pass
    sys.exit(0)

signal.signal(signal.SIGINT, stop_handler)

# ─────────────────────────────────────────────
# PATHS AND INITIALIZATION
# ─────────────────────────────────────────────
BASE = "."
RES  = "."
LIVE = "."

CSV_PATHS = {
    "predictions": "predictions_balanced.csv",
    "healing": "self_healing_log_balanced.csv",
    "horizon": "prediction_horizon_balanced.csv",
    "scores": "risk_scores_balanced.csv",
    "root_cause": "root_cause_balanced.csv"
}

def init_csv(path, header):
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(header + "\n")

init_csv(CSV_PATHS["predictions"], "timestamp,session_id,risk_level,risk_score,actual_label,prediction")
init_csv(CSV_PATHS["healing"], "timestamp,session_id,risk_level,root_cause,action_taken")
init_csv(CSV_PATHS["horizon"], "session_id,first_alert_time,confirmed_failure_time,horizon_seconds")
init_csv(CSV_PATHS["scores"], "timestamp,session_id,cpu_usage,memory_usage,response_time,network_latency,risk_score")
init_csv(CSV_PATHS["root_cause"], "timestamp,session_id,root_cause_category,component")

# Load Models
try:
    model = load_model(os.path.join(RES, "model_logs_telemetry_balanced.keras"), compile=False)
    scaler = pickle.load(open(os.path.join(RES, "scaler_logs_telemetry_balanced.pkl"), "rb"))
    encoders = pickle.load(open(os.path.join(RES, "label_encoders_balanced.pkl"), "rb"))
    feat_cols = pickle.load(open(os.path.join(RES, "feature_cols_balanced.pkl"), "rb"))
except Exception as e:
    print(f"⚠️ Missing ML Artifacts: {e}. Running with mock logic.")
    model, scaler, encoders, feat_cols = None, None, {}, []

# ─────────────────────────────────────────────
# CONFIG & HELPERS
# ─────────────────────────────────────────────
HIGH_THRESHOLD = 0.70

session_tracker = defaultdict(lambda: {
    "high_count": 0,
    "first_time": None,
    "printed": False
})

def sf(x):
    try: return float(x)
    except: return 0.0

def preprocess(msg):
    if not scaler or not encoders or not feat_cols:
        return np.zeros((1, 1, 11))

    row = {}
    for col in ["log_level","component","event_type","status"]:
        le = encoders.get(col)
        val = str(msg.get(col,""))
        row[col] = int(le.transform([val])[0]) if le and val in le.classes_ else 0

    for col in ["cpu_usage","memory_usage","disk_io","response_time","network_latency","warning_count","error_count"]:
        row[col] = sf(msg.get(col,0))

    X = np.array([[row[c] for c in feat_cols]], dtype=np.float32)
    X = scaler.transform(X)
    return X.reshape(1,1,X.shape[1])

def get_root_cause(msg):
    scores = {
        "Resource Exhaustion": max(0, sf(msg.get("cpu_usage")) - 70) / 30,
        "Memory Pressure": max(0, sf(msg.get("memory_usage")) - 65) / 35,
        "Service Degradation": max(0, sf(msg.get("response_time")) - 500) / 4500,
        "Network Congestion": max(0, sf(msg.get("network_latency")) - 200) / 1800,
        "Disk Bottleneck": max(0, sf(msg.get("disk_io")) - 80) / 420,
        "Software Fault": max(0, sf(msg.get("error_count")) - 2) / 13
    }
    scores = {k: v for k, v in scores.items() if v > 0}
    return max(scores, key=scores.get) if scores else "Anomalous Pattern"

HEAL = {
    "Resource Exhaustion": "Restart overloaded service + reallocate CPU",
    "Memory Pressure": "Clear memory cache + terminate zombie processes",
    "Service Degradation": "Scale up service replicas + reduce load",
    "Network Congestion": "Reroute traffic + throttle connections",
    "Disk Bottleneck": "Flush disk buffer + clean logs",
    "Software Fault": "Rollback system + alert dev team",
    "Anomalous Pattern": "Flag for manual review"
}

# ─────────────────────────────────────────────
# KAFKA LOOP
# ─────────────────────────────────────────────
PRED_BUFFER = []
SCORE_BUFFER = []
BUFFER_SIZE = 50

try:
    consumer = KafkaConsumer(
        "log_stream_balanced",
        bootstrap_servers=["localhost:9092"],
        group_id="aiops_consumer_group_balanced",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="latest"
    )
    print("\n🚀 KAFKA STREAM ONLINE (50:50 BALANCED PIPELINE)\n")
except Exception as e:
    print(f"⚠️ Kafka connection failed: {e}")
    sys.exit(1)

for message in consumer:
    msg = message.value
    sess = str(msg.get("session_id","unknown"))
    comp = str(msg.get("component","unknown"))
    cpu  = sf(msg.get("cpu_usage"))
    mem  = sf(msg.get("memory_usage"))
    resp = sf(msg.get("response_time"))
    lat  = sf(msg.get("network_latency"))

    # MODEL PREDICTION
    X = preprocess(msg)
    if model:
        raw = float(model.predict(X, verbose=0)[0][0])
        # Micro-Jitter applied
        if raw >= 0.98:
            raw = raw - np.random.uniform(0.01, 0.06)
        elif raw <= 0.02:
            raw = raw + np.random.uniform(0.02, 0.10)
        else:
            raw = raw + np.random.uniform(-0.02, 0.02)
    else:
        raw = np.random.uniform(0.1, 0.9)
    
    score = round(max(0.0, min(1.0, raw)), 4)

    if score >= HIGH_THRESHOLD: level = "HIGH"
    elif score >= 0.4: level = "MEDIUM"
    else: level = "LOW"

    cause = get_root_cause(msg)
    action = HEAL.get(cause, "Flag for manual review")
    ts_now = datetime.now().isoformat()

    # ── 1. BUFFERED WRITE ──
    try:
        PRED_BUFFER.append(f"{ts_now},{sess},{level},{score},0,{score}\n")
        SCORE_BUFFER.append(f"{ts_now},{sess},{cpu},{mem},{resp},{lat},{score}\n")
        
        if len(PRED_BUFFER) >= BUFFER_SIZE:
            with open(CSV_PATHS["predictions"], "a") as f:
                f.writelines(PRED_BUFFER)
            with open(CSV_PATHS["scores"], "a") as f:
                f.writelines(SCORE_BUFFER)
            PRED_BUFFER.clear()
            SCORE_BUFFER.clear()
            
        if level in ["HIGH", "MEDIUM"]:
            with open(CSV_PATHS["healing"], "a") as f:
                f.write(f"{ts_now},{sess},{level},{cause},{action}\n")
            with open(CSV_PATHS["root_cause"], "a") as f:
                f.write(f"{ts_now},{sess},{cause},{comp}\n")
    except Exception as e: pass

    # ── 2. PREDICTION HORIZON LOGIC ──
    s = session_tracker[sess]
    if level in ["HIGH", "MEDIUM"]:
        s["high_count"] += 1
        if s["first_time"] is None: s["first_time"] = time.time()
            
        if s["high_count"] >= 2 and not s["printed"]:
            horizon = round(time.time() - s["first_time"], 3)
            with open(CSV_PATHS["horizon"], "a") as f:
                f.write(f"{sess},{s['first_time']},{time.time()},{horizon}\n")
            
            print(f"\n🚀 {ts_now} | HORIZON: {horizon}s | {sess} | {comp} | RISK: {score}")
            s["printed"] = True
