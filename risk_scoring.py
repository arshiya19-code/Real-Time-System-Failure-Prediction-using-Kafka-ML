"""
=============================================================================
PREDICTIVE SYSTEM FAILURE DETECTION USING LOG INTELLIGENCE
Step 1 — Risk Scoring

For each session in structured_logs.csv:
    - Computes anomaly probability from telemetry + log features
    - Applies risk thresholds to assign Low / Medium / High
    - Saves risk_scores.csv to results directory

RISK FORMULA:
    Score = weighted combination of:
        - Normalized CPU usage
        - Normalized response time
        - Normalized network latency
        - Normalized error count
        - Anomaly label probability

THRESHOLDS (derived from MapReduce Job 4 results):
    Low    : score < 0.25
    Medium : 0.25 <= score < 0.50
    High   : score >= 0.50
=============================================================================
"""

import os
import gc
import numpy as np
import pandas as pd

DATASET_PATH = os.path.expanduser("~/realistic_log_project/dataset/structured_logs.csv")
RESULTS_DIR  = os.path.expanduser("~/realistic_log_project/results")
OUTPUT_PATH  = os.path.join(RESULTS_DIR, "risk_scores.csv")
RANDOM_SEED  = 42

os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────────

print("[STEP 1] Loading dataset...")
df = pd.read_csv(DATASET_PATH, low_memory=False)
print(f"         Rows    : {len(df):,}")
print(f"         Normal  : {(df.anomaly_label==0).sum():,}")
print(f"         Anomaly : {(df.anomaly_label==1).sum():,}")

# ─────────────────────────────────────────────
# STEP 2 — SESSION-LEVEL AGGREGATION
# ─────────────────────────────────────────────

print("\n[STEP 2] Aggregating by session...")

session_df = df.groupby("session_id").agg(
    anomaly_label    = ("anomaly_label",    "max"),
    log_count        = ("anomaly_label",    "count"),
    error_count_avg  = ("error_count",      "mean"),
    warning_count_avg= ("warning_count",    "mean"),
    cpu_avg          = ("cpu_usage",        "mean"),
    cpu_max          = ("cpu_usage",        "max"),
    memory_avg       = ("memory_usage",     "mean"),
    memory_max       = ("memory_usage",     "max"),
    disk_io_avg      = ("disk_io",          "mean"),
    disk_io_max      = ("disk_io",          "max"),
    response_avg     = ("response_time",    "mean"),
    response_max     = ("response_time",    "max"),
    latency_avg      = ("network_latency",  "mean"),
    latency_max      = ("network_latency",  "max"),
    log_level_error  = ("log_level",        lambda x: (x=="ERROR").sum()),
    log_level_warn   = ("log_level",        lambda x: (x=="WARNING").sum()),
    failed_events    = ("status",           lambda x: (x=="failed").sum()),
    component        = ("component",        lambda x: x.mode()[0]),
    event_type       = ("event_type",       lambda x: x.mode()[0]),
).reset_index()

print(f"         Total sessions : {len(session_df):,}")
print(f"         Normal sessions: {(session_df.anomaly_label==0).sum():,}")
print(f"         Anomaly sessions: {(session_df.anomaly_label==1).sum():,}")

# ─────────────────────────────────────────────
# STEP 3 — NORMALIZE FEATURES TO [0,1]
# ─────────────────────────────────────────────

print("\n[STEP 3] Normalizing features...")

def normalize(series):
    mn, mx = series.min(), series.max()
    return (series - mn) / (mx - mn + 1e-8)

session_df["cpu_norm"]      = normalize(session_df["cpu_max"])
session_df["memory_norm"]   = normalize(session_df["memory_max"])
session_df["response_norm"] = normalize(session_df["response_max"])
session_df["latency_norm"]  = normalize(session_df["latency_max"])
session_df["error_norm"]    = normalize(session_df["error_count_avg"])
session_df["disk_norm"]     = normalize(session_df["disk_io_max"])
session_df["failed_norm"]   = normalize(session_df["failed_events"])

# ─────────────────────────────────────────────
# STEP 4 — COMPUTE RISK SCORE
# ─────────────────────────────────────────────

print("\n[STEP 4] Computing risk scores...")

# Weighted formula — response time and latency are strongest predictors
# (proven by MapReduce Job 4: 14x and 20x difference)
session_df["risk_score"] = (
    0.25 * session_df["response_norm"] +   # response time — strongest signal
    0.20 * session_df["latency_norm"]  +   # network latency — 20x difference
    0.20 * session_df["cpu_norm"]      +   # CPU usage — 2.17x difference
    0.15 * session_df["error_norm"]    +   # error count — direct failure signal
    0.10 * session_df["memory_norm"]   +   # memory — secondary signal
    0.10 * session_df["failed_norm"]       # failed events — from MapReduce Job 3
)

# Round to 4 decimal places
session_df["risk_score"] = session_df["risk_score"].round(4)

# ─────────────────────────────────────────────
# STEP 5 — ASSIGN RISK LEVELS
# ─────────────────────────────────────────────

print("\n[STEP 5] Assigning risk levels...")

def assign_risk(score):
    if score >= 0.50:
        return "HIGH"
    elif score >= 0.25:
        return "MEDIUM"
    else:
        return "LOW"

session_df["risk_level"] = session_df["risk_score"].apply(assign_risk)

low    = (session_df.risk_level=="LOW").sum()
medium = (session_df.risk_level=="MEDIUM").sum()
high   = (session_df.risk_level=="HIGH").sum()
total  = len(session_df)

print(f"         LOW    : {low:,}  ({low/total*100:.1f}%)")
print(f"         MEDIUM : {medium:,}  ({medium/total*100:.1f}%)")
print(f"         HIGH   : {high:,}  ({high/total*100:.1f}%)")

# ─────────────────────────────────────────────
# STEP 6 — SAVE RESULTS
# ─────────────────────────────────────────────

print("\n[STEP 6] Saving risk scores...")

output_cols = [
    "session_id", "risk_score", "risk_level", "anomaly_label",
    "log_count", "cpu_avg", "cpu_max", "memory_avg", "memory_max",
    "disk_io_avg", "disk_io_max", "response_avg", "response_max",
    "latency_avg", "latency_max", "error_count_avg", "warning_count_avg",
    "failed_events", "log_level_error", "log_level_warn",
    "component", "event_type"
]

session_df[output_cols].sort_values(
    "risk_score", ascending=False
).to_csv(OUTPUT_PATH, index=False)

print(f"         Saved : {OUTPUT_PATH}")

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────

print("\n" + "="*55)
print("  RISK SCORING COMPLETE")
print("="*55)
print(f"  Total sessions : {total:,}")
print(f"  LOW risk       : {low:,} ({low/total*100:.1f}%)")
print(f"  MEDIUM risk    : {medium:,} ({medium/total*100:.1f}%)")
print(f"  HIGH risk      : {high:,} ({high/total*100:.1f}%)")
print(f"\n  Top 5 Highest Risk Sessions:")
print(session_df[output_cols].sort_values(
    "risk_score", ascending=False).head(5)[
    ["session_id","risk_score","risk_level","cpu_max",
     "response_max","latency_max","component"]].to_string(index=False))
print("="*55)
