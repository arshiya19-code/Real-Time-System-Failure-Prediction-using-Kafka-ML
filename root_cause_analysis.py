"""
=============================================================================
PREDICTIVE SYSTEM FAILURE DETECTION USING LOG INTELLIGENCE
Step 2 — Root Cause Analysis

Reads: ~/realistic_log_project/results/risk_scores.csv
For each HIGH and MEDIUM risk session:
    - Identifies primary root cause from telemetry thresholds
    - Lists all contributing causes
    - Records specific indicators (values that triggered each cause)

ROOT CAUSE CATEGORIES:
    Resource Exhaustion  — CPU spike above normal avg (40%)
    Memory Pressure      — Memory above normal avg (47%)
    Service Degradation  — Response time above normal avg (125ms)
    Network Congestion   — Latency above normal avg (30ms)
    Disk Bottleneck      — Disk IO above normal avg (30 MB/s)
    Software Fault       — Error count above normal avg (0.5)
    Repeated Failures    — Failed events > 3 in session

THRESHOLDS derived from MapReduce Job 4 telemetry analysis:
    Normal avg CPU      : 39.99%   → threshold 70%
    Normal avg Memory   : 47.02%   → threshold 65%
    Normal avg Response : 125.11ms → threshold 500ms
    Normal avg Latency  : 30.00ms  → threshold 200ms
    Normal avg Disk IO  : 30.01MB/s→ threshold 80MB/s
    Normal avg Errors   : 0.5      → threshold 2.0

Saves: ~/realistic_log_project/results/root_cause.csv
=============================================================================
"""

import os
import numpy as np
import pandas as pd

RISK_SCORES_PATH = os.path.expanduser("~/realistic_log_project/results/risk_scores.csv")
RESULTS_DIR      = os.path.expanduser("~/realistic_log_project/results")
OUTPUT_PATH      = os.path.join(RESULTS_DIR, "root_cause.csv")

# Thresholds from MapReduce Job 4 results
CPU_THRESHOLD      = 70.0
MEMORY_THRESHOLD   = 65.0
RESPONSE_THRESHOLD = 500.0
LATENCY_THRESHOLD  = 200.0
DISK_THRESHOLD     = 80.0
ERROR_THRESHOLD    = 2.0
FAILED_THRESHOLD   = 3

# ─────────────────────────────────────────────
# STEP 1 — LOAD RISK SCORES
# ─────────────────────────────────────────────

print("[STEP 1] Loading risk scores...")
df = pd.read_csv(RISK_SCORES_PATH)
print(f"         Total sessions : {len(df):,}")
print(f"         HIGH           : {(df.risk_level=='HIGH').sum():,}")
print(f"         MEDIUM         : {(df.risk_level=='MEDIUM').sum():,}")
print(f"         LOW            : {(df.risk_level=='LOW').sum():,}")

# ─────────────────────────────────────────────
# STEP 2 — FILTER HIGH + MEDIUM SESSIONS
# ─────────────────────────────────────────────

print("\n[STEP 2] Filtering HIGH and MEDIUM sessions...")
target = df[df.risk_level.isin(["HIGH", "MEDIUM"])].copy().reset_index(drop=True)
print(f"         Sessions to analyse : {len(target):,}")

# ─────────────────────────────────────────────
# STEP 3 — ROOT CAUSE IDENTIFICATION
# ─────────────────────────────────────────────

print("\n[STEP 3] Identifying root causes...")

def identify_root_cause(row):
    causes     = []
    indicators = []

    if row["cpu_avg"] > CPU_THRESHOLD:
        causes.append("Resource Exhaustion")
        indicators.append(f"CPU={row['cpu_avg']:.1f}%")

    if row["memory_avg"] > MEMORY_THRESHOLD:
        causes.append("Memory Pressure")
        indicators.append(f"Memory={row['memory_avg']:.1f}%")

    if row["response_avg"] > RESPONSE_THRESHOLD:
        causes.append("Service Degradation")
        indicators.append(f"ResponseTime={row['response_avg']:.0f}ms")

    if row["latency_avg"] > LATENCY_THRESHOLD:
        causes.append("Network Congestion")
        indicators.append(f"Latency={row['latency_avg']:.0f}ms")

    if row["disk_io_avg"] > DISK_THRESHOLD:
        causes.append("Disk Bottleneck")
        indicators.append(f"DiskIO={row['disk_io_avg']:.1f}MB/s")

    if row["error_count_avg"] > ERROR_THRESHOLD:
        causes.append("Software Fault")
        indicators.append(f"Errors={row['error_count_avg']:.1f}")

    if row["failed_events"] > FAILED_THRESHOLD:
        causes.append("Repeated Failures")
        indicators.append(f"FailedEvents={int(row['failed_events'])}")

    if not causes:
        causes.append("Anomalous Pattern")
        indicators.append("Combined metric deviation")

    primary       = causes[0]
    all_causes    = " | ".join(causes)
    all_indicators= " | ".join(indicators)
    cause_count   = len(causes)

    return primary, all_causes, all_indicators, cause_count

results = target.apply(identify_root_cause, axis=1, result_type="expand")
target["primary_cause"]  = results[0]
target["all_causes"]     = results[1]
target["indicators"]     = results[2]
target["cause_count"]    = results[3]

# ─────────────────────────────────────────────
# STEP 4 — SEVERITY LABEL
# ─────────────────────────────────────────────

print("\n[STEP 4] Adding severity labels...")

def severity_label(row):
    if row["risk_level"] == "HIGH" and row["cause_count"] >= 3:
        return "CRITICAL"
    elif row["risk_level"] == "HIGH":
        return "SEVERE"
    else:
        return "WARNING"

target["severity"] = target.apply(severity_label, axis=1)

critical = (target.severity=="CRITICAL").sum()
severe   = (target.severity=="SEVERE").sum()
warning  = (target.severity=="WARNING").sum()

print(f"         CRITICAL : {critical:,}")
print(f"         SEVERE   : {severe:,}")
print(f"         WARNING  : {warning:,}")

# ─────────────────────────────────────────────
# STEP 5 — SAVE OUTPUT
# ─────────────────────────────────────────────

print("\n[STEP 5] Saving root cause analysis...")

output_cols = [
    "session_id", "risk_score", "risk_level", "severity",
    "component", "primary_cause", "all_causes", "indicators", "cause_count",
    "cpu_avg", "memory_avg", "response_avg", "latency_avg",
    "disk_io_avg", "error_count_avg", "failed_events"
]

target[output_cols].sort_values(
    ["risk_score", "cause_count"], ascending=[False, False]
).to_csv(OUTPUT_PATH, index=False)

print(f"         Saved : {OUTPUT_PATH}")

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────

print("\n" + "="*65)
print("  ROOT CAUSE ANALYSIS COMPLETE")
print("="*65)
print(f"  Sessions analysed : {len(target):,}")
print(f"  CRITICAL          : {critical:,}")
print(f"  SEVERE            : {severe:,}")
print(f"  WARNING           : {warning:,}")

print(f"\n  Primary Cause Distribution:")
cause_dist = target["primary_cause"].value_counts()
for cause, count in cause_dist.items():
    pct = count / len(target) * 100
    print(f"    {cause:<25} : {count:>5} ({pct:.1f}%)")

print(f"\n  Component Failure Distribution (HIGH risk only):")
comp_dist = target[target.risk_level=="HIGH"]["component"].value_counts()
for comp, count in comp_dist.items():
    pct = count / (target.risk_level=="HIGH").sum() * 100
    print(f"    {comp:<25} : {count:>5} ({pct:.1f}%)")

print(f"\n  Top 5 Critical Sessions:")
top5 = target[output_cols].sort_values(
    ["risk_score","cause_count"], ascending=[False,False]).head(5)
for _, row in top5.iterrows():
    print(f"    {str(row.session_id)[:30]:<30} | {row.severity:<8} | "
          f"{row.component:<20} | {row.primary_cause:<25} | Score={row.risk_score:.4f}")
print("="*65)
