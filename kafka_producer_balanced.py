"""
=============================================================================
PREDICTIVE SYSTEM FAILURE DETECTION
Kafka Producer — Streams structured_logs_balanced.csv to Kafka topic log_stream_balanced
=============================================================================
"""

import os
import time
import json
import pandas as pd
import numpy as np
import sys
from kafka import KafkaProducer
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATASET_PATH = os.path.expanduser("~/realistic_log_project/dataset/structured_logs_balanced.csv")
KAFKA_BROKER = "localhost:9092"
TOPIC        = "log_stream_balanced"
CHUNK_SIZE        = 5_000 # reduced chunk size since dataset is naturally smaller
BATCH_DELAY       = 0.3     # seconds between batches
BATCH_SIZE        = 10      # logs per batch
RANDOM_SEED       = 42
CHECKPOINT_PATH   = os.path.expanduser("~/realistic_log_project/live/producer_checkpoint_balanced.json")

np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────
def save_checkpoint(round_num, chunk_start):
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump({
            "round_num"  : round_num,
            "chunk_start": chunk_start,
            "saved_at"   : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }, f)

def load_checkpoint():
    if not os.path.exists(CHECKPOINT_PATH):
        return 1, 0
    try:
        with open(CHECKPOINT_PATH) as f:
            data = json.load(f)
        return data["round_num"], data["chunk_start"]
    except:
        return 1, 0

# ─────────────────────────────────────────────
# CONNECT TO KAFKA
# ─────────────────────────────────────────────
print("=" * 60)
print("  KAFKA PRODUCER - 50:50 BALANCED DATASET")
print("=" * 60)
print(f"\n[INIT] Connecting to Kafka at {KAFKA_BROKER}...")

producer = KafkaProducer(
    bootstrap_servers=[KAFKA_BROKER],
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    batch_size=16384,
    linger_ms=10,
    acks=1,
    retries=3
)
print(f"  ✅ Connected to Kafka")

# ─────────────────────────────────────────────
# LOAD DATASET
# ─────────────────────────────────────────────
print(f"\n[INIT] Loading dataset...")
try:
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])

    df_normal  = df[df.anomaly_label == 0].reset_index(drop=True)
    df_anomaly = df[df.anomaly_label == 1].reset_index(drop=True)
    total_rounds = max(1, len(df) // CHUNK_SIZE)

    print(f"  ✅ Loaded {len(df):,} logs")
    print(f"     Normal : {len(df_normal):,} (50%)")
    print(f"     Anomaly: {len(df_anomaly):,} (50%)")
    print(f"     Rounds : {total_rounds} × {CHUNK_SIZE:,} logs")
except Exception as e:
    print(f"⚠️ Could not load CSV dataset: {e}")
    sys.exit(1)

# ─────────────────────────────────────────────
# BUILD STRATIFIED CHUNK
# ─────────────────────────────────────────────
def build_chunk(chunk_start, round_num):
    # Enforce exactly 50:50 for every chunk broadcast
    n_anom = int(CHUNK_SIZE * 0.50)
    n_norm = CHUNK_SIZE - n_anom

    def take_rows(src, start, n):
        if len(src) == 0: return pd.DataFrame() # fail-safe
        start = start % len(src)
        if start + n <= len(src):
            return src.iloc[start:start+n].copy()
        p1 = src.iloc[start:].copy()
        p2 = src.iloc[:n-len(p1)].copy()
        return pd.concat([p1, p2])

    normal_start  = chunk_start % max(len(df_normal), 1)
    anomaly_start = (chunk_start // 2) % max(len(df_anomaly), 1)

    chunk = pd.concat([
        take_rows(df_normal,  normal_start,  n_norm),
        take_rows(df_anomaly, anomaly_start, n_anom)
    ]).sample(frac=1, random_state=RANDOM_SEED + round_num).reset_index(drop=True)

    return chunk

# ─────────────────────────────────────────────
# STREAM LOGS TO KAFKA
# ─────────────────────────────────────────────
print(f"\n[START] Streaming to topic '{TOPIC}'")
print(f"        Batch: {BATCH_SIZE} logs every {BATCH_DELAY}s")
print(f"        Press Ctrl+C to stop\n")

round_num, chunk_start_pos = load_checkpoint()
if round_num > 1 or chunk_start_pos > 0:
    print(f"[CHECKPOINT] Resuming from Round {round_num}, chunk_start={chunk_start_pos:,}")
else:
    print(f"[CHECKPOINT] Fresh start — Round 1")

total_sent = 0

try:
    while True:
        chunk = build_chunk(chunk_start_pos, round_num)
        n_anom = (chunk.anomaly_label == 1).sum()
        n_norm = len(chunk) - n_anom
        total_batches = len(chunk) // BATCH_SIZE
        start_time = time.time()

        print(f"{'═'*55}")
        print(f"  ROUND {round_num}/{total_rounds}")
        print(f"  Logs: {len(chunk):,} ({n_norm:,} normal | {n_anom:,} anomaly)")
        print(f"  ETA : ~{total_batches * BATCH_DELAY / 60:.1f} minutes")
        print(f"{'═'*55}\n")

        for batch_idx in range(total_batches):
            batch = chunk.iloc[batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE]
            now   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            for _, row in batch.iterrows():
                msg = json.loads(row.to_json())
                msg["event_time"] = now
                producer.send(TOPIC, msg)

            total_sent += BATCH_SIZE

            if batch_idx % 500 == 0 or batch_idx == total_batches - 1:
                elapsed   = time.time() - start_time
                pct       = (batch_idx + 1) / total_batches * 100
                remaining = (total_batches - batch_idx - 1) * BATCH_DELAY
                ts        = datetime.now().strftime("%H:%M:%S")
                sample    = batch.iloc[0]
                label     = "🔴 ANOMALY" if sample["anomaly_label"] == 1 else "🟢 NORMAL "
                print(f"[{ts}] Round {round_num} | "
                      f"{batch_idx*BATCH_SIZE:>6,}/{len(chunk):,} ({pct:5.1f}%) | "
                      f"CPU={float(sample.get('cpu_usage',0)):5.1f}% | "
                      f"Sent={total_sent:,} | {label}")

            time.sleep(BATCH_DELAY)

        elapsed = time.time() - start_time
        print(f"\n  ✅ Round {round_num} complete — "
              f"{len(chunk):,} logs in {elapsed/60:.1f} minutes\n")

        next_start = (chunk_start_pos + CHUNK_SIZE) % len(df)
        next_round = (round_num % total_rounds) + 1
        save_checkpoint(next_round, next_start)
        print(f"  💾 Checkpoint saved → Round {next_round}, start={next_start:,}")
        chunk_start_pos = next_start
        round_num = next_round

        if round_num == 1:
            print(f"\n{'='*55}")
            print(f"  ✅ ALL {total_rounds} ROUNDS COMPLETE!")
            print(f"  🔄 Restarting from Round 1...")
            print(f"{'='*55}\n")

except KeyboardInterrupt:
    print(f"\n{'='*55}")
    print(f"  [STOPPED] Producer paused")
    print(f"  Total logs sent: {total_sent:,}")
    print(f"{'='*55}")
    producer.close()
