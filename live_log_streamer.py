"""
=============================================================================
PREDICTIVE SYSTEM FAILURE DETECTION USING LOG INTELLIGENCE
Script A — Live Log Streamer (Verified Final)

HOW IT WORKS:
    Reads structured_logs.csv (500,000 logs)
    Streams in chunks of 25,000 logs maintaining 96.2/3.8 ratio
    Writes 10 logs every 0.3s → 12.5 minutes per chunk
    20 chunks × 25,000 = all 500,000 logs per cycle
    Saves checkpoint after every chunk → resumes if VM closes
    After all 20 rounds → restarts from Round 1 continuously

CHECKPOINT:
    Saved to ~/realistic_log_project/live/checkpoint.txt
    Restart script → reads checkpoint → continues from next chunk

Run in Terminal 1:
    python3 live_log_streamer.py
=============================================================================
"""

import os
import time
import numpy as np
import pandas as pd
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

DATASET_PATH    = os.path.expanduser("~/realistic_log_project/dataset/structured_logs.csv")
LIVE_DIR        = os.path.expanduser("~/realistic_log_project/live")
LIVE_LOG_PATH   = os.path.join(LIVE_DIR, "live_logs.csv")
CHECKPOINT_PATH = os.path.join(LIVE_DIR, "checkpoint.txt")

CHUNK_SIZE  = 25_000
BATCH_SIZE  = 10
BATCH_DELAY = 0.3
RANDOM_SEED = 42

os.makedirs(LIVE_DIR, exist_ok=True)
np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────

def save_checkpoint(round_num, chunk_start):
    with open(CHECKPOINT_PATH, "w") as f:
        f.write(f"round={round_num}\n")
        f.write(f"chunk_start={chunk_start}\n")
        f.write(f"saved_at={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

def load_checkpoint():
    if not os.path.exists(CHECKPOINT_PATH):
        return 1, 0
    try:
        data = {}
        with open(CHECKPOINT_PATH) as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    data[k] = v
        return int(data.get("round", 1)), int(data.get("chunk_start", 0))
    except:
        return 1, 0

# ─────────────────────────────────────────────
# BUILD STRATIFIED CHUNK
# ─────────────────────────────────────────────

def build_chunk(df_normal, df_anomaly, chunk_start, round_num):
    n_anom = int(CHUNK_SIZE * 0.038)
    n_norm = CHUNK_SIZE - n_anom

    def take_rows(src, start, n):
        if start + n <= len(src):
            return src.iloc[start:start+n].copy()
        part1 = src.iloc[start:].copy()
        part2 = src.iloc[:n - len(part1)].copy()
        return pd.concat([part1, part2])

    normal_start  = chunk_start % len(df_normal)
    anomaly_start = (chunk_start // 25) % len(df_anomaly)

    chunk = pd.concat([
        take_rows(df_normal,  normal_start,  n_norm),
        take_rows(df_anomaly, anomaly_start, n_anom)
    ]).sample(frac=1, random_state=RANDOM_SEED + round_num).reset_index(drop=True)

    return chunk

# ─────────────────────────────────────────────
# STREAM ONE CHUNK
# ─────────────────────────────────────────────

def stream_chunk(chunk, round_num, total_rounds):
    total_logs    = len(chunk)
    n_anomaly     = (chunk.anomaly_label == 1).sum()
    n_normal      = total_logs - n_anomaly
    total_batches = total_logs // BATCH_SIZE
    logs_written  = 0
    start_time    = time.time()

    print(f"\n{'═'*65}")
    print(f"  ROUND {round_num} OF {total_rounds}")
    print(f"  Logs    : {total_logs:,}  ({n_normal:,} normal | {n_anomaly:,} anomaly)")
    print(f"  Batches : {total_batches:,} × {BATCH_SIZE} logs every {BATCH_DELAY}s")
    print(f"  ETA     : ~{total_batches * BATCH_DELAY / 60:.1f} minutes")
    print(f"{'═'*65}\n")

    for batch_idx in range(total_batches):
        batch = chunk.iloc[batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE].copy()
        batch["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(LIVE_LOG_PATH, "a") as f:
            for _, row in batch.iterrows():
                f.write(",".join(str(v) for v in row.values) + "\n")

        logs_written += BATCH_SIZE

        # Progress every 500 batches
        if batch_idx % 500 == 0 or batch_idx == total_batches - 1:
            elapsed   = time.time() - start_time
            pct       = logs_written / total_logs * 100
            remaining = (total_batches - batch_idx - 1) * BATCH_DELAY
            ts        = datetime.now().strftime("%H:%M:%S")
            sample    = batch.iloc[0]
            label     = "🔴 ANOMALY" if sample["anomaly_label"] == 1 else "🟢 NORMAL "
            print(f"[{ts}] Round {round_num}/{total_rounds} | "
                  f"{logs_written:>6,}/{total_logs:,} ({pct:5.1f}%) | "
                  f"CPU={float(sample['cpu_usage']):5.1f}% | "
                  f"Resp={float(sample['response_time']):7.1f}ms | "
                  f"Elapsed {elapsed/60:.1f}m | ETA {remaining/60:.1f}m | {label}")

        time.sleep(BATCH_DELAY)

    elapsed = time.time() - start_time
    print(f"\n  ✅ Round {round_num} complete — "
          f"{logs_written:,} logs in {elapsed/60:.1f} minutes\n")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("="*65)
    print("  LIVE LOG STREAMER")
    print(f"  Dataset    : structured_logs.csv (500,000 logs)")
    print(f"  Chunk size : {CHUNK_SIZE:,} logs per round")
    print(f"  Batch      : {BATCH_SIZE} logs every {BATCH_DELAY}s")
    print(f"  Per round  : ~{CHUNK_SIZE/BATCH_SIZE*BATCH_DELAY/60:.1f} minutes")
    print(f"  Ratio      : 96.2% normal / 3.8% anomaly per chunk")
    print(f"  Checkpoint : saves after every round — resumes on restart")
    print(f"  Press Ctrl+C to stop")
    print("="*65)

    # Load dataset
    print("\n[INIT] Loading dataset...")
    df = pd.read_csv(DATASET_PATH, low_memory=False)

    # Drop timestamp if exists — we add fresh timestamps
    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])

    df_normal  = df[df.anomaly_label == 0].reset_index(drop=True)
    df_anomaly = df[df.anomaly_label == 1].reset_index(drop=True)
    total_rounds = len(df) // CHUNK_SIZE

    print(f"       Total   : {len(df):,}")
    print(f"       Normal  : {len(df_normal):,} ({len(df_normal)/len(df)*100:.1f}%)")
    print(f"       Anomaly : {len(df_anomaly):,} ({len(df_anomaly)/len(df)*100:.1f}%)")
    print(f"       Rounds  : {total_rounds} × {CHUNK_SIZE:,} = {total_rounds*CHUNK_SIZE:,} logs per cycle")

    # Load checkpoint
    start_round, chunk_start = load_checkpoint()
    if start_round > 1 or chunk_start > 0:
        print(f"\n[CHECKPOINT] Resuming from Round {start_round} (chunk_start={chunk_start:,})")
    else:
        print(f"\n[CHECKPOINT] Fresh start — Round 1")
        # Write header only on fresh start
        header_cols = list(df.columns) + ["timestamp"]
        with open(LIVE_LOG_PATH, "w") as f:
            f.write(",".join(header_cols) + "\n")
        print(f"[INIT] Live log file created: {LIVE_LOG_PATH}")

    print(f"\n[START] Streaming {total_rounds} rounds of {CHUNK_SIZE:,} logs each\n")

    round_num       = start_round
    chunk_start_pos = chunk_start

    try:
        while True:
            chunk = build_chunk(df_normal, df_anomaly, chunk_start_pos, round_num)
            stream_chunk(chunk, round_num, total_rounds)

            # Save checkpoint
            next_start = (chunk_start_pos + CHUNK_SIZE) % len(df)
            next_round = (round_num % total_rounds) + 1
            save_checkpoint(next_round, next_start)
            print(f"  💾 Checkpoint saved → Round {next_round}, start={next_start:,}")

            round_num       = next_round
            chunk_start_pos = next_start

            if round_num == 1:
                print(f"\n{'='*65}")
                print(f"  ✅ ALL {total_rounds} ROUNDS COMPLETE — 500,000 logs streamed!")
                print(f"  🔄 Restarting from Round 1...")
                print(f"{'='*65}\n")

    except KeyboardInterrupt:
        print(f"\n\n{'='*65}")
        print(f"  [STOPPED] Paused at Round {round_num}")
        print(f"  Checkpoint saved — restart to continue")
        print(f"  Run: python3 live_log_streamer.py")
        print(f"{'='*65}")

if __name__ == "__main__":
    main()
