#!/usr/bin/env python3
import sys

data = {"NORMAL": [], "ANOMALY": []}

for line in sys.stdin:
    line = line.strip()
    parts = line.split("\t")
    if len(parts) != 2:
        continue
    key, values = parts
    if key not in data:
        continue
    try:
        vals = [float(v) for v in values.split(",")]
        data[key].append(vals)
    except:
        continue

for key in ["NORMAL", "ANOMALY"]:
    rows = data[key]
    if not rows:
        continue
    n = len(rows)
    avg_cpu  = sum(r[0] for r in rows) / n
    avg_mem  = sum(r[1] for r in rows) / n
    avg_disk = sum(r[2] for r in rows) / n
    avg_resp = sum(r[3] for r in rows) / n
    avg_net  = sum(r[4] for r in rows) / n
    print(f"{key}\tcount={n} avg_cpu={avg_cpu:.2f}% avg_memory={avg_mem:.2f}% "
          f"avg_disk_io={avg_disk:.2f}MB/s avg_response_time={avg_resp:.2f}ms "
          f"avg_network_latency={avg_net:.2f}ms")
