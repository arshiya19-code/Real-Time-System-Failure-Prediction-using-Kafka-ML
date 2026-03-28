#!/usr/bin/env python3
import sys

for line in sys.stdin:
    line = line.strip()
    if line.startswith("timestamp"):
        continue
    fields = line.split(",")
    if len(fields) >= 9:
        component     = fields[4]   # component column
        anomaly_label = fields[8]   # anomaly_label column
        if anomaly_label == "1":
            print(f"{component}\t1")
