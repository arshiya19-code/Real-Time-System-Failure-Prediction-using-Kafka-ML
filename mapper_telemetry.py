#!/usr/bin/env python3
import sys

for line in sys.stdin:
    line = line.strip()
    if line.startswith("timestamp"):
        continue
    fields = line.split(",")
    if len(fields) >= 16:
        anomaly_label  = fields[8]
        cpu_usage      = fields[9]
        memory_usage   = fields[10]
        disk_io        = fields[11]
        response_time  = fields[12]
        network_latency= fields[13]
        try:
            label = int(anomaly_label)
            cpu   = float(cpu_usage)
            mem   = float(memory_usage)
            disk  = float(disk_io)
            resp  = float(response_time)
            net   = float(network_latency)
            key   = "ANOMALY" if label == 1 else "NORMAL"
            print(f"{key}\t{cpu},{mem},{disk},{resp},{net}")
        except:
            continue
