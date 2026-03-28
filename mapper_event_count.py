#!/usr/bin/env python3
import sys

for line in sys.stdin:
    line = line.strip()
    if line.startswith("timestamp"):
        continue
    fields = line.split(",")
    if len(fields) >= 6:
        event_type = fields[5]  # event_type column
        print(f"{event_type}\t1")
