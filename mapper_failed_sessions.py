#!/usr/bin/env python3
import sys

for line in sys.stdin:
    line = line.strip()
    if line.startswith("timestamp"):
        continue
    fields = line.split(",")
    if len(fields) >= 8:
        session_id = fields[3]   # session_id column
        status     = fields[7]   # status column
        if status == "failed":
            print(f"{session_id}\t1")
