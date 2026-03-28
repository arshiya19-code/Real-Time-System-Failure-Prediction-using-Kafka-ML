#!/usr/bin/env python3
import sys

current_session = None
count = 0

for line in sys.stdin:
    line = line.strip()
    parts = line.split("\t")
    if len(parts) != 2:
        continue
    session, val = parts
    val = int(val)
    if session == current_session:
        count += val
    else:
        if current_session:
            if count >= 3:  # Only report sessions with 3+ failures
                print(f"{current_session}\t{count}")
        current_session = session
        count = val

if current_session:
    if count >= 3:
        print(f"{current_session}\t{count}")
