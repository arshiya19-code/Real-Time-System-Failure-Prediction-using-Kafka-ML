#!/usr/bin/env python3
import sys

current_event = None
count = 0

for line in sys.stdin:
    line = line.strip()
    parts = line.split("\t")
    if len(parts) != 2:
        continue
    event, val = parts
    val = int(val)
    if event == current_event:
        count += val
    else:
        if current_event:
            print(f"{current_event}\t{count}")
        current_event = event
        count = val

if current_event:
    print(f"{current_event}\t{count}")
