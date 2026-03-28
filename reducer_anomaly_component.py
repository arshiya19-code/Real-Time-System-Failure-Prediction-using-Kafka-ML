#!/usr/bin/env python3
import sys

current_component = None
count = 0

for line in sys.stdin:
    line = line.strip()
    parts = line.split("\t")
    if len(parts) != 2:
        continue
    component, val = parts
    val = int(val)
    if component == current_component:
        count += val
    else:
        if current_component:
            print(f"{current_component}\t{count}")
        current_component = component
        count = val

if current_component:
    print(f"{current_component}\t{count}")
