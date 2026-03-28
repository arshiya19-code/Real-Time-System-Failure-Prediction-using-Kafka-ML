"""
Debug script — checks exact Kafka message fields
Run: python3 ~/realistic_log_project/debug_kafka.py
"""
from kafka import KafkaConsumer
import json

print("Connecting to Kafka...")
c = KafkaConsumer(
    "log_stream",
    bootstrap_servers=["localhost:9092"],
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    auto_offset_reset="earliest",
    group_id="debug_check3",
    consumer_timeout_ms=8000
)

print("Reading messages...\n")
for i, msg in enumerate(c):
    v = msg.value
    print(f"=== MESSAGE {i+1} ===")
    print(f"ALL KEYS    : {list(v.keys())}")
    print(f"session_id  : {v.get('session_id',  'MISSING')}")
    print(f"anomaly_label: {v.get('anomaly_label', 'MISSING')}")
    print(f"cpu_usage   : {v.get('cpu_usage',   'MISSING')}")
    print(f"memory_usage: {v.get('memory_usage','MISSING')}")
    print(f"response_time:{v.get('response_time','MISSING')}")
    print(f"log_level   : {v.get('log_level',   'MISSING')}")
    print(f"component   : {v.get('component',   'MISSING')}")
    print(f"event_type  : {v.get('event_type',  'MISSING')}")
    print(f"status      : {v.get('status',      'MISSING')}")
    print(f"event_time  : {v.get('event_time',  'MISSING')}")
    print()
    if i >= 4:
        break

c.close()
print("Done.")
