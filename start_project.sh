#!/bin/bash
# ============================================================
# PREDICTIVE SYSTEM FAILURE DETECTION — STARTUP SCRIPT
# Run this every time you open your VM:
#   bash ~/start_project.sh
# ============================================================

echo "============================================"
echo "  PREDICTIVE FAILURE DETECTION — STARTUP"
echo "============================================"

# ── STEP 1: HADOOP ──────────────────────────────
echo ""
echo "[1/4] Starting Hadoop..."
start-dfs.sh > /dev/null 2>&1
start-yarn.sh > /dev/null 2>&1
echo "      ✅ Hadoop started"

# ── STEP 2: KAFKA ───────────────────────────────
echo ""
echo "[2/4] Starting Kafka..."
pkill -f kafka > /dev/null 2>&1
sleep 2
rm -rf /tmp/kraft-combined-logs

KAFKA_CLUSTER_ID=$(~/kafka/bin/kafka-storage.sh random-uuid 2>/dev/null)
~/kafka/bin/kafka-storage.sh format \
    -t $KAFKA_CLUSTER_ID \
    -c ~/kafka/config/kraft/server.properties > /dev/null 2>&1

export KAFKA_HEAP_OPTS="-Xmx512m -Xms256m"
nohup ~/kafka/bin/kafka-server-start.sh \
    ~/kafka/config/kraft/server.properties \
    > ~/kafka/kafka.log 2>&1 &

echo "      ⏳ Waiting for Kafka to start..."
sleep 10

# ── STEP 3: KAFKA TOPIC ─────────────────────────
echo ""
echo "[3/4] Setting up Kafka topic..."
~/kafka/bin/kafka-topics.sh --create \
    --topic log_stream \
    --bootstrap-server localhost:9092 \
    --partitions 1 \
    --replication-factor 1 > /dev/null 2>&1

TOPICS=$(~/kafka/bin/kafka-topics.sh --list --bootstrap-server localhost:9092 2>/dev/null)
if echo "$TOPICS" | grep -q "log_stream"; then
    echo "      ✅ Topic log_stream ready"
else
    echo "      ❌ Topic creation failed — check kafka.log"
fi

# ── STEP 4: DIRECTORIES ─────────────────────────
echo ""
echo "[4/4] Setting up directories..."
mkdir -p ~/realistic_log_project/live
cd ~/realistic_log_project
echo "      ✅ Ready in ~/realistic_log_project"

# ── SUMMARY ─────────────────────────────────────
echo ""
echo "============================================"
echo "  EVERYTHING IS READY!"
echo "============================================"
echo ""
echo "  Now open 3 more terminal tabs and run:"
echo ""
echo "  Tab 2: python3 ~/realistic_log_project/kafka_producer.py"
echo "  Tab 3: python3 ~/realistic_log_project/kafka_consumer.py"
echo "  Tab 4: streamlit run ~/realistic_log_project/dashboard.py --server.port 8501"
echo ""
echo "  Dashboard: http://localhost:8501"
echo "============================================"
