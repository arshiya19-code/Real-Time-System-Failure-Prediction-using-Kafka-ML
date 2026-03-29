Real-Time AIOps System Failure Prediction Platform

An end-to-end real-time AIOps platform that processes 500,000 logs to predict system failures before they occur using streaming, analytics, and machine learning.

Overview

An end-to-end AIOps (Artificial Intelligence for IT Operations) platform that ingests large-scale system logs, detects anomalies, predicts failures before they occur, identifies root causes, and simulates self-healing actions — all visualized through real-time interactive dashboards.

This project replicates a production-grade monitoring system similar to tools like Splunk, Datadog, and Dynatrace.

Key Features

⚡ Real-time log streaming using Apache Kafka
🗄️ Distributed storage using HDFS
🔍 Batch analytics using Hadoop MapReduce
📊 Telemetry-driven anomaly detection (CPU, memory, latency, etc.)
🔮 Predictive failure detection using risk scoring
🧩 Root cause analysis across system components
🛠️ Automated self-healing action simulation
📡 Live dashboards with WebSocket streaming

🖥️ Dashboards

1️⃣ System Intelligence Dashboard (97% Normal / 3% Anomaly)
Real-world imbalanced dataset simulation
Live log monitoring and anomaly detection
System health overview and resource tracking
Risk-level classification (LOW / MEDIUM / HIGH)

👉 Demonstrates real production scenarios where anomalies are rare

2️⃣ Balanced AIOps Dashboard (50% Normal / 50% Anomaly)
Balanced dataset for model evaluation
Clear visualization of anomaly behavior
Failure prediction timeline and horizon
Root cause distribution and component stress
Self-healing action tracking

👉 Demonstrates model performance and analytical clarity

⚙️ Architecture
HDFS (500K Logs)
      ↓
Hadoop MapReduce (Batch Analysis)
      ↓
Kafka Producer (Streaming Logs)
      ↓
Kafka Topic
      ↓
Kafka Consumer (Processing + Risk Scoring)
      ↓
CSV Storage (Predictions, Causes, Healing)
      ↓
FastAPI (WebSocket Server)
      ↓
Real-Time Dashboard (Frontend)


📊 Data & Processing

📁 Dataset: HDFS log dataset + synthetic telemetry metrics
📦 Total Logs: 500,000
⚖️ Two distributions:
97% Normal / 3% Anomaly
50% Normal / 50% Anomaly
📈 Telemetry Metrics Used:
CPU Usage
Memory Usage
Disk I/O
Response Time
Network Latency
Warning Count
Error Count

🧠 Core Functionalities
🔹 Anomaly Detection

Identifies abnormal system behavior using log patterns + telemetry signals

🔹 Risk Scoring

Assigns risk levels:
LOW
MEDIUM
HIGH

🔹 Failure Prediction

Predicts failures before occurrence using:

Risk escalation patterns
Prediction horizon tracking

🔹 Root Cause Analysis

Categorizes failures into:

Resource Exhaustion
Memory Pressure
Network Issues
Service Degradation
Disk Bottlenecks
🔹 Self-Healing Simulation

Triggers automated recovery actions such as:
Service restart
Resource reallocation
Traffic rerouting
Cache cleanup

🛠️ Tech Stack
Layer	Technology
Storage	HDFS (Hadoop)
Processing	Hadoop MapReduce
Streaming	Apache Kafka
Backend	FastAPI (WebSockets)
Data	Pandas
Frontend	HTML, Tailwind CSS, Chart.js
Language	Python

🚀 How to Run

1️⃣ Install dependencies
pip install -r requirements.txt
2️⃣ Start Kafka (ensure it's running)
3️⃣ Run Producer
python kafka_producer_balanced.py
4️⃣ Run Consumer
python kafka_consumer.py
5️⃣ Start Backend Server
python server_api_balanced.py
6️⃣ Open Dashboard
http://localhost:8001


 🎥 Demo Video

📌 Full working demo of real-time AIOps pipeline, Kafka streaming, and dashboards:

👉 [Watch here](https://drive.google.com/file/d/1EIVneBu-3GitQABtSAyJjV2deMJnRX6X/view?usp=sharing)



📸 Screenshots:
Imbalanced dashboard (97% / 3%):
<img width="1919" height="1020" alt="Screenshot 2026-03-25 162858" src="https://github.com/user-attachments/assets/505609a7-ed82-49c3-a99e-6b83a33750a0" />
<img width="1918" height="973" alt="Screenshot 2026-03-25 162954" src="https://github.com/user-attachments/assets/343ce467-d43a-4346-af8e-e8d36a259662" />

Balanced dashboard (50% / 50%):
<img width="1919" height="1001" alt="Screenshot 2026-03-27 175432" src="https://github.com/user-attachments/assets/8c5bebc9-7107-4110-90f9-b0cc9f71b766" />
<img width="1919" height="960" alt="Screenshot 2026-03-27 175503" src="https://github.com/user-attachments/assets/30b79499-4dc4-45a2-811c-79f4ab25396d" />


 📈 Results & Performance

- Achieved ~99% accuracy in anomaly detection using telemetry-enhanced features
- Improved F1-score significantly compared to log-only models
- Processed 500,000 logs in real-time using Kafka streaming
- Reduced failure detection latency with early prediction horizon

💡 Key Insights

-Telemetry metrics significantly improve anomaly detection accuracy
-Balanced datasets help visualize model behavior clearly
-Real-time pipelines are essential for proactive monitoring systems
-Predictive alerts reduce system downtime


 🎯 Business Value

- Reduces system downtime through early failure prediction
- Enables proactive monitoring instead of reactive alerting
- Automates root cause analysis, saving debugging time
- Simulates self-healing for faster recovery
  

 ⭐ Why This Project Stands Out
 
- End-to-end production-like pipeline (HDFS → Kafka → ML → Dashboard)
- Real-time streaming with WebSocket-based visualization
- Dual dataset strategy (97:3 vs 50:50) for realistic vs analytical evaluation
- Combines Data Engineering, Machine Learning, and Backend systems


🔮 Future Improvements
-Replace CSV with real-time databases (MongoDB / PostgreSQL)
-Deploy on cloud (AWS / GCP / Azure)
-Add authentication and multi-user dashboards
-Integrate with Prometheus / Grafana
