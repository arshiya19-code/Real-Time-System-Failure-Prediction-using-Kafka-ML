import pandas as pd
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import os, asyncio, uvicorn
import io
import collections
 
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])
 
BASE = os.path.dirname(os.path.abspath(__file__))
LIVE_DIR = BASE
 
# ── File map ────────────────────────────────────────────────────────────────
FILES = {
    "scores":      "risk_scores_balanced.csv",
    "predictions": "predictions_balanced.csv",
    "healing":     "self_healing_log_balanced.csv",
    "horizon":     "prediction_horizon_balanced.csv",
    "causes":      "root_cause_balanced.csv",
}
 
# ── Startup check ────────────────────────────────────────────────────────────
print("\n📂 Checking required CSV files...")
for key, fname in FILES.items():
    full = os.path.join(LIVE_DIR, fname)
    exists = os.path.exists(full)
    size   = os.path.getsize(full) if exists else 0
    print(f"  {'✅' if exists and size > 0 else '❌'} {fname} — {'EXISTS' if exists else 'MISSING'}, {size} bytes")
 
print(f"\n📄 Serving HTML: {os.path.join(BASE, 'index_balanced.html')}")
print(f"   Exists: {os.path.exists(os.path.join(BASE, 'index_balanced.html'))}\n")
 
 
# ── Helpers ──────────────────────────────────────────────────────────────────
def get_total_lines(path):
    if not os.path.exists(path):
        return 0
    try:
        with open(path, "rb") as f:
            return sum(1 for _ in f) - 1          # subtract header
    except Exception as e:
        print(f"[line-count error] {path}: {e}")
        return 0
 
 
def get_high_anomalies(path):
    if not os.path.exists(path):
        return 0
    try:
        df = pd.read_csv(path, usecols=["risk_level"])
        return int((df["risk_level"] == "HIGH").sum())
    except Exception as e:
        print(f"[high-count error] {path}: {e}")
        return 0
 
 
def tail_csv(path, n=50):
    if not os.path.exists(path):
        print(f"[tail_csv] FILE MISSING: {path}")
        return []
    try:
        size = os.path.getsize(path)
        if size == 0:
            print(f"[tail_csv] EMPTY FILE: {path}")
            return []
 
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            header = f.readline()
            if not header.strip():
                return []
            tail_lines = collections.deque(f, maxlen=n + 10)
 
        if not tail_lines:
            print(f"[tail_csv] NO DATA ROWS: {path}")
            return []
 
        csv_data = header + "".join(tail_lines)
        df = pd.read_csv(
            io.StringIO(csv_data),
            on_bad_lines="skip",
            engine="python"
        )
        if df.empty:
            return []

        # ✅ CRITICAL FIX: The old index_balanced.html in your VM will crash if a row
        # has an empty timestamp or is missing the 'T'. We drop those proactively!
        if 'timestamp' in df.columns:
            df = df.dropna(subset=['timestamp'])
            df['timestamp'] = df['timestamp'].astype(str)
            df = df[df['timestamp'].str.contains("T", na=False)]
            
        if 'horizon_seconds' in df.columns:
            df = df.dropna(subset=['horizon_seconds'])

        import json
        json_str = df.tail(n).fillna("").to_json(orient="records")
        return json.loads(json_str)
  
    except Exception as e:
        print(f"[tail_csv ERROR] {path}: {e}")
        return []
 
 
# ── WebSocket ────────────────────────────────────────────────────────────────
@app.websocket("/ws/data")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client = websocket.client
    print(f"[WS] Client connected: {client}")
 
    send_count = 0
    while True:
        try:
            scores      = tail_csv(os.path.join(LIVE_DIR, FILES["scores"]),      60)
            predictions = tail_csv(os.path.join(LIVE_DIR, FILES["predictions"]), 25)
            healing     = tail_csv(os.path.join(LIVE_DIR, FILES["healing"]),     20)
            horizon     = tail_csv(os.path.join(LIVE_DIR, FILES["horizon"]),     20)
            causes      = tail_csv(os.path.join(LIVE_DIR, FILES["causes"]),      80)
 
            data = {
                "scores":      scores,
                "predictions": predictions,
                "healing":     healing,
                "horizon":     horizon,
                "causes":      causes,
                "log_count":   get_total_lines(os.path.join(LIVE_DIR, FILES["predictions"])),
                "high_count":  get_high_anomalies(os.path.join(LIVE_DIR, FILES["predictions"])),
            }
 
            # Log every 10th send so you can see data is flowing
            send_count += 1
            if send_count % 10 == 1:
                print(
                    f"[WS] send #{send_count} | "
                    f"scores={len(scores)} preds={len(predictions)} "
                    f"healing={len(healing)} horizon={len(horizon)} causes={len(causes)}"
                )
 
            await websocket.send_json(data)
            await asyncio.sleep(1)
 
        except Exception as e:
            print(f"[WS] Connection closed: {e}")
            break
 
 
# ── Debug endpoint — open in browser to verify CSVs are readable ─────────────
@app.get("/debug")
def debug():
    result = {}
    for key, fname in FILES.items():
        path = os.path.join(LIVE_DIR, fname)
        rows = tail_csv(path, 3)
        result[key] = {
            "file":   fname,
            "exists": os.path.exists(path),
            "size":   os.path.getsize(path) if os.path.exists(path) else 0,
            "sample": rows[:2],
        }
    return JSONResponse(result)
 
 
# ── Serve HTML (always open via FastAPI, not directly as a file) ──────────────
@app.get("/")
def index():
    html_path = os.path.join(BASE, "index_balanced.html")
    if not os.path.exists(html_path):
        return JSONResponse({"error": "index_balanced.html not found in " + BASE}, status_code=404)
    return FileResponse(html_path)
 
 
if __name__ == "__main__":
    print("🚀 Balanced Dashboard API starting → http://localhost:8001")
    print("   Debug endpoint  → http://localhost:8001/debug\n")
    uvicorn.run(app, host="0.0.0.0", port=8001)
