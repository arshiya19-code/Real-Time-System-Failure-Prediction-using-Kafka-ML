import pandas as pd
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os, asyncio, uvicorn

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])

BASE = os.path.dirname(os.path.abspath(__file__))

def get_total_lines(path):
    if not os.path.exists(path): return 0
    try:
        with open(path, 'rb') as f:
            return sum(1 for _ in f) - 1
    except: return 0

def get_high_anomalies(path):
    if not os.path.exists(path): return 0
    try:
        df = pd.read_csv(path, usecols=['risk_level'])
        return int((df['risk_level'] == 'HIGH').sum())
    except: return 0

def tail_csv(path, n=50):
    if not os.path.exists(path): return []
    try:
        df = pd.read_csv(path)
        if df.empty: return []
        # fillna('') is critical to prevent NaN breaking JSON parsers in front-end
        return df.tail(n).fillna('').to_dict(orient="records")
    except: return []

@app.websocket("/ws/data")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            data = {
                "scores": tail_csv(os.path.join(BASE, "risk_scores.csv"), 60),
                "predictions": tail_csv(os.path.join(BASE, "predictions.csv"), 25),
                "healing": tail_csv(os.path.join(BASE, "self_healing_log.csv"), 20),
                "horizon": tail_csv(os.path.join(BASE, "prediction_horizon.csv"), 20),
                "causes": tail_csv(os.path.join(BASE, "root_cause.csv"), 80),
                "log_count": get_total_lines(os.path.join(BASE, "predictions.csv")),
                "high_count": get_high_anomalies(os.path.join(BASE, "predictions.csv"))
            }
            await websocket.send_json(data)
            await asyncio.sleep(1) # Broadcast cleanly every 1 second without stacking
        except Exception:
            break

@app.get("/")
def index():
    return FileResponse(os.path.join(BASE, "index.html"))

if __name__ == "__main__":
    print("🚀 System Failure Prediction API Server starting on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
