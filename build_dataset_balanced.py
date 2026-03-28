"""
PREDICTIVE SYSTEM FAILURE DETECTION USING LOG INTELLIGENCE
Dataset Construction Pipeline - 50:50 Balanced Version
Algorithm: Conditional Gaussian Augmentation (CGA)
Base Data: HDFS_v1 (Loghub)
"""

import os, re, gc
import numpy as np
import pandas as pd
from scipy.stats import entropy
from datetime import datetime
from sklearn.model_selection import train_test_split

RAW_DIR      = os.path.expanduser("~/realistic_log_project/raw_dataset")
OUTPUT_DIR   = os.path.expanduser("~/realistic_log_project/dataset")
HDFS_LOG     = os.path.join(RAW_DIR, "HDFS.log")
ANOMALY_FILE = os.path.join(RAW_DIR, "preprocessed/anomaly_label.csv")
RANDOM_SEED  = 42
CHUNK_SIZE   = 100_000

TELEMETRY_PARAMS = {
    "cpu_usage":       (40.0, 10.0,  87.0,   8.0),
    "memory_usage":    (47.0, 10.0,  82.0,   7.0),
    "disk_io":         (30.0, 10.0, 140.0,  30.0),
    "response_time":  (125.0, 40.0,1750.0, 500.0),
    "network_latency": (30.0, 10.0, 600.0, 200.0),
    "warning_count":   ( 1.0,  0.5,   6.5,   2.0),
    "error_count":     ( 0.5,  0.3,   5.0,   1.5),
}
TELEMETRY_CLIP = {
    "cpu_usage":      (0.0, 100.0),
    "memory_usage":   (0.0, 100.0),
    "disk_io":        (0.0, 500.0),
    "response_time":  (10.0,5000.0),
    "network_latency":(1.0,2000.0),
    "warning_count":  (0, 20),
    "error_count":    (0, 15),
}

HDFS_PAT  = re.compile(r"(\d{6})\s+(\d{6})\s+\d+\s+(\w+)\s+([\w.$]+):\s+(.*)")
BLK_PAT   = re.compile(r"(blk_-?\d+)")
IP_PAT    = re.compile(r"/(\d+\.\d+\.\d+\.\d+)")
LEVEL_MAP = {"INFO":"INFO","WARN":"WARNING","ERROR":"ERROR","FATAL":"ERROR","DEBUG":"INFO"}
COMP_MAP  = {"DataXceiver":"datanode_service","FSNamesystem":"namenode_service",
             "DataBlockScanner":"block_scanner","PacketResponder":"packet_responder",
             "DFSClient":"dfs_client","NameNode":"namenode_service","DataNode":"datanode_service"}
EVT_MAP   = {"receiving":"disk_write","received":"disk_write","served":"request_served",
             "terminating":"service_stop","exception":"service_error","error":"service_error",
             "failed":"service_failure","starting":"service_start","deleting":"data_deletion",
             "writing":"disk_write","reading":"disk_read","heartbeat":"heartbeat",
             "allocate":"resource_allocation","block":"block_operation"}

def pcomp(r):
    for k,v in COMP_MAP.items():
        if k.lower() in r.lower(): return v
    return r.split(".")[-1].lower()[:30]

def pevt(msg):
    m=msg.lower()
    for k,v in EVT_MAP.items():
        if k in m: return v
    return "system_event"

def pstat(lvl, msg):
    if lvl in ["ERROR","FATAL"]: return "failed"
    if any(w in msg.lower() for w in ["fail","error","exception","terminat"]): return "failed"
    return "success"

def pts(d,t):
    try: return datetime.strptime(d+t,"%y%m%d%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
    except: return "2008-11-09 00:00:00"

def load_labels():
    print("[STEP 1] Loading anomaly labels...")
    df = pd.read_csv(ANOMALY_FILE)
    df.columns = df.columns.str.strip()
    lmap = dict(zip(df["BlockId"].str.strip(),
                    df["Label"].str.strip().map({"Normal":0,"Anomaly":1})))
    print(f"         Loaded {len(lmap):,} labels")
    return lmap

def parse_logs(lmap):
    print(f"\n[STEP 2] Parsing logs (Full extraction of HDFS)...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tmp = os.path.join(OUTPUT_DIR,"temp_full.csv")
    total, recs, first = 0, [], True
    cols = ["timestamp","log_level","ip","session_id","component",
            "event_type","message","status","anomaly_label"]

    with open(HDFS_LOG,"r",encoding="utf-8",errors="ignore") as f:
        for line in f:
            m = HDFS_PAT.match(line.strip())
            if not m: continue
            d,t,lvl,comp,msg = m.groups()
            bm = BLK_PAT.search(msg)
            if not bm: continue
            sid = bm.group(1)
            ip  = IP_PAT.search(msg)
            recs.append((pts(d,t), LEVEL_MAP.get(lvl,"INFO"),
                         ip.group(1) if ip else "0.0.0.0",
                         sid, pcomp(comp), pevt(msg),
                         msg[:80].strip(), pstat(lvl,msg),
                         lmap.get(sid,0)))
            if len(recs) >= CHUNK_SIZE:
                pd.DataFrame(recs,columns=cols).to_csv(
                    tmp, mode="w" if first else "a", header=first, index=False)
                total+=len(recs); recs=[]; first=False; gc.collect()
                if total % 500_000 == 0:
                    print(f"         Parsed {total:,} rows from HDFS...")

    if recs:
        pd.DataFrame(recs,columns=cols).to_csv(
            tmp, mode="w" if first else "a", header=first, index=False)
        total+=len(recs); gc.collect()

    print(f"         Done Parsing. Total Raw Logs: {total:,}")
    df = pd.read_csv(tmp, low_memory=False)
    os.remove(tmp)
    
    print("\n[STEP 2.5] Performing 50:50 Balancing...")
    df_anomaly = df[df.anomaly_label == 1]
    df_normal = df[df.anomaly_label == 0]
    
    target_count = len(df_anomaly)
    print(f"         Found {target_count:,} natural anomaly logs.")
    
    if len(df_normal) > target_count:
        df_normal = df_normal.sample(n=target_count, random_state=RANDOM_SEED)
    
    df_balanced = pd.concat([df_anomaly, df_normal]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"         Balanced dataset to exactly 50:50 ({len(df_balanced):,} total rows)")
    
    return df_balanced

def augment(df):
    print("\n[STEP 3] Conditional Gaussian Augmentation (CGA)...")
    rng = np.random.default_rng(RANDOM_SEED)
    n   = len(df); lbl = df["anomaly_label"].values
    for metric,(mu_n,sig_n,mu_a,sig_a) in TELEMETRY_PARAMS.items():
        v = np.where(lbl==0, rng.normal(mu_n,sig_n,n), rng.normal(mu_a,sig_a,n))
        lo,hi = TELEMETRY_CLIP[metric]
        v = np.clip(v,lo,hi)
        v = np.round(v).astype(int) if metric in ["warning_count","error_count"] else np.round(v,2)
        df[metric] = v
        print(f"         {metric:<22} normal={df[df.anomaly_label==0][metric].mean():.1f} anomaly={df[df.anomaly_label==1][metric].mean():.1f}")
    return df

def validate(df):
    print("\n[STEP 4] KL Divergence Validation:")
    print(f"  {'Metric':<22} {'KL_Normal':<12} {'KL_Anomaly':<12} Result")
    print("  "+"-"*55)
    nd=df[df.anomaly_label==0]; ad=df[df.anomaly_label==1]
    for metric,(mu_n,sig_n,mu_a,sig_a) in TELEMETRY_PARAMS.items():
        kls=[]
        for sub,mu,sig in [(nd,mu_n,sig_n),(ad,mu_a,sig_a)]:
            hg,ed=np.histogram(sub[metric].values,bins=30,density=True)
            cx=(ed[:-1]+ed[1:])/2
            he=(1/(sig*np.sqrt(2*np.pi)))*np.exp(-0.5*((cx-mu)/sig)**2)
            hg=(hg+1e-10)/(hg+1e-10).sum(); he=(he+1e-10)/(he+1e-10).sum()
            kls.append(entropy(hg,he))
        print(f"  {metric:<22} {kls[0]:<12.4f} {kls[1]:<12.4f} {'PASS' if all(k<0.5 for k in kls) else 'CHECK'}")
    print()

def export(df):
    cols=["timestamp","log_level","ip","session_id","component","event_type",
          "message","status","anomaly_label","cpu_usage","memory_usage",
          "disk_io","response_time","network_latency","warning_count","error_count"]
    df=df[cols]
    tr,te=train_test_split(df,test_size=0.2,random_state=RANDOM_SEED,stratify=df.anomaly_label)
    
    df.to_csv(os.path.join(OUTPUT_DIR,"structured_logs_balanced.csv"),index=False)
    tr.to_csv(os.path.join(OUTPUT_DIR,"train_logs_balanced.csv"),index=False)
    te.to_csv(os.path.join(OUTPUT_DIR,"test_logs_balanced.csv"),index=False)
    print(f"[STEP 5] Exported 50:50 dataset: {len(df):,} rows | {len(cols)} cols")
    print(f"         Train: {len(tr):,} | Test: {len(te):,}")
    print(f"         Saved to: {OUTPUT_DIR}/structured_logs_balanced.csv")

def main():
    print("="*55)
    print(" DATASET BUILDER — 50:50 Balanced Version")
    print(" Algorithm: Conditional Gaussian Augmentation (CGA)")
    print(" Strategy : Natural Anomalies + Equivalent Natural Normals")
    print("="*55+"\n")
    lmap = load_labels()
    df   = parse_logs(lmap)
    df   = augment(df)
    validate(df)
    export(df)
    print("\n PIPELINE COMPLETE")

if __name__ == "__main__":
    main()
