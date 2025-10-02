import requests
import pandas as pd
import time
import os

PROMETHEUS = "http://localhost:9090"
TARGETS = ["https://google.com", "https://example.com"]
STEP = 15  # seconds
END = int(time.time())
START = END - 600  # last 10 minutes

def query_range(query):
    url = f"{PROMETHEUS}/api/v1/query_range"
    params = {
        "query": query,
        "start": START,
        "end": END,
        "step": STEP
    }
    response = requests.get(url, params=params)
    data = response.json()
    if data["status"] == "success" and data["data"]["result"]:
        values = data["data"]["result"][0]["values"]
        df = pd.DataFrame(values, columns=["timestamp", "value"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df["value"] = df["value"].astype(float)
        return df
    return pd.DataFrame()

all_data = []

for target in TARGETS:
    latency_df = query_range(f'probe_duration_seconds{{instance="{target}"}}')
    success_df = query_range(f'probe_success{{instance="{target}"}}')

    if latency_df.empty or success_df.empty:
        print(f"⚠️ No data for {target}")
        continue

    df = pd.merge(latency_df, success_df, on="timestamp", suffixes=("_latency", "_success"))
    df["target"] = target

    # Derived metrics
    df["latency"] = df["value_latency"]
    df["packet_loss"] = 1 - df["value_success"]
    df["jitter"] = df["latency"].rolling(window=3).std()

    df.drop(columns=["value_latency", "value_success"], inplace=True)
    all_data.append(df)

# ➕ Throughput metrics from windows_exporter
# Sent and Received in bytes/sec
sent_df = query_range("rate(wmi_net_bytes_sent_total[1m])")
recv_df = query_range("rate(wmi_net_bytes_received_total[1m])")

if not sent_df.empty and not recv_df.empty:
    throughput_df = pd.merge(sent_df, recv_df, on="timestamp", suffixes=("_sent", "_recv"))
    throughput_df.rename(columns={
        "value_sent": "throughput_sent_bps",
        "value_recv": "throughput_recv_bps"
    }, inplace=True)
else:
    throughput_df = pd.DataFrame()
    print("⚠️ No throughput data found")

# Merge everything
if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)
    if not throughput_df.empty:
        combined_df = pd.merge(combined_df, throughput_df, on="timestamp", how="left")

    # Safe column selection based on what's available
    columns = ["timestamp", "target", "latency", "jitter", "packet_loss"]
    if "throughput_sent_bps" in combined_df.columns and "throughput_recv_bps" in combined_df.columns:
        columns += ["throughput_sent_bps", "throughput_recv_bps"]

    combined_df = combined_df[columns]


    os.makedirs("data", exist_ok=True)
    combined_df.to_csv("data/metrics.csv", index=False)
    print("✅ Exported data/metrics.csv with all 4 features.")
else:
    print("⚠️ No usable target data collected.")
