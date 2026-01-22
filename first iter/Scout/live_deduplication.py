import socket
import json
import time
import threading
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

# ---------------- CONFIG ----------------
LISTEN_IP = "0.0.0.0"
LISTEN_PORT = 5005

CLUSTER_RADIUS_METERS = 3.5
EARTH_RADIUS = 6378137.0

WAYPOINT_OUTPUT = "waypoints_live.csv"

# ---------------- STATE ----------------
detections = []
detections_lock = threading.Lock()

# ---------------- HELPERS ----------------
def latlon_to_radians(lat, lon):
    return np.radians([lat, lon])

def recompute_waypoints():
    with detections_lock:
        if len(detections) == 0:
            return

        df = pd.DataFrame(detections)

    coords_rad = np.radians(df[["target_lat", "target_lon"]].to_numpy())
    eps_rad = CLUSTER_RADIUS_METERS / EARTH_RADIUS

    clustering = DBSCAN(
        eps=eps_rad,
        min_samples=1,
        metric="haversine"
    ).fit(coords_rad)

    df["cluster_id"] = clustering.labels_

    waypoints = []
    for cid in sorted(df["cluster_id"].unique()):
        cluster = df[df["cluster_id"] == cid]
        best = cluster.loc[cluster["det_conf"].idxmax()]

        waypoints.append({
            "wp_id": f"WP_{cid}",
            "lat": float(best["target_lat"]),
            "lon": float(best["target_lon"]),
            "det_conf": float(best["det_conf"]),
            "num_detections": int(len(cluster))
        })

    wp_df = pd.DataFrame(waypoints)
    wp_df.to_csv(WAYPOINT_OUTPUT, index=False)

    print(f"[WAYPOINT UPDATE] {len(wp_df)} waypoints active")

# ---------------- NETWORK LISTENER ----------------
def udp_listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((LISTEN_IP, LISTEN_PORT))

    print(f"[LISTENING] UDP {LISTEN_IP}:{LISTEN_PORT}")

    while True:
        data, addr = sock.recvfrom(2048)
        try:
            msg = json.loads(data.decode("utf-8"))
        except Exception:
            continue

        required = ["target_lat", "target_lon", "det_conf"]
        if not all(k in msg for k in required):
            continue

        det = {
            "target_lat": float(msg["target_lat"]),
            "target_lon": float(msg["target_lon"]),
            "det_conf": float(msg["det_conf"]),
            "timestamp": msg.get("timestamp", time.time())
        }

        with detections_lock:
            detections.append(det)

        print(f"[RX] Detection @ {det['target_lat']:.7f}, {det['target_lon']:.7f}")

        recompute_waypoints()

# ---------------- MAIN ----------------
if __name__ == "__main__":
    udp_listener()

