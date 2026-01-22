import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

# ---------------- CONFIG ----------------
INPUT_CSV = "detections_raw.csv"
OUTPUT_CSV = "waypoints.csv"
CLUSTER_RADIUS_METERS = 4.0
EARTH_RADIUS = 6378137.0  # meters

# ---------------- HELPERS ----------------
def latlon_to_radians(df):
    """Convert lat/lon columns to radians for haversine."""
    return np.radians(df[["lat", "lon"]].values)

# ---------------- MAIN ----------------
def deduplicate():
    print("[INFO] Loading detections...")
    df = pd.read_csv(INPUT_CSV)

    if df.empty:
        print("[WARN] No detections found.")
        return

    print(f"[INFO] Loaded {len(df)} raw detections")

    # Convert lat/lon to radians for haversine distance
    coords_rad = latlon_to_radians(df)

    # DBSCAN with haversine metric
    eps_rad = CLUSTER_RADIUS_METERS / EARTH_RADIUS

    clustering = DBSCAN(
        eps=eps_rad,
        min_samples=1,
        metric="haversine"
    ).fit(coords_rad)

    df["cluster_id"] = clustering.labels_

    print(f"[INFO] Found {df['cluster_id'].nunique()} clusters")

    waypoints = []

    for cluster_id in sorted(df["cluster_id"].unique()):
        cluster_points = df[df["cluster_id"] == cluster_id]

        # Pick highest confidence detection in cluster
        best = cluster_points.loc[cluster_points["confidence"].idxmax()]

        waypoints.append({
            "wp_id": f"WP_{cluster_id}",
            "lat": best["lat"],
            "lon": best["lon"],
            "confidence": best["confidence"],
            "num_detections": len(cluster_points)
        })

    wp_df = pd.DataFrame(waypoints)
    wp_df.to_csv(OUTPUT_CSV, index=False)

    print(f"[SUCCESS] Saved {len(wp_df)} waypoints to '{OUTPUT_CSV}'")

# ---------------- ENTRY ----------------
if __name__ == "__main__":
    deduplicate()
