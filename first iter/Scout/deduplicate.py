import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import sys

# ---------------- CONFIG ----------------
INPUT_CSV = "detections_log.csv"
OUTPUT_CSV = "waypoints.csv"
CLUSTER_RADIUS_METERS = 3.5
EARTH_RADIUS = 6378137.0  # meters

REQUIRED_COLUMNS = ["target_lat", "target_lon", "det_conf"]

# ---------------- HELPERS ----------------
def safe_read_csv(path):
    try:
        df = pd.read_csv(path)
        print(f"[INFO] Loaded CSV with {len(df)} rows")
        return df
    except FileNotFoundError:
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to read CSV: {e}")
        sys.exit(1)

def validate_columns(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing required columns: {missing}")
        print(f"[INFO] Available columns: {list(df.columns)}")
        sys.exit(1)

def clean_dataframe(df):
    df = df.copy()

    # Force numeric types
    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with NaN
    before = len(df)
    df = df.dropna(subset=REQUIRED_COLUMNS)
    after = len(df)

    print(f"[INFO] Dropped {before - after} rows with NaN values")

    # Remove invalid lat/lon ranges
    df = df[
        (df["target_lat"].between(-90, 90)) &
        (df["target_lon"].between(-180, 180))
    ]

    if df.empty:
        print("[ERROR] No valid geotagged detections remain after cleaning")
        sys.exit(1)

    return df.reset_index(drop=True)

def latlon_to_radians(df):
    return np.radians(df[["target_lat", "target_lon"]].to_numpy())

# ---------------- MAIN ----------------
def deduplicate():
    print("[INFO] Starting deduplication")

    df = safe_read_csv(INPUT_CSV)
    validate_columns(df)
    df = clean_dataframe(df)

    print(f"[INFO] Using {len(df)} cleaned detections")

    coords_rad = latlon_to_radians(df)

    eps_rad = CLUSTER_RADIUS_METERS / EARTH_RADIUS

    try:
        clustering = DBSCAN(
            eps=eps_rad,
            min_samples=1,
            metric="haversine"
        ).fit(coords_rad)
    except Exception as e:
        print(f"[ERROR] DBSCAN failed: {e}")
        sys.exit(1)

    df["cluster_id"] = clustering.labels_
    n_clusters = df["cluster_id"].nunique()
    print(f"[INFO] Found {n_clusters} spatial clusters")

    waypoints = []

    for cluster_id in sorted(df["cluster_id"].unique()):
        cluster = df[df["cluster_id"] == cluster_id]

        # Pick highest-confidence detection
        best = cluster.loc[cluster["det_conf"].idxmax()]

        waypoints.append({
            "wp_id": f"WP_{cluster_id}",
            "lat": float(best["target_lat"]),
            "lon": float(best["target_lon"]),
            "det_conf": float(best["det_conf"]),
            "num_detections": int(len(cluster))
        })

    wp_df = pd.DataFrame(waypoints)

    try:
        wp_df.to_csv(OUTPUT_CSV, index=False)
    except Exception as e:
        print(f"[ERROR] Failed to write output CSV: {e}")
        sys.exit(1)

    print(f"[SUCCESS] Saved {len(wp_df)} waypoints to '{OUTPUT_CSV}'")

# ---------------- ENTRY ----------------
if __name__ == "__main__":
    deduplicate()

