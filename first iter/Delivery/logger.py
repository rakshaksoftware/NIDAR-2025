import csv
import time
import os

LOG_FILE = "delivery_log.csv"


if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "state",
            "waypoint_index",
            "message"
        ])

def log_event(state, wp_index, message):
    ts = time.time()

    # Console (real-time feedback)
    print(f"[LOG] {state} | WP {wp_index} | {message}")

    # CSV (post-flight analysis)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            ts,
            state,
            wp_index,
            message
        ])

def log_telemetry(vehicle, state, wp_index):
    try:
        alt = vehicle.location.global_relative_frame.alt
        mode = vehicle.mode.name
        batt = vehicle.battery.voltage if vehicle.battery else None

        msg = f"alt={alt:.2f}m mode={mode} batt={batt}"
        log_event(state, wp_index, msg)
    except Exception:
        pass
