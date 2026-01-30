#!/usr/bin/env python3
"""
delivery_combined_mission_final.py

What it does:
- Listens for HUMAN detections over STATUSTEXT
- Logs detections (optional)
- DBSCAN dedupe -> waypoints.csv
- When >= MIN_FOR_START clusters exist AND pilot has set GUIDED, it starts an autonomous GUIDED mission:
  takeoff -> visit each waypoint -> trigger servo -> RTL -> wait for landing (NO auto disarm on errors)

IMPORTANT SAFETY:
- On any error/abort it will NOT disarm automatically.
- It will stop autonomous control by switching to LOITER (or LAND if you prefer).
"""

import time
import math
import csv
import argparse
import threading
import sys
from pymavlink import mavutil

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

# ---------------- CONFIG ----------------
# Recommended: use two different local UDP ports from mavlink-router:
#  - 14550 for listener (detections)
#  - 14552 for mission control (commands)
LISTEN_CONN = "udpin:0.0.0.0:14550"
FC_CONN = "udp:127.0.0.1:14550"   # <-- change this only if you don't have router endpoint yet

DETECTIONS_LOG = "detections_log.csv"
WAYPOINTS_CSV = "waypoints.csv"

CLUSTER_RADIUS_METERS = 2.0
EARTH_RADIUS = 6378137.0
MIN_FOR_START = 5
SAVE_DETECTIONS = True

# Flight / servo settings
TAKEOFF_ALT = 7.0
CRUISE_ALT = 7.0
ARRIVAL_RADIUS_M = 3.0
WP_TIMEOUT = 90
TAKEOFF_TIMEOUT = 45

SERVO_NUMBERS = [9, 10, 12, 13, 11]
SERVO_PWM_TRIGGER = 1900
SERVO_COMMAND_GAP = 1.0

# When something goes wrong:
# - "LOITER" is a safe hold (requires GPS).
# - You can change to "LAND" if you prefer it to come down.
FAILSAFE_MODE_ON_ERROR = "LOITER"

LOG_COLUMNS = ["ts", "lat", "lon", "conf", "alt"]

# ---------------- globals ----------------
active_detections = []
pending_detections = []
detections_lock = threading.Lock()

mission_running = False
mission_lock = threading.Lock()


# ---------------- parsing helpers ----------------
def parse_statustext_text(text: str):
    """
    Accepts:
      - "HUMAN,lat=...,lon=...,alt=...,conf=..."
      - "H,lat,lon,alt,conf"
      - "H,<id>,lat,lon,alt"   (id ignored)
      - "H,<id>,lat,lon,alt,conf" (id ignored)
    Returns dict or None
    """
    s = text.strip()

    # HUMAN,lat=...,lon=...,alt=...,conf=...
    if s.startswith("HUMAN"):
        parts = s.split(",", 1)
        if len(parts) < 2:
            return None
        body = parts[1]
        out = {}
        for kv in body.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                k = k.strip()
                v = v.strip()
                try:
                    out[k] = float(v)
                except:
                    out[k] = v

        if "lat" in out and "lon" in out:
            c = out.get("conf", None)
            if c is None:
                c = out.get("co", None)
            return {
                "lat": float(out["lat"]),
                "lon": float(out["lon"]),
                "alt": float(out["alt"]) if out.get("alt") is not None else None,
                "conf": float(c) if c is not None else None,
            }
        return None

    # compact CSV-like form:
    # H,lat,lon,alt,conf
    # OR H,<id>,lat,lon,alt,(conf)
    if s.startswith("H,"):
        parts = s.split(",")

        # Detect whether parts[1] is id or lat
        # If it's an int, assume it's id
        idx = 1
        try:
            _maybe_id = int(parts[1])
            idx = 2  # lat starts at parts[2]
        except:
            idx = 1  # lat starts at parts[1]

        try:
            lat = float(parts[idx + 0])
            lon = float(parts[idx + 1])
            alt = float(parts[idx + 2]) if len(parts) > (idx + 2) and parts[idx + 2] != "" else None
            conf = float(parts[idx + 3]) if len(parts) > (idx + 3) and parts[idx + 3] != "" else None
            return {"lat": lat, "lon": lon, "alt": alt, "conf": conf}
        except:
            return None

    return None


# ---------------- logging ----------------
def append_detection_log(ts, lat, lon, conf, alt, path=DETECTIONS_LOG):
    row = {
        "ts": ts,
        "lat": lat,
        "lon": lon,
        "conf": conf if conf is not None else float("nan"),
        "alt": alt if alt is not None else float("nan"),
    }
    write_header = False
    try:
        with open(path, "r"):
            pass
    except FileNotFoundError:
        write_header = True

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------- DBSCAN dedupe ----------------
def run_dbscan_and_write_waypoints(detections_list, cluster_radius_m=CLUSTER_RADIUS_METERS, out_csv=WAYPOINTS_CSV):
    if not detections_list:
        return 0, []

    df = pd.DataFrame(detections_list)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["conf"] = pd.to_numeric(df.get("conf", pd.Series([np.nan] * len(df))), errors="coerce")

    df = df.dropna(subset=["lat", "lon"])
    if df.empty:
        return 0, []

    coords = np.radians(df[["lat", "lon"]].to_numpy())
    eps_rad = cluster_radius_m / EARTH_RADIUS

    clustering = DBSCAN(eps=eps_rad, min_samples=1, metric="haversine").fit(coords)
    df["cluster_id"] = clustering.labels_

    waypoints = []
    for cid in sorted(df["cluster_id"].unique()):
        cluster = df[df["cluster_id"] == cid]
        if "conf" in cluster.columns and cluster["conf"].notna().any():
            best_idx = cluster["conf"].idxmax()
            best = cluster.loc[best_idx]
        else:
            best = cluster.iloc[0]

        waypoints.append({
            "wp_id": f"WP_{cid}",
            "lat": float(best["lat"]),
            "lon": float(best["lon"]),
            "conf": float(best["conf"]) if not pd.isna(best.get("conf", np.nan)) else None,
            "num_detections": int(len(cluster)),
        })

    pd.DataFrame(waypoints).to_csv(out_csv, index=False)
    return len(waypoints), waypoints


# ---------------- mission helper functions ----------------
def wait_heartbeat(master, timeout=10):
    t0 = time.time()
    while time.time() - t0 < timeout:
        hb = master.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
        if hb:
            return True
    return False

def wait_gps_fix(master, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        gps = master.recv_match(type="GPS_RAW_INT", blocking=True, timeout=1)
        if gps:
            fix = int(getattr(gps, "fix_type", 0))
            sats = int(getattr(gps, "satellites_visible", -1))
            print(f"  GPS fix_type={fix}, sats={sats}", end="\r")
            if fix >= 3:
                print()
                return True
    print()
    return False

def wait_mode_via_heartbeat(master, mode_name, timeout=8):
    """
    More reliable than reading master.flightmode alone.
    Forces fresh HEARTBEAT reads.
    """
    t0 = time.time()
    while time.time() - t0 < timeout:
        hb = master.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
        if hb:
            if getattr(master, "flightmode", None) == mode_name:
                return True
    return False

def is_armed_from_hb(hb):
    return (hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0

def wait_armed(master, armed=True, timeout=12):
    t0 = time.time()
    while time.time() - t0 < timeout:
        hb = master.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
        if hb:
            if is_armed_from_hb(hb) == armed:
                return True
    return False

def get_global_pos(master, timeout=1):
    msg = master.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=timeout)
    if not msg:
        return None
    lat = msg.lat / 1e7
    lon = msg.lon / 1e7
    rel_alt = msg.relative_alt / 1000.0
    return lat, lon, rel_alt

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2)**2
    return 2 * R * math.asin(math.sqrt(a))

def wait_command_ack(master, command_id, timeout=5):
    t0 = time.time()
    while time.time() - t0 < timeout:
        ack = master.recv_match(type="COMMAND_ACK", blocking=True, timeout=1)
        if ack and ack.command == command_id:
            return ack.result
    return None

def send_guided_position(master, lat_deg, lon_deg, rel_alt_m):
    lat_int = int(lat_deg * 1e7)
    lon_int = int(lon_deg * 1e7)
    type_mask = (
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
    )
    master.mav.set_position_target_global_int_send(
        0,
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
        type_mask,
        lat_int, lon_int, float(rel_alt_m),
        0, 0, 0,
        0, 0, 0,
        0, 0
    )

def goto_waypoint_guided(master, lat_t, lon_t, alt_t, radius_m=ARRIVAL_RADIUS_M, timeout=WP_TIMEOUT):
    """
    Fix: refresh HEARTBEAT non-blocking in loop so master.flightmode is always fresh.
    """
    t0 = time.time()
    last_dist = None
    last_alt = None

    while time.time() - t0 < timeout:
        # refresh flightmode
        master.recv_match(type="HEARTBEAT", blocking=False)

        if getattr(master, "flightmode", None) != "GUIDED":
            raise RuntimeError(f"Pilot changed mode to {getattr(master,'flightmode',None)} — aborting.")

        send_guided_position(master, lat_t, lon_t, alt_t)

        pos = get_global_pos(master, timeout=1)
        if pos:
            lat, lon, rel_alt = pos
            dist = haversine_m(lat, lon, lat_t, lon_t)
            last_dist = dist
            last_alt = rel_alt
            print(f"  Dist: {dist:6.2f} m | Alt: {rel_alt:4.2f} m", end="\r")

            if dist <= radius_m and rel_alt >= (alt_t - 0.5):
                print()
                return True, last_dist, last_alt

        time.sleep(0.2)

    print()
    return False, last_dist, last_alt

def trigger_servo(master, servo_num, pwm):
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
        0,
        float(servo_num),
        float(pwm),
        0, 0, 0, 0, 0
    )
    ack = wait_command_ack(master, mavutil.mavlink.MAV_CMD_DO_SET_SERVO, timeout=3)
    if ack is not None and ack != mavutil.mavlink.MAV_RESULT_ACCEPTED:
        print(f"?? Servo command rejected for SERVO{servo_num} (ACK result={ack}).")


# ---------------- mission runner ----------------
def mission_runner(fc_conn, waypoints):
    global mission_running

    with mission_lock:
        if mission_running:
            print("[MISSION] already running, abort start.")
            return
        mission_running = True

    master = None
    try:
        print("[MISSION] Connecting to FC:", fc_conn)
        master = mavutil.mavlink_connection(fc_conn)

        if not wait_heartbeat(master, timeout=10):
            raise RuntimeError("No heartbeat from FC.")

        print("[MISSION] Heartbeat OK. Checking GPS...")
        if not wait_gps_fix(master, timeout=30):
            raise RuntimeError("No GPS fix (3D).")

        # Extra: confirm we are receiving mode updates (sanity)
        print("[MISSION] Checking HEARTBEAT mode stream...")
        for _ in range(5):
            master.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
        print(f"[MISSION] Current mode seen as: {getattr(master,'flightmode',None)}")

        print("[MISSION] Setting GUIDED and arming...")
        master.set_mode("GUIDED")
        if not wait_mode_via_heartbeat(master, "GUIDED", timeout=8):
            raise RuntimeError("Failed to enter GUIDED")

        master.arducopter_arm()
        if not wait_armed(master, True, timeout=12):
            raise RuntimeError("Arming failed.")

        print("[MISSION] Takeoff to", TAKEOFF_ALT, "m")
        master.mav.command_long_send(
            master.target_system, master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0,
            0, 0, 0, 0,
            0, 0, TAKEOFF_ALT
        )
        ack = wait_command_ack(master, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, timeout=5)
        if ack is not None and ack != mavutil.mavlink.MAV_RESULT_ACCEPTED:
            raise RuntimeError("TAKEOFF rejected")

        # wait altitude
        t0 = time.time()
        last_alt = None
        while time.time() - t0 < TAKEOFF_TIMEOUT:
            pos = get_global_pos(master, timeout=1)
            if pos:
                _, _, last_alt = pos
                print(f"  Alt: {last_alt:.2f} m", end="\r")
                if last_alt >= (TAKEOFF_ALT - 0.5):
                    break
            time.sleep(0.2)
        print()

        if last_alt is None or last_alt < (TAKEOFF_ALT - 0.5):
            raise RuntimeError("Takeoff failed to reach target altitude.")

        # go through waypoints
        for i, wp in enumerate(waypoints, start=1):
            lat = wp["lat"]
            lon = wp["lon"]
            print(f"[MISSION] WP{i}: going to {lat:.7f},{lon:.7f}")

            ok, dist, alt = goto_waypoint_guided(
                master, lat, lon, CRUISE_ALT,
                radius_m=ARRIVAL_RADIUS_M,
                timeout=WP_TIMEOUT
            )
            if not ok:
                raise RuntimeError(f"Timeout to reach WP{i}")

            print(f"[MISSION] Reached WP{i} dist~{dist:.2f} m")
            servo_num = SERVO_NUMBERS[(i - 1) % len(SERVO_NUMBERS)]
            print(f"[MISSION] Trigger servo {servo_num}")
            trigger_servo(master, servo_num, SERVO_PWM_TRIGGER)
            time.sleep(SERVO_COMMAND_GAP)

        # finished, RTL
        print("[MISSION] Waypoints done. Switching to RTL...")
        master.set_mode("RTL")
        if not wait_mode_via_heartbeat(master, "RTL", timeout=8):
            raise RuntimeError("Failed to enter RTL")

        # wait land
        print("[MISSION] Waiting for landing...")
        t0 = time.time()
        while time.time() - t0 < 240:
            pos = get_global_pos(master, timeout=1)
            if pos:
                _, _, rel_alt = pos
                print(f"  Altitude: {rel_alt:.2f} m", end="\r")
                if rel_alt <= 0.2:
                    break
            time.sleep(0.2)
        print()

        print("[MISSION] Landed. (Not changing servos here automatically.)")
        print("[MISSION] Mission complete — leaving vehicle ARMED state decision to pilot.")
        # If you WANT auto-disarm only after landing, uncomment:
        # master.arducopter_disarm()

    except Exception as e:
        print("[MISSION] ERROR:", e)

        # SAFETY FIX: do NOT disarm automatically.
        # Stop autonomous control by switching to a safe mode.
        try:
            if FAILSAFE_MODE_ON_ERROR:
                print(f"[MISSION] Switching to {FAILSAFE_MODE_ON_ERROR} (no auto-disarm).")
                master.set_mode(FAILSAFE_MODE_ON_ERROR)
        except Exception as e2:
            print("[MISSION] Failed to change mode on error:", e2)

    finally:
        # move pending -> active for next run
        with detections_lock:
            print("[MISSION] Rolling pending detections into active batch")
            active_detections.clear()
            active_detections.extend(pending_detections)
            pending_detections.clear()

        with mission_lock:
            mission_running = False


# ---------------- listener thread ----------------
def listener_thread(listen_conn, min_for_start):
    print("[LISTENER] Starting on", listen_conn)
    m = mavutil.mavlink_connection(listen_conn)

    try:
        m.wait_heartbeat(timeout=10)
        print("[LISTENER] Heartbeat OK")
    except Exception as e:
        print("[LISTENER] Heartbeat wait:", e)

    while True:
        # keep flightmode fresh for pilot approval check
        try:
            m.recv_match(type="HEARTBEAT", blocking=False)
        except:
            pass

        msg = m.recv_match(type="STATUSTEXT", blocking=True, timeout=5)
        if msg is None:
            continue

        raw = msg.text
        text = raw.decode("utf-8", errors="ignore") if isinstance(raw, (bytes, bytearray)) else str(raw)

        if not (text.startswith("HUMAN") or text.startswith("H,")):
            continue

        parsed = parse_statustext_text(text)
        if not parsed:
            continue

        ts = time.time()
        lat = parsed["lat"]
        lon = parsed["lon"]
        alt = parsed.get("alt")
        conf = parsed.get("conf")

        print(f"[LISTENER] HUMAN -> {lat:.7f},{lon:.7f} conf={conf}")

        with detections_lock:
            if mission_running:
                pending_detections.append({"ts": ts, "lat": lat, "lon": lon,
                                          "conf": conf if conf is not None else float("nan"),
                                          "alt": alt})
                df_list = list(active_detections)
                print(f"[LISTENER] Stored in PENDING (pending={len(pending_detections)})")
            else:
                active_detections.append({"ts": ts, "lat": lat, "lon": lon,
                                          "conf": conf if conf is not None else float("nan"),
                                          "alt": alt})
                df_list = list(active_detections)
                print(f"[LISTENER] Stored in ACTIVE (active={len(active_detections)})")

        if SAVE_DETECTIONS:
            try:
                append_detection_log(ts, lat, lon, conf, alt, path=DETECTIONS_LOG)
            except Exception as e:
                print("[LISTENER] Failed to append detection log:", e)

        try:
            n_clusters, waypoints = run_dbscan_and_write_waypoints(
                df_list,
                cluster_radius_m=CLUSTER_RADIUS_METERS,
                out_csv=WAYPOINTS_CSV
            )
        except Exception as e:
            print("[LISTENER] Dedup error:", e)
            n_clusters, waypoints = 0, []

        print(f"[LISTENER] {n_clusters} unique clusters (MIN_FOR_START={min_for_start})")

        if n_clusters >= min_for_start:
            current_mode = getattr(m, "flightmode", None)
            if current_mode != "GUIDED":
                print(f"[LISTENER] Ready — waiting for pilot to set GUIDED (current={current_mode})")
                continue

            with mission_lock:
                if not mission_running:
                    print("[LISTENER] Pilot-approved GUIDED -> starting mission thread")
                    wp_copy = [dict(w) for w in waypoints]
                    t = threading.Thread(target=mission_runner, args=(FC_CONN, wp_copy), daemon=True)
                    t.start()
                else:
                    print("[LISTENER] Mission already running; skipping start")


# ---------------- main ----------------
def main():
    global LISTEN_CONN, FC_CONN, MIN_FOR_START, CLUSTER_RADIUS_METERS, DETECTIONS_LOG

    p = argparse.ArgumentParser()
    p.add_argument("--listen", default=LISTEN_CONN)
    p.add_argument("--fc_conn", default=FC_CONN)
    p.add_argument("--min_start", type=int, default=MIN_FOR_START)
    p.add_argument("--radius", type=float, default=CLUSTER_RADIUS_METERS)
    p.add_argument("--log", default=DETECTIONS_LOG)
    p.add_argument("--takeoff", type=float, default=TAKEOFF_ALT)
    p.add_argument("--cruise", type=float, default=CRUISE_ALT)
    args = p.parse_args()

    LISTEN_CONN = args.listen
    FC_CONN = args.fc_conn
    MIN_FOR_START = args.min_start
    CLUSTER_RADIUS_METERS = args.radius
    DETECTIONS_LOG = args.log

    # allow override of alts
    global TAKEOFF_ALT, CRUISE_ALT
    TAKEOFF_ALT = args.takeoff
    CRUISE_ALT = args.cruise

    t = threading.Thread(target=listener_thread, args=(LISTEN_CONN, MIN_FOR_START), daemon=True)
    t.start()

    print("[MAIN] Delivery dedup listener running. Waiting for missions to trigger.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[MAIN] Exiting (Ctrl+C)")
        sys.exit(0)

if __name__ == "__main__":
    main()
