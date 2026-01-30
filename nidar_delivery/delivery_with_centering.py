# delivery_with_centering.py
#!/usr/bin/env python3
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

# New imports for centering + vision
import cv2
from ultralytics import YOLO
from centering import CenteringParams, run_centering_routine


# ---------------- CAMERA HELPERS (JETSON SAFE) ----------------
def try_open_gst(pipeline, timeout_s=2.0):
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                return cap
        time.sleep(0.05)
    cap.release()
    return None


def open_best_capture():
    pipelines = [
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1, format=NV12 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! "
    "appsink drop=true sync=false",

    "nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1,format=NV12' ! "
    "nvvidconv ! 'video/x-raw, format=(string)BGRx, width=(int)1280, height=(int)720' ! "
    "videoconvert ! 'video/x-raw, format=(string)BGR' ! appsink max-buffers=1 drop=true sync=false",

    "v4l2src device=/dev/video0 ! 'video/x-bayer,format=RG10,width=1920,height=1080,framerate=30/1' ! "
    "bayer2rgb ! videoconvert ! 'video/x-raw, format=(string)BGR' ! appsink max-buffers=1 drop=true sync=false",

    "v4l2src device=/dev/video0 ! 'video/x-raw, format=(string)YUY2, width=(int)1280, height=(int)720, framerate=(fraction)30/1' ! "
    "videoconvert ! 'video/x-raw, format=(string)BGR' ! appsink max-buffers=1 drop=true sync=false"




    ]

    for p in pipelines:
        cap = try_open_gst(p)
        if cap is not None:
            print("[CAMERA] Using pipeline:\n", p)
            return cap

    print("[CAMERA] ERROR: No pipeline worked")
    return None


# ---------------- CONFIG (tweak as needed) ----------------
LISTEN_CONN = "udpin:0.0.0.0:14550"   # listen here for STATUSTEXT
FC_CONN = "udp:127.0.0.1:14550"       # conn string used by mission thread to talk to FC
DETECTIONS_LOG = "detections_log.csv"
WAYPOINTS_CSV = "waypoints.csv"
CLUSTER_RADIUS_METERS = 2.0
EARTH_RADIUS = 6378137.0  # meters
MIN_FOR_START = 5         # unique clusters required to start mission
SAVE_DETECTIONS = True    # append each detection to CSV log

# Flight / servo settings (tune to your drone)
TAKEOFF_ALT = 5.0
CRUISE_ALT = 5.0
ARRIVAL_RADIUS_M = 3.0
WP_TIMEOUT = 90
TAKEOFF_TIMEOUT = 45
SERVO_NUMBERS = [9, 10, 12, 13, 11]
SERVO_PWM_TRIGGER = 1900
SERVO_COMMAND_GAP = 1.0

# Centering / vision defaults (adjust path/index)
YOLO_MODEL_PATH = "model2.engine"  # path to your engine or .pt
CAMERA_INDEX = 0

# CSV log columns
LOG_COLUMNS = ["ts", "lat", "lon", "conf", "alt"]

# ---------------- globals ----------------
active_detections = []
pending_detections = []

detections_lock = threading.Lock()

mission_running = False
mission_lock = threading.Lock()

# ---------------- parsing helpers ----------------
def parse_statustext_text(text: str):
    s = text.strip()
    if s.startswith("HUMAN"):
        parts = s.split(",", 1)
        if len(parts) < 2:
            return None
        body = parts[1]
        out = {}
        for kv in body.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                try:
                    out[k.strip()] = float(v)
                except:
                    out[k.strip()] = v
        if "lat" in out and "lon" in out:
            c = out.get("conf", None)
            if c is None:
                c = out.get("co", None)
            return {"lat": float(out["lat"]), "lon": float(out["lon"]),
                    "alt": float(out.get("alt")) if out.get("alt") is not None else None,
                    "conf": float(c) if c is not None else None}
        return None

    if s.startswith("H,"):
        parts = s.split(",")
        try:
            lat = float(parts[1])
            lon = float(parts[2])
            alt = float(parts[3]) if len(parts) > 3 and parts[3] != "" else None
            conf = float(parts[4]) if len(parts) > 4 and parts[4] != "" else None
            return {"lat": lat, "lon": lon, "alt": alt, "conf": conf}
        except Exception:
            return None

    return None

# ---------------- logging ----------------
def append_detection_log(ts, lat, lon, conf, alt, path=DETECTIONS_LOG):
    row = {"ts": ts, "lat": lat, "lon": lon, "conf": conf if conf is not None else float("nan"),
           "alt": alt if alt is not None else float("nan")}
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
    df["conf"] = pd.to_numeric(df.get("conf", pd.Series([np.nan]*len(df))), errors="coerce")

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
            "num_detections": int(len(cluster))
        })

    wp_df = pd.DataFrame(waypoints)
    wp_df.to_csv(out_csv, index=False)
    return wp_df.shape[0], waypoints

# ---------------- mission helper functions ----------------
def wait_heartbeat(master, timeout=10):
    t0 = time.time()
    while time.time() - t0 < timeout:
        hb = master.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
        if hb:
            return True
    return False

def wait_mode(master, mode_name, timeout=8):
    t0 = time.time()
    while time.time() - t0 < timeout:
        master.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
        if getattr(master, "flightmode", None) == mode_name:
            return True
    return False

def is_armed_from_hb(hb):
    return (hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0

def wait_armed(master, armed=True, timeout=10):
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
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))

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

def wait_command_ack(master, command_id, timeout=5):
    t0 = time.time()
    while time.time() - t0 < timeout:
        ack = master.recv_match(type="COMMAND_ACK", blocking=True, timeout=1)
        if ack and ack.command == command_id:
            return ack.result
    return None

def safe_land(master):
    try:
        master.set_mode("LAND")
        wait_mode(master, "LAND", timeout=5)
    except:
        pass

def safe_rtl(master):
    try:
        master.set_mode("RTL")
        wait_mode(master, "RTL", timeout=5)
    except:
        pass

def safe_disarm(master, timeout=8):
    try:
        print("[MISSION] Disarming vehicle (servo outputs will be released)")
        master.arducopter_disarm()

        t0 = time.time()
        while time.time() - t0 < timeout:
            hb = master.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
            if hb:
                armed = (hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0
                if not armed:
                    print("[MISSION] Disarm confirmed — servos are now free")
                    return True
        print("⚠️ Disarm not confirmed (check vehicle state)")
    except Exception as e:
        print("[MISSION] Disarm error:", e)

    return False

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
        lat_int,
        lon_int,
        float(rel_alt_m),
        0,0,0, 0,0,0, 0,0
    )

def goto_waypoint_guided(master, lat_t, lon_t, alt_t, radius_m=ARRIVAL_RADIUS_M, timeout=WP_TIMEOUT):
    t0 = time.time()
    last_dist = None; last_alt = None
    while time.time() - t0 < timeout:
        if getattr(master, "flightmode", None) != "GUIDED":
            raise RuntimeError(f"Pilot changed mode to {master.flightmode} — aborting.")
        send_guided_position(master, lat_t, lon_t, alt_t)
        pos = get_global_pos(master, timeout=1)
        if pos:
            lat, lon, rel_alt = pos
            dist = haversine_m(lat, lon, lat_t, lon_t)
            last_dist = dist; last_alt = rel_alt
            print(f"  Dist: {dist:6.2f} m | Alt: {rel_alt:4.2f} m", end="\r")
            if dist <= radius_m and rel_alt >= (alt_t - 0.5):
                print()
                return True, last_dist, last_alt
        time.sleep(0.5)
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
        0,0,0,0,0
    )
    ack = wait_command_ack(master, mavutil.mavlink.MAV_CMD_DO_SET_SERVO, timeout=3)
    if ack is not None and ack != mavutil.mavlink.MAV_RESULT_ACCEPTED:
        print(f"⚠️ Servo command rejected for SERVO{servo_num} (ACK result={ack}).")

# ---------------- mission runner (modified to include centering) ----------------
def mission_runner(fc_conn, waypoints):
    global mission_running
    with mission_lock:
        if mission_running:
            print("[MISSION] already running, abort start.")
            return
        mission_running = True

    master = None
    cap = None
    model = None
    try:
        print("[MISSION] Connecting to FC:", fc_conn)
        master = mavutil.mavlink_connection(fc_conn)
        if not wait_heartbeat(master, timeout=10):
            raise RuntimeError("No heartbeat from FC.")

        print("[MISSION] Heartbeat OK. Checking GPS...")
        if not wait_gps_fix(master, timeout=30):
            raise RuntimeError("No GPS fix (3D).")

        print("[MISSION] Setting GUIDED and arming...")
        master.set_mode("GUIDED")
        if not wait_mode(master, "GUIDED", timeout=8):
            raise RuntimeError("Failed to enter GUIDED")

        master.arducopter_arm()
        if not wait_armed(master, True, timeout=12):
            raise RuntimeError("Arming failed.")

        print("[MISSION] Takeoff to", TAKEOFF_ALT, "m")
        master.mav.command_long_send(
            master.target_system, master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0,
            0,0,0,0, 0,0, TAKEOFF_ALT
        )
        ack = wait_command_ack(master, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, timeout=5)
        if ack is not None and ack != mavutil.mavlink.MAV_RESULT_ACCEPTED:
            raise RuntimeError("TAKEOFF rejected")

        # wait altitude
        t0 = time.time(); last_alt = None
        while time.time() - t0 < TAKEOFF_TIMEOUT:
            pos = get_global_pos(master, timeout=1)
            if pos:
                _,_,last_alt = pos
                print(f"  Alt: {last_alt:.2f} m", end="\r")
                if last_alt >= (TAKEOFF_ALT - 0.5):
                    break
            time.sleep(0.2)
        print()

        if last_alt is None or last_alt < (TAKEOFF_ALT - 0.5):
            raise RuntimeError("Takeoff failed to reach target altitude.")

        # --- Prepare vision system (load model + camera) ---
        try:
            print("[MISSION] Loading YOLO model:", YOLO_MODEL_PATH)
            model = YOLO(YOLO_MODEL_PATH, task='detect')
            cap = open_best_capture()
            if cap is None:
                print("[MISSION] Camera unavailable — centering will be skipped")
            else:
                print("[MISSION] Camera opened successfully")

            cent_params = CenteringParams()
            cent_params.CAMERA_INDEX = CAMERA_INDEX
        except Exception as e:
            print("[MISSION] Vision system failed to init:", e)
            model = None
            cap = None
            cent_params = CenteringParams()

        # go through waypoints
        for i, wp in enumerate(waypoints, start=1):
            lat = wp["lat"]; lon = wp["lon"]
            print(f"[MISSION] WP{i}: going to {lat:.7f},{lon:.7f}")
            ok, dist, alt = goto_waypoint_guided(master, lat, lon, CRUISE_ALT, radius_m=ARRIVAL_RADIUS_M, timeout=WP_TIMEOUT)
            if not ok:
                raise RuntimeError(f"Timeout to reach WP{i}")
            print(f"[MISSION] Reached WP{i} dist~{dist:.2f} m")

            # If vision is available, try centering
            print(f"[MISSION] Running centering on WP{i}...")
            centering_success = False

            if model is not None and cap is not None:
                try:
                    centering_success = run_centering_routine(master, model, cap, cent_params)
                except Exception as e:
                    print(f"[MISSION] Centering error on WP{i}:", e)
            else:
                print("[MISSION] Vision unavailable — skipping centering")

            # Always stop before drop
            send_body_velocity(master, 0, 0, 0)
            time.sleep(0.3)

            if centering_success:
                print(f"[MISSION] 🎯 Centering successful on WP{i}")
            else:
                print(f"[MISSION] ⚠️ Centering failed / timeout on WP{i} — DROPPING ANYWAY")

            # 🔴 UNCONDITIONAL DROP (matches your test script)
            servo_num = SERVO_NUMBERS[(i-1) % len(SERVO_NUMBERS)]
            print(f"[MISSION] 📦 Dropping payload using SERVO{servo_num}")
            trigger_servo(master, servo_num, SERVO_PWM_TRIGGER)
            time.sleep(SERVO_COMMAND_GAP)


        # finished, RTL
        print("[MISSION] Waypoints done. Switching to RTL...")
        master.set_mode("RTL")
        if not wait_mode(master, "RTL", timeout=8):
            raise RuntimeError("Failed to enter RTL")

        # wait land
        print("[MISSION] Waiting for landing...")
        t0 = time.time()
        while time.time() - t0 < 180:
            pos = get_global_pos(master, timeout=1)
            if pos:
                _, _, rel_alt = pos
                print(f"  Altitude: {rel_alt:.2f} m", end="\r")
                if rel_alt <= 0.2:
                    break
            time.sleep(0.2)
        print()
        print("[MISSION] Landed. Setting payload servos -> OPEN (1900)")
        for s in SERVO_NUMBERS:
            trigger_servo(master, s, SERVO_PWM_TRIGGER)
            time.sleep(0.05)
        print("[MISSION] Disarming vehicle")
        safe_disarm(master)
        print("[MISSION] Mission complete — waiting for pilot reset & GUIDED")

    except Exception as e:
        print("[MISSION] ERROR:", e)
        if master:
            safe_land(master)
            safe_disarm(master)
    finally:
        # cleanup vision
        try:
            if cap:
                cap.release()
                cv2.destroyAllWindows()
        except:
            pass

        with detections_lock:
            print("[MISSION] Rolling pending detections into active batch")
            active_detections.clear()
            active_detections.extend(pending_detections)
            pending_detections.clear()

        with mission_lock:
            mission_running = False

# ---------------- listener thread (unchanged, from your original) ----------------
def listener_thread(listen_conn, min_for_start):
    print("[LISTENER] Starting on", listen_conn)
    m = mavutil.mavlink_connection(listen_conn)
    try:
        m.wait_heartbeat(timeout=10)
        print("[LISTENER] Heartbeat OK")
    except Exception as e:
        print("[LISTENER] Heartbeat wait:", e)

    while True:
        try:
            hb = m.recv_match(type="HEARTBEAT", blocking=False)
        except Exception:
            pass

        msg = m.recv_match(type="STATUSTEXT", blocking=True, timeout=5)
        if msg is None:
            continue

        try:
            raw = msg.text
            text = raw.decode("utf-8", errors="ignore") if isinstance(raw, (bytes, bytearray)) else str(raw)
        except Exception:
            text = str(msg.text)

        if text.startswith("HUMAN") or text.startswith("H,"):
            print(f"[LISTENER] STATUSTEXT received: {text}")

        parsed = parse_statustext_text(text)
        if not parsed:
            continue

        ts = time.time()
        lat = parsed["lat"]; lon = parsed["lon"]; alt = parsed.get("alt"); conf = parsed.get("conf")
        print(f"[LISTENER] HUMAN -> {lat:.7f},{lon:.7f} conf={conf}")

        with detections_lock:
            if mission_running:
                pending_detections.append({
                    "ts": ts,
                    "lat": lat,
                    "lon": lon,
                    "conf": conf if conf is not None else float("nan"),
                    "alt": alt
                })
                print(f"[LISTENER] Stored detection in PENDING batch (pending={len(pending_detections)})")
            else:
                active_detections.append({
                    "ts": ts,
                    "lat": lat,
                    "lon": lon,
                    "conf": conf if conf is not None else float("nan"),
                    "alt": alt
                })
                print(f"[LISTENER] Stored detection in ACTIVE batch (active={len(active_detections)})")

            df_list = list(active_detections)

        if SAVE_DETECTIONS:
            try:
                append_detection_log(ts, lat, lon, conf, alt, path=DETECTIONS_LOG)
            except Exception as e:
                print("[LISTENER] Failed to append detection log:", e)

        try:
            n_clusters, waypoints = run_dbscan_and_write_waypoints(df_list, cluster_radius_m=CLUSTER_RADIUS_METERS, out_csv=WAYPOINTS_CSV)
        except Exception as e:
            print("[LISTENER] Dedup error:", e)
            n_clusters = 0
            waypoints = []

        print(f"[LISTENER] {n_clusters} unique clusters (MIN_FOR_START={min_for_start})")

        if n_clusters >= min_for_start:
            current_mode = getattr(m, "flightmode", None)
            if current_mode != "GUIDED":
                print(f"[LISTENER] {n_clusters} waypoints ready — waiting for pilot to set GUIDED (current={current_mode})")
                continue

            with mission_lock:
                if not mission_running:
                    print("[LISTENER] Pilot-approved GUIDED detected -> Triggering mission thread")
                    wp_copy = [dict(w) for w in waypoints]
                    t = threading.Thread(target=mission_runner, args=(FC_CONN, wp_copy), daemon=True)
                    t.start()
                else:
                    print("[LISTENER] Mission already running; skipping start")

# ---------------- main ----------------
def main():
    global LISTEN_CONN, FC_CONN, MIN_FOR_START, CLUSTER_RADIUS_METERS, DETECTIONS_LOG, YOLO_MODEL_PATH, CAMERA_INDEX
    p = argparse.ArgumentParser()
    p.add_argument("--listen", default=LISTEN_CONN)
    p.add_argument("--fc_conn", default=FC_CONN)
    p.add_argument("--min_start", type=int, default=MIN_FOR_START)
    p.add_argument("--radius", type=float, default=CLUSTER_RADIUS_METERS)
    p.add_argument("--log", default=DETECTIONS_LOG)
    p.add_argument("--model", default=YOLO_MODEL_PATH)
    p.add_argument("--cam", type=int, default=CAMERA_INDEX)
    args = p.parse_args()

    LISTEN_CONN = args.listen; FC_CONN = args.fc_conn; MIN_FOR_START = args.min_start
    CLUSTER_RADIUS_METERS = args.radius; DETECTIONS_LOG = args.log
    YOLO_MODEL_PATH = args.model; CAMERA_INDEX = args.cam

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
