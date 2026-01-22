#!/usr/bin/env python3
import cv2
import time
import csv
import os
import numpy as np
from datetime import datetime
from ultralytics import YOLO

from scout_statustext_sender import send_human
from geo_projection import geolocate_target_from_pixel
from telemetry import telemetry, telemetry_lock

# =========================
# SETTINGS
# =========================
ENGINE_PATH = "best.engine"        # path to your TensorRT engine
imgsz = 640                        # inference size (engine must be compatible)
display_window_name = "Disaster Drone - Human Detection"
log_csv = "detections_log.csv"
save_crops_dir = "human_crops"
os.makedirs(save_crops_dir, exist_ok=True)

# If you want this to auto-run at boot with systemd (no monitor/login), set True.
# It will NOT open any GUI windows (imshow/namedWindow).
HEADLESS = True

# Camera intrinsics placeholders (replace after calibration)
fx, fy, cx, cy = 666.65327, 891.26635, 339.31718, 252.0422
K = (fx, fy, cx, cy)

R_cam_to_body = np.array([
    [-1.0,  0.0,  0.0],
    [ 0.0, -1.0,  0.0],
    [ 0.0,  0.0,  1.0]
], dtype=float)

# Detection tuning
CONF_THRESH = 0.5
save_cooldown_s = 5.0  # save at most one crop every N seconds


# =========================
# WAIT FOR TELEMETRY
# =========================
print("[MAIN] Waiting for telemetry (GPS)...")
t0 = time.time()
while True:
    with telemetry_lock:
        lat = telemetry.get("lat")
        lon = telemetry.get("lon")
        h   = telemetry.get("h_agl")

    if lat is not None and lon is not None and h is not None:
        print(f"[MAIN] Telemetry ready: lat={lat}, lon={lon}, h={h}")
        break

    if time.time() - t0 > 30:
        print("[MAIN] WARNING: telemetry not ready after 30s, continuing anyway")
        break

    time.sleep(0.2)


# =========================
# LOAD MODEL
# =========================
print(f"Loading model engine: {ENGINE_PATH}")
model = YOLO(ENGINE_PATH, task="detect")  # be explicit
print("Model loaded.")


# =========================
# CSV HEADER
# =========================
try:
    with open(log_csv, "x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp_utc", "lat_drone", "lon_drone", "h_agl",
            "confidence", "det_class", "lat", "lon", "image"
        ])
except FileExistsError:
    pass


# =========================
# HELPERS
# =========================
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize and pad image to fit new_shape while keeping aspect ratio.
    Returns: canvas, scale, pad_x, pad_y
    """
    h, w = img.shape[:2]
    new_h, new_w = new_shape
    scale = min(new_w / w, new_h / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    img_resized = cv2.resize(img, (nw, nh))
    canvas = np.full((new_h, new_w, 3), color, dtype=np.uint8)
    pad_x = (new_w - nw) // 2
    pad_y = (new_h - nh) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = img_resized
    return canvas, scale, pad_x, pad_y


def log_human_detection(conf, telem, lat_t, lon_t, img_name=None):
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    print(
        f"[HUMAN DETECTED] {ts} | conf={conf:.2f} | "
        f"drone(lat={telem.get('lat')}, lon={telem.get('lon')}, h={telem.get('h_agl')}m) | "
        f"target(lat={lat_t:.7f}, lon={lon_t:.7f})"
        + (f" | image={img_name}" if img_name else "")
    )


# =========================
# CAPTURE HELPERS (FIXED)
# - Remove quote characters in caps (OpenCV often breaks with them)
# - Try VIC forced (compute-hw=2) first for better headless chance
# =========================
def try_open_gst(pipeline, timeout_s=2.0):
    print(f"[TRY] trying pipeline:\n{pipeline}\n")
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                print("[OK] pipeline opened and returning frames.")
                return cap
        time.sleep(0.05)
    try:
        cap.release()
    except Exception:
        pass
    print("[FAIL] pipeline failed (no frames / not opened).")
    return None


def open_best_capture():
    pipelines = []

    # (A) Argus + force VIC (often works better headless)
    pipelines.append(
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1,format=NV12 ! "
        "nvvidconv compute-hw=2 ! "
        "video/x-raw,format=BGRx ! "
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        "appsink max-buffers=1 drop=true sync=false"
    )

    # (B) Argus normal (works when EGL/GUI is available)
    pipelines.append(
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1,format=NV12 ! "
        "nvvidconv ! "
        "video/x-raw,format=BGRx ! "
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        "appsink max-buffers=1 drop=true sync=false"
    )

    # (C) V4L2 YUY2 fallback (USB cams etc.)
    pipelines.append(
        "v4l2src device=/dev/video0 ! "
        "video/x-raw,format=YUY2,width=1280,height=720,framerate=30/1 ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink max-buffers=1 drop=true sync=false"
    )

    # (D) V4L2 MJPEG fallback (common on USB cams)
    pipelines.append(
        "v4l2src device=/dev/video0 ! "
        "image/jpeg,width=1280,height=720,framerate=30/1 ! "
        "jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
        "appsink max-buffers=1 drop=true sync=false"
    )

    for p in pipelines:
        cap = try_open_gst(p, timeout_s=2.0)
        if cap is not None:
            print("[SELECTED] Using pipeline above.")
            return cap, p

    # Last resort: direct /dev/video0
    try:
        print("[FALLBACK] Trying direct /dev/video0 via OpenCV VideoCapture('/dev/video0').")
        cap = cv2.VideoCapture("/dev/video0")
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                print("[OK] /dev/video0 opened directly.")
                return cap, "v4l2:/dev/video0"
            cap.release()
    except Exception as e:
        print("[ERR] Exception trying direct /dev/video0:", e)

    print("[ERROR] No camera pipelines succeeded.")
    return None, None


def safe_read(cap_obj, max_retries=6, retry_delay=0.1):
    for _ in range(max_retries):
        ret, frame = cap_obj.read()
        if ret and (frame is not None) and frame.size > 0:
            return True, frame
        time.sleep(retry_delay)
    return False, None


# =========================
# OPEN CAMERA
# =========================
cap, used_pipeline = open_best_capture()
if cap is None:
    print("❌ Camera failed to open with all tried pipelines.")
    print("Run gst-launch tests manually to find a working pipeline, then edit open_best_capture().")
    raise SystemExit(1)

print("Camera opened. Using pipeline:", used_pipeline)


# =========================
# DISPLAY WINDOW (optional)
# =========================
if not HEADLESS:
    cv2.namedWindow(display_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(display_window_name, 800, 600)


# =========================
# MAIN LOOP
# =========================
last_saved_time = 0.0
failed_reads = 0

try:
    while True:
        ok, frame = safe_read(cap)
        if not ok:
            failed_reads += 1
            print(f"Frame read failed ({failed_reads}). Attempting to reopen capture...")
            try:
                cap.release()
            except Exception:
                pass
            time.sleep(0.2)
            cap, used_pipeline = open_best_capture()
            if cap is None:
                print("Reopen failed, waiting 1s before retry...")
                time.sleep(1.0)
                continue
            failed_reads = 0
            continue

        failed_reads = 0
        orig_h, orig_w = frame.shape[:2]

        # letterbox to model input (keeps aspect)
        frame_lb, scale, pad_x, pad_y = letterbox(frame, (imgsz, imgsz))

        # BGR -> RGB for model input
        frame_rgb = cv2.cvtColor(frame_lb, cv2.COLOR_BGR2RGB)

        # run inference
        results = model(frame_rgb, imgsz=imgsz, conf=CONF_THRESH, verbose=False)

        # snapshot telemetry
        with telemetry_lock:
            telem_snapshot = telemetry.copy()

        # print detections count (debug)
        num_boxes = 0
        if results and len(results) > 0 and results[0].boxes is not None:
            num_boxes = len(results[0].boxes)
        print("Detections:", num_boxes)

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())

                # only keep 'person' class (assumed class 0)
                if cls_id != 0:
                    continue

                # center in letterbox coords
                u_res = 0.5 * (x1 + x2)
                v_res = 0.5 * (y1 + y2)

                # map center from letterbox -> original coords
                u_lb = u_res - pad_x
                v_lb = v_res - pad_y
                u_orig = u_lb / scale
                v_orig = v_lb / scale

                # geolocate
                geo = geolocate_target_from_pixel(u_orig, v_orig, K, R_cam_to_body, telem_snapshot)
                if geo is None:
                    print("[DEBUG] Geo returned None | telemetry =", telem_snapshot)
                    continue

                lat_t, lon_t = geo

                # IMPORTANT FIX: use current h from telemetry snapshot (not stale global h)
                send_human(
                    lat=lat_t,
                    lon=lon_t,
                    alt=telem_snapshot.get("h_agl"),
                    conf=conf
                )

                # optionally save crop (rate-limited)
                now = time.time()
                img_name = None
                if now - last_saved_time > save_cooldown_s and conf > 0.6:
                    x1o = int((x1 - pad_x) / scale)
                    y1o = int((y1 - pad_y) / scale)
                    x2o = int((x2 - pad_x) / scale)
                    y2o = int((y2 - pad_y) / scale)

                    # clamp
                    x1o, y1o = max(0, x1o), max(0, y1o)
                    x2o, y2o = min(orig_w - 1, x2o), min(orig_h - 1, y2o)

                    crop = frame[y1o:y2o, x1o:x2o]
                    if crop.size > 0:
                        img_name = os.path.join(save_crops_dir, f"human_{int(now)}.jpg")
                        cv2.imwrite(img_name, crop)
                        last_saved_time = now

                # log + CSV
                log_human_detection(conf, telem_snapshot, lat_t, lon_t, img_name)
                timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
                with open(log_csv, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        timestamp,
                        telem_snapshot.get("lat"),
                        telem_snapshot.get("lon"),
                        telem_snapshot.get("h_agl"),
                        conf, cls_id, lat_t, lon_t, img_name
                    ])

                # draw boxes (only if you display)
                if not HEADLESS:
                    x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
                    label = f"HUMAN {conf:.2f}"
                    cv2.rectangle(frame_lb, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
                    cv2.putText(frame_lb, label, (x1i, max(12, y1i - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.circle(frame_lb, (int(u_res), int(v_res)), 4, (0, 255, 0), -1)

        # display (optional)
        if not HEADLESS:
            cv2.imshow(display_window_name, frame_lb)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

except KeyboardInterrupt:
    print("Interrupted by user, exiting...")

finally:
    try:
        if cap is not None:
            cap.release()
    except Exception:
        pass
    if not HEADLESS:
        cv2.destroyAllWindows()
