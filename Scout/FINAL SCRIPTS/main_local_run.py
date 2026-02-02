# main.py - cleaned & properly indented version
import cv2
import time
import csv
import os
import numpy as np
from datetime import datetime
from ultralytics import YOLO

from geo_projection import geolocate_target_from_pixel
from telemetry import telemetry, telemetry_lock

# ---------------- USER SETTINGS ----------------
ENGINE_PATH = "yolo11n.engine" #path to your TensorRT engine
imgsz = 640                   # inference size (engine must be compatible)
display_window_name = "Disaster Drone – Human Detection"
log_csv = "detections_log.csv"
save_crops_dir = "human_crops"
os.makedirs(save_crops_dir, exist_ok=True)

# Camera intrinsics placeholders (replace after calibration)
fx, fy, cx, cy = 666.65327, 891.26635, 339.31718, 252.0422
K = (fx, fy, cx, cy)
R_cam_to_body = np.array([
    [-1.0,         0.0,  0.0],
    [0.0,         -1.0,  0.0 ],
    [0.0,          0.0,  1.0]
], dtype=float)

# ---------------- model ----------------
print(f"Loading model engine: {ENGINE_PATH}")
model = YOLO(ENGINE_PATH)
print("Model loaded.")

# ---------------- CSV header ----------------
try:
    with open(log_csv, "x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_utc", "lat_drone", "lon_drone", "h_agl",
                         "det_conf", "det_class", "target_lat", "target_lon", "image"])
except FileExistsError:
    pass

# ---------------- helper functions ----------------
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Resize and pad image to fit new_shape while keeping aspect ratio.
    Returns: canvas, scale, pad_x, pad_y
    pad_x/pad_y are the left/top padding (in pixels) in the letterboxed image.
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

# ---------------- capture helpers (robust) ----------------
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

    # 1) nvarguscamerasrc 1920x1080 -> convert to BGR 1280x720
    pipelines.append(
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1, format=NV12 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! "
    "appsink drop=true sync=false"
    )

    # 2) nvarguscamerasrc 1280x720
    pipelines.append(
        "nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1,format=NV12' ! "
        "nvvidconv ! 'video/x-raw, format=(string)BGRx, width=(int)1280, height=(int)720' ! "
        "videoconvert ! 'video/x-raw, format=(string)BGR' ! appsink max-buffers=1 drop=true sync=false"
    )

    # 3) v4l2 raw bayer path (IMX477)
    pipelines.append(
        "v4l2src device=/dev/video0 ! 'video/x-bayer,format=RG10,width=1920,height=1080,framerate=30/1' ! "
        "bayer2rgb ! videoconvert ! 'video/x-raw, format=(string)BGR' ! appsink max-buffers=1 drop=true sync=false"
    )

    # 4) v4l2 YUYV fallback
    pipelines.append(
        "v4l2src device=/dev/video0 ! 'video/x-raw, format=(string)YUY2, width=(int)1280, height=(int)720, framerate=(fraction)30/1' ! "
        "videoconvert ! 'video/x-raw, format=(string)BGR' ! appsink max-buffers=1 drop=true sync=false"
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
    for i in range(max_retries):
        ret, frame = cap_obj.read()
        if ret and (frame is not None) and frame.size > 0:
            return True, frame
        time.sleep(retry_delay)
    return False, None

# ---------------- open camera ----------------
cap, used_pipeline = open_best_capture()
if cap is None:
    print("❌ Camera failed to open with all tried pipelines.")
    print("Please run gst-launch tests manually to find a working pipeline, then edit open_best_capture().")
    exit(1)

print("Camera opened. Using pipeline:", used_pipeline)

# ---------------- display window ----------------
cv2.namedWindow(display_window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(display_window_name, 800, 600)

# ---------------- detection helpers ----------------
last_saved_time = 0.0
save_cooldown_s = 5.0  # save at most one crop every N seconds
CONF_THRESH = 0.5

# ---------------- main loop ----------------
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

        # letterbox to target input size (keeps aspect)
        frame_lb, scale, pad_x, pad_y = letterbox(frame, (imgsz, imgsz))

        # convert BGR->RGB for model input
        frame_rgb = cv2.cvtColor(frame_lb, cv2.COLOR_BGR2RGB)

        # run inference - set a reasonable conf threshold for live feed
        results = model(frame_rgb, imgsz=imgsz, conf=CONF_THRESH, verbose=False)

        # debug: print number of detections
        num_boxes = 0
        if results and len(results) > 0 and results[0].boxes is not None:
            num_boxes = len(results[0].boxes)
        print("Detections:", num_boxes)

        # snapshot telemetry
        with telemetry_lock:
            telem_snapshot = telemetry.copy()

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                # extract coordinates and scores (in letterbox space)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())

                # only keep 'person' class (assumed class 0)
                if cls_id != 0:
                    continue

                # center in letterbox coords
                u_res = 0.5 * (x1 + x2)
                v_res = 0.5 * (y1 + y2)

                # Map center from letterbox -> original image coordinates
                # Remove padding and undo scale
                u_lb = u_res - pad_x
                v_lb = v_res - pad_y
                u_orig = u_lb / scale
                v_orig = v_lb / scale

                # Geolocate using your helper (expects original pixel coords)
                geo = geolocate_target_from_pixel(u_orig, v_orig, K, R_cam_to_body, telem_snapshot)
                if geo is None:
                    print("[DEBUG] Geo returned None | telemetry =", telem_snapshot)
                    continue

                lat_t, lon_t = geo

                # optionally save crop (rate-limited) - crop from ORIGINAL frame
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

                # terminal log & CSV
                log_human_detection(conf, telem_snapshot, lat_t, lon_t, img_name)
                timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
                with open(log_csv, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp,
                                     telem_snapshot.get("lat"),
                                     telem_snapshot.get("lon"),
                                     telem_snapshot.get("h_agl"),
                                     conf, cls_id, lat_t, lon_t, img_name])

                # draw boxes on letterboxed frame for display
                x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
                label = f"HUMAN {conf:.2f}"
                cv2.rectangle(frame_lb, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
                cv2.putText(frame_lb, label, (x1i, max(12, y1i - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame_lb, (int(u_res), int(v_res)), 4, (0, 255, 0), -1)

        # display - show letterboxed image (no distortion)
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
    cv2.destroyAllWindows()
