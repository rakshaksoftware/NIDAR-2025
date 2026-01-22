# main_video.py - run detection on a video file
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
VIDEO_PATH = "videofeed.mp4"   # <<< YOUR VIDEO HERE
ENGINE_PATH = "best.engine"
imgsz = 640
display_window_name = "Disaster Drone – Human Detection (VIDEO)"
log_csv = "detections_log.csv"
save_crops_dir = "human_crops"
os.makedirs(save_crops_dir, exist_ok=True)

# Camera intrinsics (replace after calibration)
fx, fy, cx, cy = 666.65327, 891.26635, 339.31718, 252.0422
K = (fx, fy, cx, cy)

R_cam_to_body = np.array([
    [0.0, -1.0, 0.0],
    [1.0,  0.0, 0.0],
    [0.0,  0.0, 1.0]
], dtype=float)

CONF_THRESH = 0.5
save_cooldown_s = 5.0

# ---------------- MODEL ----------------
print(f"Loading model engine: {ENGINE_PATH}")
model = YOLO(ENGINE_PATH)
print("Model loaded.")

# ---------------- CSV HEADER ----------------
try:
    with open(log_csv, "x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp_utc",
            "lat_drone",
            "lon_drone",
            "h_agl",
            "det_conf",
            "det_class",
            "target_lat",
            "target_lon",
            "image"
        ])
except FileExistsError:
    pass

# ---------------- HELPERS ----------------
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
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

# ---------------- OPEN VIDEO ----------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"❌ Failed to open video: {VIDEO_PATH}")
    exit(1)

print("Video opened:", VIDEO_PATH)

# ---------------- DISPLAY ----------------
cv2.namedWindow(display_window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(display_window_name, 800, 600)

last_saved_time = 0.0

# ---------------- MAIN LOOP ----------------
try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("End of video reached.")
            break

        orig_h, orig_w = frame.shape[:2]

        frame_lb, scale, pad_x, pad_y = letterbox(frame, (imgsz, imgsz))
        frame_rgb = cv2.cvtColor(frame_lb, cv2.COLOR_BGR2RGB)

        results = model(frame_rgb, imgsz=imgsz, conf=CONF_THRESH, verbose=False)

        with telemetry_lock:
            telem_snapshot = telemetry.copy()

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if cls_id != 0:
                    continue

                u_res = 0.5 * (x1 + x2)
                v_res = 0.5 * (y1 + y2)

                u_orig = (u_res - pad_x) / scale
                v_orig = (v_res - pad_y) / scale

                geo = geolocate_target_from_pixel(
                    u_orig, v_orig, K, R_cam_to_body, telem_snapshot
                )
                if geo is None:
                    continue

                lat_t, lon_t = geo

                now = time.time()
                img_name = None
                if now - last_saved_time > save_cooldown_s and conf > 0.6:
                    x1o = int((x1 - pad_x) / scale)
                    y1o = int((y1 - pad_y) / scale)
                    x2o = int((x2 - pad_x) / scale)
                    y2o = int((y2 - pad_y) / scale)

                    x1o, y1o = max(0, x1o), max(0, y1o)
                    x2o, y2o = min(orig_w - 1, x2o), min(orig_h - 1, y2o)

                    crop = frame[y1o:y2o, x1o:x2o]
                    if crop.size > 0:
                        img_name = os.path.join(save_crops_dir, f"human_{int(now)}.jpg")
                        cv2.imwrite(img_name, crop)
                        last_saved_time = now

                log_human_detection(conf, telem_snapshot, lat_t, lon_t, img_name)

                with open(log_csv, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                        telem_snapshot.get("lat"),
                        telem_snapshot.get("lon"),
                        telem_snapshot.get("h_agl"),
                        conf,
                        cls_id,
                        lat_t,
                        lon_t,
                        img_name
                    ])

                x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
                cv2.rectangle(frame_lb, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
                cv2.circle(frame_lb, (int(u_res), int(v_res)), 4, (0, 255, 0), -1)

        cv2.imshow(display_window_name, frame_lb)
        if cv2.waitKey(1) & 0xFF == 27:
            break

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()
                       
