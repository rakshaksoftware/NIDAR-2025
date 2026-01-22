import cv2
import numpy as np
import time
import csv
import os
from ultralytics import YOLO
from geo_projection import geolocate_target_from_pixel
from telemetry import telemetry, telemetry_lock
from math import sqrt

# ---------------- CONFIG ----------------
CONF_THRESH = 0.6
EMIT_INTERVAL = 0.5  # seconds
last_emit_time = 0
CSV_FILE = "detections_raw.csv"
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "lat", "lon", "confidence"])

# Camera intrinsics (replace with calibrated values)
fx, fy, cx, cy = 1496.0, 1494.0, 328.0, 697.0
K = (fx, fy, cx, cy)

# Camera to body rotation (verify mounting!)
R_cam_to_body = np.array([
    [0.0,         -1.0,  0.0],
    [1.0,          0.0,  0.0],
    [0.0,          0.0,  1.0]
], dtype=float)

# ---------------- MODEL ----------------
model = YOLO("yolo11n.engine")   # TensorRT engine

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO (TensorRT)
    results = model(frame, verbose=False)
    boxes = results[0].boxes

    # Snapshot telemetry safely
    with telemetry_lock:
        telem_snapshot = telemetry.copy()

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())

            # filter non-person
            if cls_id != 0:
                continue

            # confidence threshold
            if conf < CONF_THRESH:
                continue

            # bbox center
            u = 0.5 * (x1 + x2)
            v = 0.5 * (y1 + y2)

            geo = geolocate_target_from_pixel(
                u, v, K, R_cam_to_body, telem_snapshot
            )

            if geo is None:
                continue

            lat_t, lon_t = geo

            now = time.time()
            if now - last_emit_time < EMIT_INTERVAL:
                continue
            last_emit_time = now
            with open(CSV_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([time.time(), lat_t, lon_t, conf])

            print(f"[HUMAN] lat={lat_t:.7f}, lon={lon_t:.7f}, conf={conf:.2f}")

            # Visualization
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 255, 0), 2)
            cv2.circle(frame, (int(u), int(v)), 5, (0, 255, 0), -1)
            cv2.putText(frame,
                        f"{lat_t:.5f},{lon_t:.5f}",
                        (int(u), int(v) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 0),
                        1)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
