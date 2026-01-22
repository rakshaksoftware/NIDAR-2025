import time
import cv2
import numpy as np
from dronekit import VehicleMode
from pymavlink import mavutil
from ultralytics import YOLO

# ---------------- CONFIG ----------------
SHOW_OVERLAY = True

CONF_THRESH = 0.6

CENTER_TOL_PX = 20          # pixels
HOLD_TIME = 2.0             # seconds target must stay centered

KP = 0.003                  # proportional gain (tune carefully)
MAX_VEL = 0.5               # m/s (clamp for safety)

CAMERA_INDEX = 0


# ---------------- MAVLINK VELOCITY ----------------
def send_body_velocity(vehicle, vx, vy, vz):
    """
    Send velocity command in BODY_NED frame.
    vx: forward (+)
    vy: right (+)
    vz: down (+)
    """
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0, 0, 0,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        0b0000111111000111,  # enable velocity only
        0, 0, 0,
        vx, vy, vz,
        0, 0, 0,
        0, 0
    )
    vehicle.send_mavlink(msg)
    vehicle.flush()


# ---------------- VISUAL SERVO CORE ----------------
def visual_servo(vehicle, debug_view=False):
    """
    Runs visual servo loop.
    Returns True when target is centered stably.
    """

    print("[SERVO] Loading YOLO engine...")
    model = YOLO("best.engine")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[SERVO] Camera open failed")
        return False

    centered_since = None

    print("[SERVO] Starting visual servo loop")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        if SHOW_OVERLAY:
            alt = vehicle.location.global_relative_frame.alt

            cv2.putText(
                frame,
                f"ALT: {alt:.2f} m",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"ex={ex} ey={ey}",
                (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )


        h, w, _ = frame.shape
        cx = w // 2
        cy = h // 2

        results = model(frame, verbose=False)
        boxes = results[0].boxes

        target_found = False

        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())

                if cls_id != 0 or conf < CONF_THRESH:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                u = int((x1 + x2) / 2)
                v = int((y1 + y2) / 2)

                ex = u - cx
                ey = v - cy

                # Proportional control (BODY frame)
                vx = -KP * ey    # forward/back
                vy =  KP * ex    # right/left
                vz = 0.0

                # Clamp velocities
                vx = np.clip(vx, -MAX_VEL, MAX_VEL)
                vy = np.clip(vy, -MAX_VEL, MAX_VEL)

                send_body_velocity(vehicle, vx, vy, vz)

                target_found = True

                # Check centering
                if abs(ex) < CENTER_TOL_PX and abs(ey) < CENTER_TOL_PX:
                    if centered_since is None:
                        centered_since = time.time()
                    elif time.time() - centered_since >= HOLD_TIME:
                        print("[SERVO] Target centered")
                        cap.release()
                        cv2.destroyAllWindows()
                        return True
                else:
                    centered_since = None

                if debug_view:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                  (0, 255, 0), 2)
                    cv2.circle(frame, (u, v), 5, (0, 0, 255), -1)
                    cv2.line(frame, (cx, cy), (u, v), (255, 0, 0), 2)
                    cv2.putText(frame, f"ex={ex}, ey={ey}",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 0), 2)

                break  # only track ONE person

        if not target_found:
            # stop motion if no target
            send_body_velocity(vehicle, 0, 0, 0)
            centered_since = None

        if debug_view:
            cv2.imshow("visual_servo", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    return False
