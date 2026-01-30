# centering.py
#!/usr/bin/env python3
import time
import numpy as np
import cv2
from dataclasses import dataclass

# NOTE: the centering routine calls MAVLink body-velocity messages through master.
# This function uses the same MAV_FRAME_BODY_NED approach as in your earlier centering test.
def send_body_velocity(master, vx, vy, vz):
    """
    Sends velocity commands in BODY frame.
    vx = Forward/Back, vy = Right/Left, vz = Down/Up
    """
    try:
        master.mav.set_position_target_local_ned_send(
            0, master.target_system, master.target_component,
            8,  # MAV_FRAME_BODY_NED (int 8) - kept simple to avoid importing mavutil constant here
            0b0000111111000111,  # mask: ignore positions, use velocities
            0, 0, 0,            # pos ignored
            float(vx), float(vy), float(vz),         # velocities
            0, 0, 0,            # accel ignored
            0, 0                # yaw ignored
        )
    except Exception as e:
        # If low-level send fails, just print and continue
        print("[CENTERING] send_body_velocity error:", e)

@dataclass
@dataclass
class CenteringParams:
    # Detection
    CONF_THRESH: float = 0.5          # higher confidence for heavy drone
    CONF_HYST: float = 0.55           # hysteresis to avoid flicker

    # Pixel alignment
    CENTER_TOL_PX: int = 35            # larger tolerance (heavy inertia)
    HOLD_TIME: float = 1.5             # must hold longer

    # Timing
    VISUAL_TIMEOUT: float = 35.0       # allow more time

    # Control gains (VERY IMPORTANT)
    KP: float = 0.0011                 # ↓ from 0.002
    MAX_VEL: float = 0.35              # ↓ from 0.5

    # Stability helpers
    DEADZONE_PX: int = 10              # ignore tiny noise
    VEL_SLEW: float = 0.15             # m/s per cycle max change

    CAMERA_INDEX: int = 0


def run_centering_routine(master, model, cap, params: CenteringParams):
    """
    Runs centering visual-servoing until target held in center for HOLD_TIME or until timeout.
    - master: mavlink connection object (must provide master.mav/... and target system/component)
    - model: ultralytics.YOLO model object (already loaded)
    - cap: cv2.VideoCapture object (already opened)
    - params: CenteringParams
    Returns True if centered & locked; False on timeout or failure.
    """
    print("[CENTERING] Visual Centering Started...")
    start_time = time.time()
    centered_start = None

    while time.time() - start_time < params.VISUAL_TIMEOUT:
        ret, frame = cap.read()
        if not ret:
            # small sleep so we don't busy-loop on camera failure
            time.sleep(0.05)
            continue

        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2

        # 1. Inference
        try:
            results = model(frame, verbose=False)
            boxes = results[0].boxes
        except Exception as e:
            print("[CENTERING] model inference error:", e)
            boxes = []

        target_found = False
        err_x = 0.0
        err_y = 0.0

        # 2. Find best target (highest confidence)
        if boxes:
            try:
                best_box = max(boxes, key=lambda b: float(b.conf[0]))
                conf = float(best_box.conf[0])
                if conf > params.CONF_THRESH:
                    target_found = True
                    x1, y1, x2, y2 = [float(v) for v in best_box.xyxy[0]]
                    ux, uy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

                    # pixel error (positive = target to the right/down)
                    err_x = -float(ux - cx)
                    err_y = -float(uy - cy)

                    # Map pixel error to body-frame velocities (tweak KP sign as needed for your camera orientation)
                    # Comments: In many cam-to-body setups:
                    # - if target is high in image (err_y negative), you need to move forward (vx positive).
                    # - if target is right in image (err_x positive), you need to move right (vy positive).
                    vx = np.clip(-params.KP * err_y, -params.MAX_VEL, params.MAX_VEL)
                    vy = np.clip( params.KP * err_x, -params.MAX_VEL, params.MAX_VEL)

                    send_body_velocity(master, vx, vy, 0.0)

                    # Check alignment in pixels
                    if abs(err_x) < params.CENTER_TOL_PX and abs(err_y) < params.CENTER_TOL_PX:
                        if centered_start is None:
                            centered_start = time.time()
                        elif time.time() - centered_start > params.HOLD_TIME:
                            print("[CENTERING] Target locked.")
                            # stop motion
                            send_body_velocity(master, 0, 0, 0)
                            return True
                    else:
                        centered_start = None
                else:
                    # Found box but below confidence threshold
                    target_found = False
            except Exception as e:
                print("[CENTERING] box processing error:", e)
                target_found = False

        # 3. If no target found, stop motion (hover)
        if not target_found:
            send_body_velocity(master, 0, 0, 0)
            centered_start = None

        # small loop delay
        time.sleep(0.02)

    print("[CENTERING] Visual Timeout: Failed to lock onto target.")
    # ensure stopped
    send_body_velocity(master, 0, 0, 0)
    return False
