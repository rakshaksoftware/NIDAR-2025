import cv2, time, threading, queue
import numpy as np
from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil
import pycuda.driver as cuda
import geo_coord as gc

from bullseye_det_eff import TRTEngineWrapper, preprocess, postprocess, draw_boxes

# === DroneKit setup ===
vehicle = connect("/dev/ttyUSB0", baud=115200, wait_ready=False)
print("[INFO] Connected to vehicle")

vehicle.mode = VehicleMode("GUIDED")
vehicle.armed = True
while not vehicle.armed:
    time.sleep(0.1)

takeoff_alt = 10.0
vehicle.simple_takeoff(takeoff_alt)
while True:
    alt = vehicle.location.global_relative_frame.alt
    print(f"[INFO] Altitude: {alt:.1f}")
    if alt >= takeoff_alt * 0.95:
        break
    time.sleep(0.2)

# === Velocity Control ===
def send_velocity(vx, vy, vz=0, yaw_rate=0):
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0, 0, 0,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        0b0000111111000111,
        0, 0, 0,
        vx, vy, vz,
        0, 0, 0,
        0, yaw_rate
    )
    vehicle.send_mavlink(msg)
    vehicle.flush()

# # Rectangle coordinates (lat, lon) - you can change them
# p1 = (35.06012, -118.158)
# p2 = (35.06012, -118.16)
# p3 = (35.05935, -118.16)
# p4 = (35.05935, -118.158)

# SITL coordinates dont use this in real testing

p1 = (-35.362877, 149.1662367)
p2 = (-35.362877, 149.1642367)
p3 = (-35.363647, 149.1642367)
p4 = (-35.363647, 149.1662367)

altitude = H = 30


coords = gc.interpolate_rectangle(p1,p2,p3,p4,H)

waypoints = [
    LocationGlobalRelative(lat, lon, takeoff_alt)
    for lat, lon in coords
]
current_tile_idx = 0

def get_distance_meters(loc1, loc2):
    dlat = loc2.lat - loc1.lat
    dlon = loc2.lon - loc1.lon
    return ((dlat*2 + dlon*2) ** 0.5) * 1.113195e5

def go_to_next_waypoint():
    global current_tile_idx
    if current_tile_idx >= len(waypoints):
        print("[INFO] All waypoints visited. Ending mission.")
        return False

    target = waypoints[current_tile_idx]
    current_tile_idx += 1
    print(f"[INFO] Navigating to waypoint {current_tile_idx}: {target}")
    vehicle.simple_goto(target)

    while True:
        curr = vehicle.location.global_relative_frame
        dist = get_distance_meters(curr, target)
        print(f"[INFO] Distance to waypoint: {dist:.2f}m")
        if dist < 1.5:
            print("[INFO] Tile center reached.")
            return True
        time.sleep(1)

# === Global constants and queues ===
INPUT_W, INPUT_H = 640, 640
Kx, Ky = 0.5, 0.5
frame_queue = queue.Queue(maxsize=3)
result_queue = queue.Queue(maxsize=3)
stop_event = threading.Event()
stop_camera = threading.Event()

# === Thread: Camera ===
def camera_thread():
    cap = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=3280, height=2464, "
                           "format=NV12, framerate=20/1 ! nvvidconv flip-method=0 ! video/x-raw, "
                           "width=640, height=640, format=BGRx ! videoconvert ! appsink",
                           cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError("[ERROR] Camera failed to open.")

    print("[INFO] Camera thread started")
    while not stop_camera.is_set():
        ret, frame = cap.read()
        if not ret:
            continue
        if not frame_queue.full():
            frame_queue.put(frame)

    cap.release()
    print("[INFO] Camera thread exited")

# === Thread: Inference ===
def processing_thread():
    print("[INFO] Initializing inference engine")
    device = cuda.Device(0)
    context = device.make_context()
    engine = TRTEngineWrapper("bbpreyash.trt", context)

    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=1)
        except queue.Empty:
            continue

        input_tensor = preprocess(frame)
        try:
            out = engine.infer(input_tensor)
            boxes, scores = postprocess(out)
            result_queue.put((frame, boxes, scores))
        except Exception as e:
            print(f"[ERROR] Inference failed: {e}")

    context.pop()
    print("[INFO] Inference thread exited")

# === Thread: Tracking & Velocity ===
def tracking_thread():
    cont_frames = 0
    missed_frames = 0
    MAX_MISSED_FRAMES = 40

    while not stop_event.is_set():
        try:
            frame, boxes, scores = result_queue.get(timeout=1)
        except queue.Empty:
            missed_frames += 1
            if missed_frames >= MAX_MISSED_FRAMES:
                print("[INFO] Object lost for too long. Moving to next waypoint.")
                stop_camera.set()
                stop_event.set()
                return
            continue

        if boxes.shape[0]:
            missed_frames = 0
            box = boxes[0]
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            ex = (INPUT_W / 2 - cx) / (INPUT_W / 2)
            ey = (INPUT_H / 2 - cy) / (INPUT_H / 2)
            vx = - Kx * ex
            vy = - Ky * ey

            if vx < 0.05 and vy < 0.05:
                cont_frames += 1
                if cont_frames >= 4:
                    call_drop()
                    stop_event.set()
                    break
            else:
                cont_frames = 0

            send_velocity(vx, vy)
        else:
            send_velocity(0, 0)

        out = draw_boxes(frame, boxes, scores)
        cv2.circle(out, (INPUT_W//2, INPUT_H//2), 5, (255, 0, 0), -1)
        cv2.imshow("Tracking", out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    cv2.destroyAllWindows()
    print("[INFO] Tracking thread exited")

def call_drop():
    print("[INFO] Target locked. Descending for payload drop.")
    vehicle.simple_goto(LocationGlobalRelative(
        vehicle.location.global_relative_frame.lat,
        vehicle.location.global_relative_frame.lon,
        5
    ))
    while True:
        alt = vehicle.location.global_relative_frame.alt
        print(f"[INFO] Descending... Altitude: {alt:.1f}")
        if alt <= 5.5:
            SERVO_CHANNEL = 7      # SERVO6 output (e.g., AUX2 on Pixhawk)
            PWM_VALUE = 1700       # In microseconds (valid range: 1000–2000)

            # Create the MAVLink command for DO_SET_SERVO
            msg = vehicle.message_factory.command_long_encode(
                0, 0,                                      # target_system, target_component (0 = default)
                mavutil.mavlink.MAV_CMD_DO_SET_SERVO,     # command
                0,                                         # confirmation
                SERVO_CHANNEL,                             # param1: servo number
                PWM_VALUE,                                 # param2: PWM value
                0, 0, 0, 0, 0                              # param3-7: not used
            )
            
            # Send command to vehicle

            vehicle.send_mavlink(msg)
            vehicle.flush()

            print(f"Sent {PWM_VALUE}µs PWM to SERVO{SERVO_CHANNEL}")



            break
        time.sleep(0.5)
    print("[INFO] Payload dropped.")

# === Detection Wrapper ===
def detect_center():
    print("[INFO] Starting threads...")
    threads = [
        threading.Thread(target=camera_thread),
        threading.Thread(target=processing_thread),
        threading.Thread(target=tracking_thread)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

# === Tile Loop ===
while current_tile_idx < len(waypoints):
    if not go_to_next_waypoint():
        break
    detect_center()
    if stop_event.is_set():
        break

# === Cleanup ===
send_velocity(0, 0, 0)
vehicle.mode = VehicleMode("LOITER")
vehicle.close()
print("[INFO] Vehicle disarmed and closed.")

takeoff_alt = 10.0
vehicle.simple_takeoff(takeoff_alt)
while True:
    alt = vehicle.location.global_relative_frame.alt
    print(f"[INFO] Altitude: {alt:.1f}")
    if alt >= takeoff_alt * 0.95:
        break
    time.sleep(0.2)

# === Velocity Control ===
def send_velocity(vx, vy, vz=0, yaw_rate=0):
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0, 0, 0,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        0b0000111111000111,
        0, 0, 0,
        vx, vy, vz,
        0, 0, 0,
        0, yaw_rate
    )
    vehicle.send_mavlink(msg)
    vehicle.flush()

# # Rectangle coordinates (lat, lon) - you can change them
p1 = (35.06012, -118.158)
p2 = (35.06012, -118.16)
p3 = (35.05935, -118.16)
p4 = (35.05935, -118.158)

# SITL coordinates dont use this in real testing

#p1 = (-35.362877, 149.1662367)
#p2 = (-35.362877, 149.1642367)
#p3 = (-35.363647, 149.1642367)
#p4 = (-35.363647, 149.1662367)

altitude = H = 30


coords = gc.interpolate_rectangle(p1,p2,p3,p4,H)

waypoints = [
    LocationGlobalRelative(lat, lon, takeoff_alt)
    for lat, lon in coords
]
current_tile_idx = 0

def get_distance_meters(loc1, loc2):
    dlat = loc2.lat - loc1.lat
    dlon = loc2.lon - loc1.lon
    return ((dlat*2 + dlon*2) ** 0.5) * 1.113195e5

def go_to_next_waypoint():
    global current_tile_idx
    if current_tile_idx >= len(waypoints):
        print("[INFO] All waypoints visited. Ending mission.")
        return False

    target = waypoints[current_tile_idx]
    current_tile_idx += 1
    print(f"[INFO] Navigating to waypoint {current_tile_idx}: {target}")
    vehicle.simple_goto(target)

    while True:
        curr = vehicle.location.global_relative_frame
        dist = get_distance_meters(curr, target)
        print(f"[INFO] Distance to waypoint: {dist:.2f}m")
        if dist < 1.5:
            print("[INFO] Tile center reached.")
            return True
        time.sleep(1)

# === Global constants and queues ===
INPUT_W, INPUT_H = 640, 640
Kx, Ky = 0.5, 0.5
frame_queue = queue.Queue(maxsize=3)
result_queue = queue.Queue(maxsize=3)
stop_event = threading.Event()
stop_camera = threading.Event()

# === Thread: Camera ===
def camera_thread():
    cap = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=3280, height=2464, "
                           "format=NV12, framerate=20/1 ! nvvidconv flip-method=0 ! video/x-raw, "
                           "width=640, height=640, format=BGRx ! videoconvert ! appsink",
                           cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError("[ERROR] Camera failed to open.")

    print("[INFO] Camera thread started")
    while not stop_camera.is_set():
        ret, frame = cap.read()
        if not ret:
            continue
        if not frame_queue.full():
            frame_queue.put(frame)

    cap.release()
    print("[INFO] Camera thread exited")

# === Thread: Inference ===
def processing_thread():
    print("[INFO] Initializing inference engine")
    device = cuda.Device(0)
    context = device.make_context()
    engine = TRTEngineWrapper("bbpreyash.trt", context)

    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=1)
        except queue.Empty:
            continue

        input_tensor = preprocess(frame)
        try:
            out = engine.infer(input_tensor)
            boxes, scores = postprocess(out)
            result_queue.put((frame, boxes, scores))
        except Exception as e:
            print(f"[ERROR] Inference failed: {e}")

    context.pop()
    print("[INFO] Inference thread exited")

# === Thread: Tracking & Velocity ===
def tracking_thread():
    cont_frames = 0
    missed_frames = 0
    MAX_MISSED_FRAMES = 40

    while not stop_event.is_set():
        try:
            frame, boxes, scores = result_queue.get(timeout=1)
        except queue.Empty:
            missed_frames += 1
            if missed_frames >= MAX_MISSED_FRAMES:
                print("[INFO] Object lost for too long. Moving to next waypoint.")
                stop_camera.set()
                stop_event.set()
                return
            continue

        if boxes.shape[0]:
            missed_frames = 0
            box = boxes[0]
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            ex = (INPUT_W / 2 - cx) / (INPUT_W / 2)
            ey = (INPUT_H / 2 - cy) / (INPUT_H / 2)
            vx = - Kx * ex
            vy = - Ky * ey

            if vx < 0.05 and vy < 0.05:
                cont_frames += 1
                if cont_frames >= 4:
                    call_drop()
                    stop_event.set()
                    break
            else:
                cont_frames = 0

            send_velocity(vx, vy)
        else:
            send_velocity(0, 0)

        out = draw_boxes(frame, boxes, scores)
        cv2.circle(out, (INPUT_W//2, INPUT_H//2), 5, (255, 0, 0), -1)
        cv2.imshow("Tracking", out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    cv2.destroyAllWindows()
    print("[INFO] Tracking thread exited")

def call_drop():
    print("[INFO] Target locked. Descending for payload drop.")
    vehicle.simple_goto(LocationGlobalRelative(
        vehicle.location.global_relative_frame.lat,
        vehicle.location.global_relative_frame.lon,
        5
    ))
    while True:
        alt = vehicle.location.global_relative_frame.alt
        print(f"[INFO] Descending... Altitude: {alt:.1f}")
        if alt <= 5.5:
            SERVO_CHANNEL = 7      # SERVO6 output (e.g., AUX2 on Pixhawk)
            PWM_VALUE = 1700       # In microseconds (valid range: 1000–2000)

            # Create the MAVLink command for DO_SET_SERVO
            msg = vehicle.message_factory.command_long_encode(
                0, 0,                                      # target_system, target_component (0 = default)
                mavutil.mavlink.MAV_CMD_DO_SET_SERVO,     # command
                0,                                         # confirmation
                SERVO_CHANNEL,                             # param1: servo number
                PWM_VALUE,                                 # param2: PWM value
                0, 0, 0, 0, 0                              # param3-7: not used
            )
            
            # Send command to vehicle

            vehicle.send_mavlink(msg)
            vehicle.flush()

            print(f"Sent {PWM_VALUE}µs PWM to SERVO{SERVO_CHANNEL}")



            break
        time.sleep(0.5)
    print("[INFO] Payload dropped.")

# === Detection Wrapper ===
def detect_center():
    print("[INFO] Starting threads...")
    threads = [
        threading.Thread(target=camera_thread),
        threading.Thread(target=processing_thread),
        threading.Thread(target=tracking_thread)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

# === Tile Loop ===
while current_tile_idx < len(waypoints):
    if not go_to_next_waypoint():
        break
    detect_center()
    if stop_event.is_set():
        break

# === Cleanup ===
send_velocity(0, 0, 0)
vehicle.mode = VehicleMode("LOITER")
vehicle.close()
print("[INFO] Vehicle disarmed and closed.")
