from pymavlink import mavutil
import threading
import time
import numpy as np

telemetry = {
    "lat": None,
    "lon": None,
    "h_agl": None,
    "roll": 0.0,
    "pitch": 0.0,
    "yaw": 0.0,
}
telemetry_lock = threading.Lock()

def telemetry_thread():
    print("[TELEM] Connecting to MAVLink router UDP 14550")
    master = mavutil.mavlink_connection("udpin:0.0.0.0:14550")
# or: "udpin:127.0.0.1:14550"

    print("[TELEM] Waiting for heartbeat...")
    master.wait_heartbeat()
    print("[TELEM] Heartbeat OK")

    while True:
        msg = master.recv_match(blocking=True)
        if msg is None:
            continue

        with telemetry_lock:
            if msg.get_type() == "GLOBAL_POSITION_INT":
                telemetry["lat"] = msg.lat / 1e7
                telemetry["lon"] = msg.lon / 1e7
                telemetry["h_agl"] = msg.relative_alt / 1000.0

            elif msg.get_type() == "ATTITUDE":
                telemetry["roll"]  = msg.roll
                telemetry["pitch"] = msg.pitch
                telemetry["yaw"]   = msg.yaw

        time.sleep(0.001)

# start thread
t = threading.Thread(target=telemetry_thread, daemon=True)
t.start()
