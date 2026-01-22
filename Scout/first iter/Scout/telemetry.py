import threading
import time
import numpy as np
from pymavlink import mavutil

telemetry = {
    "lat": None,
    "lon": None,
    "h_agl": None,
    "roll": 0.0,
    "pitch": 0.0,
    "yaw": 0.0,
}
telemetry_lock = threading.Lock()

def rpy_to_R_body_to_NED(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    R_roll = np.array([[1, 0, 0],
                       [0, cr, -sr],
                       [0, sr,  cr]])

    R_pitch = np.array([[ cp, 0, sp],
                        [  0, 1,  0],
                        [-sp, 0, cp]])

    R_yaw = np.array([[cy, -sy, 0],
                      [sy,  cy, 0],
                      [ 0,   0, 1]])

    return R_yaw @ R_pitch @ R_roll

def telemetry_thread(port="/dev/ttyACM0", baud=57600): #check the port which we are using
    master = mavutil.mavlink_connection(port, baud=baud)
    master.wait_heartbeat()
    print("MAVLink connected")

    while True:
        msg = master.recv_match(blocking=True)
        if msg is None:
            continue

        with telemetry_lock:
            if msg.get_type() == "GLOBAL_POSITION_INT":
                telemetry["lat"]   = msg.lat / 1e7
                telemetry["lon"]   = msg.lon / 1e7
                telemetry["h_agl"] = msg.relative_alt / 1000.0  
            elif msg.get_type() == "ATTITUDE":
                telemetry["roll"]  = msg.roll
                telemetry["pitch"] = msg.pitch
                telemetry["yaw"]   = msg.yaw
        # tiny sleep to avoid busy-wait
        time.sleep(0.001)

t = threading.Thread(target=telemetry_thread, daemon=True)
t.start()
