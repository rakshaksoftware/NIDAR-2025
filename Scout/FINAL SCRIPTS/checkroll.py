from pymavlink import mavutil
import math
import time

# CHANGE THIS PORT IF NEEDED
PORT = "/dev/ttyACM0"   # or "/dev/ttyTHS1"
BAUD = 115200

print(f"Connecting to MAVLink on {PORT}...")
master = mavutil.mavlink_connection(PORT, baud=BAUD)

# Wait for heartbeat
master.wait_heartbeat()
print("Heartbeat received")

# IMPORTANT: request attitude data stream
master.mav.request_data_stream_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_DATA_STREAM_EXTRA1,  # ATTITUDE lives here
    10,   # Hz
    1
)

print("Streaming ATTITUDE messages...\n")

while True:
    msg = master.recv_match(type="ATTITUDE", blocking=True)
    if msg is None:
        continue

    # Radians → Degrees
    roll_deg  = math.degrees(msg.roll)
    pitch_deg = math.degrees(msg.pitch)
    yaw_deg   = math.degrees(msg.yaw)

    print(
        f"ROLL: {roll_deg:7.2f}° | "
        f"PITCH: {pitch_deg:7.2f}° | "
        f"YAW: {yaw_deg:7.2f}°"
    )

    time.sleep(0.05)
