from pymavlink import mavutil
import time

# Change port & baud if needed
PORT = "/dev/ttyACM0"
BAUD = 115200

master = mavutil.mavlink_connection(PORT, baud=BAUD)

print("[INFO] Waiting for heartbeat...")
master.wait_heartbeat()
print("[INFO] Heartbeat received")

print("[INFO] Waiting for GPS data...")

while True:
    msg = master.recv_match(type="GLOBAL_POSITION_INT", blocking=True)
    if msg is None:
        continue

    lat = msg.lat / 1e7
    lon = msg.lon / 1e7
    alt = msg.relative_alt / 1000.0  # meters

    print(f"LAT: {lat:.7f}, LON: {lon:.7f}, ALT: {alt:.2f} m")
    time.sleep(1)
