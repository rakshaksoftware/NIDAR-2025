from pymavlink import mavutil
import time

PORT = "/dev/ttyACM0"
BAUD = 115200

m = mavutil.mavlink_connection(PORT, baud=BAUD)

print("Waiting for heartbeat...")
m.wait_heartbeat()
print("Heartbeat OK from system %u component %u" % (m.target_system, m.target_component))

last_print = 0
while True:
    msg = m.recv_match(type=['GLOBAL_POSITION_INT', 'GPS_RAW_INT'], blocking=True, timeout=2)

    if msg is None:
        continue

    now = time.time()
    if now - last_print < 0.5:
        continue
    last_print = now

    if msg.get_type() == "GLOBAL_POSITION_INT":
        lat = msg.lat / 1e7
        lon = msg.lon / 1e7
        alt = msg.relative_alt / 1000.0  # meters
        print(f"GLOBAL_POSITION_INT  lat={lat:.7f} lon={lon:.7f} rel_alt={alt:.2f}m")

    elif msg.get_type() == "GPS_RAW_INT":
        lat = msg.lat / 1e7
        lon = msg.lon / 1e7
        alt = msg.alt / 1000.0  # meters
        sats = msg.satellites_visible
        fix = msg.fix_type
        print(f"GPS_RAW_INT lat={lat:.7f} lon={lon:.7f} alt={alt:.2f}m fix={fix} sats={sats}")
