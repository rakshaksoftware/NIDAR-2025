from pymavlink import mavutil
import time

PORT = "/dev/ttyACM0"   # change only if needed
BAUD = 115200           # try 57600 if this fails

print("[SANITY] Connecting to MAVLink...")
m = mavutil.mavlink_connection(PORT, baud=BAUD)

print("[SANITY] Waiting for heartbeat...")
m.wait_heartbeat(timeout=10)
print("[SANITY] Heartbeat OK")

print("[SANITY] Listening for telemetry...\n")

while True:
    msg = m.recv_match(
        type=["GLOBAL_POSITION_INT", "ATTITUDE"],
        blocking=True,
        timeout=5
    )

    if msg is None:
        print("[SANITY] No telemetry in last 5 seconds")
        continue

    if msg.get_type() == "GLOBAL_POSITION_INT":
        lat = msg.lat / 1e7
        lon = msg.lon / 1e7
        alt = msg.relative_alt / 1000.0
        print(f"[GPS] lat={lat:.7f}, lon={lon:.7f}, alt={alt:.2f}")

    elif msg.get_type() == "ATTITUDE":
        print(
            f"[ATT] roll={msg.roll:.3f}, "
            f"pitch={msg.pitch:.3f}, "
            f"yaw={msg.yaw:.3f}"
        )
