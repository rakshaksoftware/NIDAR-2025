from pymavlink import mavutil
import time

print("[SCOUT] Connecting to mavlink-router TCP 127.0.0.1:5760")
m = mavutil.mavlink_connection("tcp:127.0.0.1:5760")

# Optional: if wait_heartbeat blocks sometimes, give it a timeout
m.wait_heartbeat(timeout=10)
print("[SCOUT] Connected")

def send_human(lat, lon, alt=None, conf=None):
    parts = [f"HUMAN,lat={lat:.6f}", f"lon={lon:.6f}"]
    if alt is not None:
        parts.append(f"alt={alt:.2f}")
    if conf is not None:
        parts.append(f"co={conf:.2f}")   # shorter key

    text = ",".join(parts)

    # hard cap safety (STATUSTEXT limit)
    text = text[:50]

    for _ in range(3):
        m.mav.statustext_send(6, text.encode("utf-8")[:50])
        time.sleep(0.15)

    print("[SCOUT SENT]", text, "| len=", len(text))

