from pymavlink import mavutil
import time

# Connect to MAVLink
m = mavutil.mavlink_connection("tcp:127.0.0.1:5760")

# List of coordinates (lat, lon)
coordinates = [
    (19.129260, 72.917976),
    (19.129119, 72.917995),
    (19.129108, 72.917853),
    (19.129212, 72.917749),
    (19.129201, 72.917885),
]

ALT = 4.0            # altitude set to 4 meters
REPEAT_COUNT = 6     # each coordinate sent 6 times
DELAY = 0.2          # seconds between messages

for idx, (lat, lon) in enumerate(coordinates, start=1):
    text = f"HUMAN,lat={lat:.6f},lon={lon:.6f},alt={ALT:.2f}"
    print(f"\n[SENDING COORD {idx}] {text}")

    for i in range(REPEAT_COUNT):
        m.mav.statustext_send(
            mavutil.mavlink.MAV_SEVERITY_INFO,
            text.encode("utf-8")[:50]
        )
        print(f"  -> Sent {i+1}/{REPEAT_COUNT}")
        time.sleep(DELAY)

print("\n[SCOUT] All coordinates sent successfully.")
