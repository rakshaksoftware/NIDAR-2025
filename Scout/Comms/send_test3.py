#!/usr/bin/env python3
from pymavlink import mavutil
import time

# Scout -> GCS/router connection
#   "tcp:127.0.0.1:5760"  (Mission Planner bridge)
#   "udp:127.0.0.1:14650" (mavlink-router scriptin)
CONN = "tcp:127.0.0.1:5760"

ALT = 4.0
CO  = 0.92   # confidence

coords = [
        (28.422471, 77.526201 ),
        (28.422696, 77.526284),
        (28.422671, 77.526464),
        (28.422378, 77.526433),
        (28.422265, 77.526249),
    ]
REPEAT_EACH = 6
GAP_S = 0.20
SEPARATOR_GAP_S = 0.5

def build_payload(lat, lon, alt, co):
    """
    Build a HUMAN message that always fits STATUSTEXT (<= 50 chars).
    Uses 'co=' (shorter than 'conf=') and reduces lat/lon decimals if needed.
    """
    # Try 6 decimals first
    txt = f"HUMAN,lat={lat:.6f},lon={lon:.6f},alt={alt:.1f},co={co:.2f}"
    if len(txt) <= 50:
        return txt

    # Try 5 decimals
    txt = f"HUMAN,lat={lat:.5f},lon={lon:.5f},alt={alt:.1f},co={co:.2f}"
    if len(txt) <= 50:
        return txt

    # Try 4 decimals
    txt = f"HUMAN,lat={lat:.4f},lon={lon:.4f},alt={alt:.1f},co={co:.2f}"
    if len(txt) <= 50:
        return txt

    # Last resort: shorten co precision
    txt = f"HUMAN,lat={lat:.4f},lon={lon:.4f},alt={alt:.1f},co={co:.1f}"
    return txt[:50]

def main():
    print(f"[SCOUT] Connecting: {CONN}")
    m = mavutil.mavlink_connection(CONN)

    try:
        m.wait_heartbeat(timeout=5)
        print("[SCOUT] Heartbeat OK")
    except Exception:
        print("[SCOUT] No heartbeat (still sending STATUSTEXT anyway)")

    for i, (lat, lon) in enumerate(coords, start=1):
        text = build_payload(lat, lon, ALT, CO)

        for k in range(REPEAT_EACH):
            m.mav.statustext_send(6, text.encode("utf-8"))  # already <=50
            print(f"[SCOUT SENT] WP{i} rep {k+1}/{REPEAT_EACH}: {text} | len={len(text)}")
            time.sleep(GAP_S)

        time.sleep(SEPARATOR_GAP_S)

    print("[SCOUT] Done sending 5 dummy coordinates.")

if __name__ == "__main__":
    main()
