#!/usr/bin/env python3
from pymavlink import mavutil
import time
import random

# Use whatever works in your scout setup today:
CONN = "tcp:127.0.0.1:5760"   # (or udpout:..., etc)

SEND_SEV = 6
ACK_SEV = 6

RETRY_INTERVAL = 0.25   # seconds between re-sends (tune 0.15–0.4)
MAX_RETRIES = 10        # 40 * 0.25 = 10s max per coordinate
ACK_PREFIX = "ACK,"
H_PREFIX = "HUMAN,"         # H,<id>,<lat>,<lon>,<alt>

def send_statustext(m, text, sev=SEND_SEV):
    m.mav.statustext_send(sev, text.encode("utf-8")[:50])

def wait_ack(m, target_id, timeout_s):
    """Wait up to timeout_s for ACK,<target_id>"""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        msg = m.recv_match(type="STATUSTEXT", blocking=True, timeout=0.1)
        if not msg:
            continue
        text = msg.text
        if isinstance(text, (bytes, bytearray)):
            text = text.decode("utf-8", errors="ignore")
        else:
            text = str(text)

        if text.startswith(ACK_PREFIX):
            parts = text.split(",")
            if len(parts) >= 2:
                try:
                    ack_id = int(parts[1])
                    if ack_id == target_id:
                        return True
                except:
                    pass
    return False

def send_coordinate_reliably(m, coord_id, lat, lon, alt):
    payload = f"HUMAN,{coord_id},{lat:.6f},{lon:.6f},{alt:.1f}"
    for attempt in range(1, MAX_RETRIES + 1):
        send_statustext(m, payload)
        print(f"[SCOUT] SENT (attempt {attempt}/{MAX_RETRIES}): {payload}")

        # wait for ACK within retry interval
        if wait_ack(m, coord_id, timeout_s=RETRY_INTERVAL):
            print(f"[SCOUT] ✅ ACK received for id={coord_id}")
            return True

    print(f"[SCOUT] ❌ NO ACK after retries for id={coord_id} (giving up)")
    return False

def new_id():
    # short int id (fits in 50 chars easily)
    return random.randint(1, 2_000_000_000)

def main():
    m = mavutil.mavlink_connection(CONN)
    m.wait_heartbeat()
    print("[SCOUT] Connected")

    detections = [
        (28.422471, 77.526201 , 4.0),
        (28.42275, 77.52568, 4.0),
        (28.42275, 77.52579, 4.0),
        (28.4225510, 77.5263417, 4.0),
        (28.422346, 77.526494, 4.0),
    ]

    for (lat, lon, alt) in detections:
        cid = new_id()
        ok = send_coordinate_reliably(m, cid, lat, lon, alt)
        if not ok:
            # optional: keep it in a queue to retry later
            pass

if __name__ == "__main__":
    main()
