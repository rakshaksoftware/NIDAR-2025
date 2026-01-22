from pymavlink import mavutil
import time

m = mavutil.mavlink_connection("tcp:127.0.0.1:5760")

text = "HUMAN,lat=19.1337000,lon=72.9158000,alt=12.30,conf=0.9"
for _ in range(8):
    m.mav.statustext_send(6, text.encode("utf-8")[:50])
    print("[SCOUT SENT]", text)
    time.sleep(0.2)
