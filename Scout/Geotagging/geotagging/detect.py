# detect_mavlink.py
import glob
import time
from pymavlink import mavutil

# candidate device patterns
candidates = []
candidates += glob.glob('/dev/ttyACM*')
candidates += glob.glob('/dev/ttyUSB*')
candidates += glob.glob('/dev/ttyS*')

# If none found, still try common names
if not candidates:
    candidates = ['/dev/ttyACM0', '/dev/ttyUSB0', '/dev/ttyS0']

baud_rates = [57600, 115200, 230400, 921600, 9600]

print("Scanning ports:", candidates)
for port in candidates:
    for baud in baud_rates:
        try:
            print(f"Trying {port} @ {baud} ...", end="", flush=True)
            master = mavutil.mavlink_connection(port, baud=baud, autoreconnect=False)
            # wait for a short heartbeat (timeout)
            hb = master.recv_match(type='HEARTBEAT', blocking=True, timeout=5)
            if hb:
                print("  --> FOUND")
                print(f"Working port: {port}, baud: {baud}")
                master.close()
                raise SystemExit(0)
            else:
                print("  --> no heartbeat")
            master.close()
        except Exception as e:
            print("  --> fail")
            # continue trying
print("No MAVLink heartbeat detected. Check cables, FC power, or try different adapters/ports.")
