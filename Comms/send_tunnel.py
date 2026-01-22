from pymavlink import mavutil
import struct, time

UART="/dev/ttyACM0"
BAUD=115200

DELIVERY_SYSID = 2
DELIVERY_COMPID = 191  # companion computer

m = mavutil.mavlink_connection(UART, baud=BAUD, source_system=1, source_component=191)
hb = m.wait_heartbeat(timeout=15)
if not hb:
    print("No heartbeat from SCOUT Pixhawk on UART.")
    raise SystemExit

print("Scout: heartbeat OK. Sending TUNNEL coordinate packet...")

lat_deg = 19.1330000
lon_deg = 72.9130000

lat_e7 = int(lat_deg * 1e7)
lon_e7 = int(lon_deg * 1e7)
seq = int(time.time()) & 0xFFFFFFFF

# Pack: seq(uint32), lat_e7(int32), lon_e7(int32)
payload12 = struct.pack("<Iii", seq, lat_e7, lon_e7)

# TUNNEL payload field is fixed 128 bytes
payload128 = payload12 + bytes(128 - len(payload12))

PAYLOAD_TYPE = 1

# IMPORTANT: pass payload as list of ints for pymavlink compatibility
m.mav.tunnel_send(
    DELIVERY_SYSID,
    DELIVERY_COMPID,
    PAYLOAD_TYPE,
    len(payload12),
    list(payload128)
)

print("Scout: sent.")
