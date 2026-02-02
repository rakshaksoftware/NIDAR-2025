"""
Dual Mission Planner Bridge (Windows) + Reliable HUMAN relay

- Scout COM14 <-> Mission Planner UDP 14551 (full-duplex => params OK)
- Delivery COM8 <-> Mission Planner UDP 14550 (full-duplex => params OK)
- Relay: Scout STATUSTEXT starting with "HUMAN," -> Delivery STATUSTEXT

Install:
  py -m pip install pyserial pymavlink
Run:
  py dual_mp_bridge_and_relay_final.py

MP:
  Window #1 -> UDP 14551 (Scout)
  Window #2 -> UDP 14550 (Delivery)
"""

import time
import socket
import serial
from pymavlink import mavutil

SCOUT_COM = "COM14"
DELIVERY_COM = "COM8"
BAUD = 57600

SCOUT_MP_PORT = 14551
DELIVERY_MP_PORT = 14550

LOCAL_SCOUT_UDP_PORT = 14561
LOCAL_DELIVERY_UDP_PORT = 14560

PREFIX = "HUMAN,"


def open_serial(port: str) -> serial.Serial:
    return serial.Serial(port=port, baudrate=BAUD, timeout=0)


def open_udp_bound(local_port: int) -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", local_port))
    s.setblocking(False)
    return s


def main():
    # Raw serial streams for MP full-duplex bridge
    scout_ser = open_serial(SCOUT_COM)
    delivery_ser = open_serial(DELIVERY_COM)

    # Also open pymavlink decoders for *reliable* STATUSTEXT detection
    # IMPORTANT: this uses the same COM ports, so we must NOT open them twice.
    # Solution: decode from the same bytes we read from scout_ser using mavutil.MAVLink
    mav = mavutil.mavlink.MAVLink(None)

    scout_udp = open_udp_bound(LOCAL_SCOUT_UDP_PORT)
    delivery_udp = open_udp_bound(LOCAL_DELIVERY_UDP_PORT)

    scout_mp_addr = ("127.0.0.1", SCOUT_MP_PORT)
    delivery_mp_addr = ("127.0.0.1", DELIVERY_MP_PORT)

    print(f"[OK] Scout   {SCOUT_COM} @ {BAUD}  <-> UDP(local {LOCAL_SCOUT_UDP_PORT}) <-> MP UDP {SCOUT_MP_PORT}")
    print(f"[OK] Delivery{DELIVERY_COM} @ {BAUD}  <-> UDP(local {LOCAL_DELIVERY_UDP_PORT}) <-> MP UDP {DELIVERY_MP_PORT}")
    print("[OK] Full-duplex enabled for BOTH (params should download in both MP windows).")
    print(f"[OK] Relay enabled: Scout STATUSTEXT starting '{PREFIX}' -> Delivery STATUSTEXT")

    last_relay = 0.0
    carry = bytearray()

    while True:
        # ---------------- Scout serial -> MP (and decode for relay) ----------------
        b = scout_ser.read(4096)
        if b:
            scout_udp.sendto(b, scout_mp_addr)

            # Reliable decode: feed bytes to mavutil MAVLink parser (handles noise better)
            carry.extend(b)
            try:
                msgs = mav.parse_buffer(bytes(carry))  # may return list or None
                carry.clear()
            except Exception:
                # if parsing blows up, drop buffer to avoid infinite growth
                msgs = None
                carry.clear()

            if msgs:
                for msg in msgs:
                    if msg.get_type() == "STATUSTEXT":
                        text = msg.text
                        if isinstance(text, (bytes, bytearray)):
                            text = text.decode("utf-8", errors="ignore")
                        else:
                            text = str(text)

                        if text.startswith(PREFIX):
                            now = time.time()
                            if now - last_relay > 0.2:
                                last_relay = now

                                # Build a new STATUSTEXT packet and write to Delivery serial
                                sev = int(getattr(msg, "severity", 6))
                                pkt = mavutil.mavlink.MAVLink_statustext_message(
                                    sev, text.encode("utf-8")[:50]
                                ).pack(mav)

                                delivery_ser.write(pkt)
                                time.sleep(0.05)
                                delivery_ser.write(pkt)

                                print("[RELAYED]", text)

        # ---------------- MP -> Scout serial (params/control) ----------------
        try:
            data, _ = scout_udp.recvfrom(4096)
            if data:
                scout_ser.write(data)
        except (BlockingIOError, ConnectionResetError):
            pass

        # ---------------- Delivery serial -> MP ----------------
        b2 = delivery_ser.read(4096)
        if b2:
            delivery_udp.sendto(b2, delivery_mp_addr)

        # ---------------- MP -> Delivery serial (params/control) ----------------
        try:
            data2, _ = delivery_udp.recvfrom(4096)
            if data2:
                delivery_ser.write(data2)
        except (BlockingIOError, ConnectionResetError):
            pass

        time.sleep(0.001)


if __name__ == "__main__":
    main()