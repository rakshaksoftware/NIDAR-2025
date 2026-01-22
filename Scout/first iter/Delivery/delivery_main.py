import time
import csv
from payload import drop_payload
from logger import log_event, log_telemetry

from nav import (
    connect_vehicle,
    arm_and_takeoff,
    goto_gps,
    return_to_launch
)

from visual_servo import visual_servo



# CONFIG 

WAYPOINTS_FILE = "waypoints.csv"

TRANSIT_ALT = 10.0        # meters (safe cruise altitude)
DROP_ALT = 6.0            # meters (≈ 20 feet)
ASCEND_ALT = 10.0         # meters (post-drop safety climb)
ALT_TOL = 0.5             # meters tolerance

HOVER_TIME = 3.0          # seconds before vision
RELOAD_WAIT = 10.0        # seconds after RTL (manual reload)

MAX_SERVO_RETRIES = 3
SERVO_TIMEOUT = 20.0
MAX_DESCEND_TIME = 15.0

# STATES
INIT = "INIT"
TAKEOFF = "TAKEOFF"
GOTO_WAYPOINT = "GOTO_WAYPOINT"
DESCEND_FOR_DROP = "DESCEND_FOR_DROP"
HOVER = "HOVER"
VISUAL_SERVO = "VISUAL_SERVO"
DROP_PAYLOAD = "DROP_PAYLOAD"
ASCEND = "ASCEND"
RTL = "RTL"
NEXT_WAYPOINT = "NEXT_WAYPOINT"
MISSION_COMPLETE = "MISSION_COMPLETE"


def load_waypoints():
    waypoints = []

    with open(WAYPOINTS_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            waypoints.append({
                "lat": float(row["lat"]),
                "lon": float(row["lon"])
            })

    print(f"[MISSION] Loaded {len(waypoints)} waypoints")
    return waypoints



# MAIN STATE MACHINE

def main():

    state = INIT
    wp_index = 0
    servo_attempts = 0

    print("[MISSION] Connecting to vehicle...")
    vehicle = connect_vehicle("127.0.0.1:14550")  # CHANGE FOR REAL DRONE

    waypoints = load_waypoints()

    while True:

        if state == INIT:
            log_event(state, wp_index, "Mission initialized")
            print("[STATE] INIT")
            if not waypoints:
                print("[MISSION] No waypoints found")
                state = MISSION_COMPLETE
            else:
                state = TAKEOFF

        elif state == TAKEOFF:
            log_event(state, wp_index, "Arming and takeoff")
            print("[STATE] TAKEOFF")
            arm_and_takeoff(vehicle, TRANSIT_ALT)
            state = GOTO_WAYPOINT

        elif state == GOTO_WAYPOINT:
            log_event(state, wp_index, "Navigating to waypoint")
            wp = waypoints[wp_index]
            print(f"[STATE] GOTO_WAYPOINT {wp_index}")

            goto_gps(vehicle, wp["lat"], wp["lon"], TRANSIT_ALT)
            state = DESCEND_FOR_DROP

        elif state == DESCEND_FOR_DROP:
            log_event(state, wp_index, "Descending for payload drop")
            print("[STATE] DESCEND_FOR_DROP")

            current = vehicle.location.global_relative_frame
            goto_gps(vehicle, current.lat, current.lon, DROP_ALT)

            start_time = time.time()

            while True:
                alt = vehicle.location.global_relative_frame.alt
                print(f"[NAV] Altitude: {alt:.2f} m")

                if alt <= DROP_ALT + ALT_TOL:
                    log_telemetry(vehicle, state, wp_index)
                    print("[NAV] Drop altitude reached")
                    state = HOVER
                    break

                if time.time() - start_time > MAX_DESCEND_TIME:
                    print("[ERROR] Descend timeout, retrying waypoint")
                    state = GOTO_WAYPOINT
                    break

                time.sleep(0.5)


            state = HOVER

        elif state == HOVER:
            log_event(state, wp_index, "Hovering before visual servo")
            print("[STATE] HOVER")
            time.sleep(HOVER_TIME)
            state = VISUAL_SERVO

        elif state == VISUAL_SERVO:
            log_event(state, wp_index, f"Visual servo attempt {servo_attempts + 1}")
            print(f"[STATE] VISUAL_SERVO (attempt {servo_attempts + 1})")

            centered = visual_servo(
                vehicle,
                debug_view=False,
            )

            if centered:
                servo_attempts = 0
                state = DROP_PAYLOAD

            else:
                servo_attempts += 1

                if servo_attempts >= MAX_SERVO_RETRIES:
                    print("[FAIL] Visual servo failed too many times")
                    servo_attempts = 0
                    state = RTL   # abandon this waypoint safely

                else:
                    print("[RETRY] Visual servo retrying")
                    state = HOVER


        elif state == DROP_PAYLOAD:
            log_event(state, wp_index, "Payload dropped")
            print("[STATE] DROP_PAYLOAD")

            alt = vehicle.location.global_relative_frame.alt
            if alt > DROP_ALT + ALT_TOL:
                print("[SAFETY] Too high to drop, re-descending")
                state = DESCEND_FOR_DROP
                continue

            drop_payload(vehicle)
            state = ASCEND

        elif state == ASCEND:
            log_event(state, wp_index, "Ascending post-drop")
            print("[STATE] ASCEND")

            current = vehicle.location.global_relative_frame
            goto_gps(vehicle, current.lat, current.lon, ASCEND_ALT)

            state = RTL

        elif state == RTL:
            log_event(state, wp_index, "Returning to launch")
            print("[STATE] RTL")    
            return_to_launch(vehicle)

            print("[MISSION] Waiting for reload...")
            time.sleep(RELOAD_WAIT)
            state = NEXT_WAYPOINT
        
        elif state == NEXT_WAYPOINT:
            log_event(state, wp_index, "Proceeding to next waypoint")
            print("[STATE] NEXT_WAYPOINT")
            wp_index += 1
            servo_attempts = 0  

            if wp_index >= len(waypoints):
                print("[MISSION] All waypoints completed")
                state = MISSION_COMPLETE
            else:
                print(f"[MISSION] Proceeding to waypoint {wp_index}")
                state = TAKEOFF
        
        elif state == MISSION_COMPLETE:
            log_event(state, wp_index, "Mission complete")
            print("[MISSION] Mission complete. Standing by.")
            break