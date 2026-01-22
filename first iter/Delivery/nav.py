from dronekit import connect, VehicleMode, LocationGlobalRelative
import time
import math

# ---------------- CONNECT ----------------
def connect_vehicle(connection_string):
    print(f"[NAV] Connecting to vehicle on {connection_string}")
    vehicle = connect(connection_string, wait_ready=True)
    print("[NAV] Connected")
    return vehicle


# ---------------- ARM & TAKEOFF ----------------
def arm_and_takeoff(vehicle, target_alt):
    print("[NAV] Pre-arm checks")
    while not vehicle.is_armable:
        print("[NAV] Waiting for vehicle to initialise...")
        time.sleep(1)

    print("[NAV] Arming motors")
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    while not vehicle.armed:
        print("[NAV] Waiting for arming...")
        time.sleep(1)

    print(f"[NAV] Taking off to {target_alt} m")
    vehicle.simple_takeoff(target_alt)

    while True:
        alt = vehicle.location.global_relative_frame.alt
        print(f"[NAV] Altitude: {alt:.2f} m")
        if alt >= 0.95 * target_alt:
            print("[NAV] Target altitude reached")
            break
        time.sleep(1)


# ---------------- DISTANCE CALC ----------------
def get_distance_meters(loc1, loc2):
    """
    Approx distance between two GPS locations.
    Good enough for short distances (<100m).
    """
    dlat = loc2.lat - loc1.lat
    dlon = loc2.lon - loc1.lon
    return math.sqrt(dlat*dlat + dlon*dlon) * 1.113195e5


# ---------------- GOTO WAYPOINT ----------------
def goto_gps(vehicle, lat, lon, alt, threshold=1.5):
    print(f"[NAV] Going to waypoint lat={lat}, lon={lon}, alt={alt}")
    target = LocationGlobalRelative(lat, lon, alt)
    vehicle.simple_goto(target)

    while True:
        current = vehicle.location.global_relative_frame
        dist = get_distance_meters(current, target)
        print(f"[NAV] Distance to target: {dist:.2f} m")

        if dist <= threshold:
            print("[NAV] Waypoint reached")
            break

        time.sleep(1)


# ---------------- RTL ----------------
def return_to_launch(vehicle):
    print("[NAV] Returning to launch")
    vehicle.mode = VehicleMode("RTL")
