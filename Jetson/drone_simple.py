from dronekit import connect, VehicleMode
from pymavlink import mavutil
import time

# --- Set the connection string for UDP
# This script will listen for incoming UDP packets on port 14550.
# Ensure your flight controller is configured to send MAVLink data
# to this computer's IP address on port 14550.
connection_string = "udpin:0.0.0.0:14550"

# --- Connect to the vehicle
print(f"Waiting for vehicle to connect on: {connection_string}")
# Baud rate is not needed for UDP connections
vehicle = connect(connection_string, wait_ready=True)
print("Vehicle connected!")

def arm_and_takeoff(target_altitude):
    """
    Arms vehicle and fly to aTargetAltitude.
    """
    print("Pre-arm checks")
    # Wait for the vehicle to be ready to arm
    while not vehicle.is_armable:
        print(" Waiting for vehicle to initialize...")
        time.sleep(1)

    print("Arming motors")
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    # Confirm vehicle armed before attempting to take off
    while not vehicle.armed:
        print(" Waiting for arming...")
        time.sleep(1)

    print(f"Taking off to {target_altitude}m!")
    vehicle.simple_takeoff(target_altitude)

    # Wait until the vehicle reaches a safe height
    while True:
        altitude = vehicle.location.global_relative_frame.alt
        print(f" Altitude: {altitude:.1f}m")
        # Break and return from function just below target altitude.
        if altitude >= target_altitude * 0.95:
            print("Reached target altitude")
            break
        time.sleep(1)

def move_forward(distance_m):
    """
    Moves the drone forward relative to its current orientation.
    """
    print(f"Moving forward {distance_m} meters")
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,      # time_boot_ms (not used)
        0, 0,   # target system, target component
        # Frame of reference is relative to the drone's body
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        0b0000111111111000, # A special "mask" to only use position
        distance_m, 0, 0,   # x, y, z positions (m)
        0, 0, 0,            # x, y, z velocity (not used)
        0, 0, 0,            # x, y, z acceleration (not used)
        0, 0)               # yaw, yaw_rate (not used)

    # Send the command to the vehicle
    vehicle.send_mavlink(msg)


# --- Main script execution ---
try:
    # 1. Arm and take off to an altitude of 3 meters
    arm_and_takeoff(3)

    # 2. Move forward 5 meters
    move_forward(5)

    # 3. Wait for 6 seconds to allow the drone to complete the movement
    print("Waiting for movement to complete...")
    time.sleep(6)

    # 4. Land the drone
    print("Landing...")
    vehicle.mode = VehicleMode("LAND")

    # Wait until the drone has disarmed
    while vehicle.armed:
        print(" Waiting for disarm...")
        time.sleep(1)

finally:
    # 5. Close the connection
    print("Mission Complete. Closing connection.")
    vehicle.close()
