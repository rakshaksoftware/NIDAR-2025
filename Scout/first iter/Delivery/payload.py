import time
from pymavlink import mavutil


SERVO_CHANNEL = 9          # AUX1 = SERVO9
SERVO_OPEN_PWM = 1900      # release position
SERVO_CLOSE_PWM = 1100     # locked position
SERVO_PULSE_TIME = 1.0     # seconds


# LOW-LEVEL SERVO COMMAND

def set_servo(vehicle, channel, pwm):
    """
    Sends a MAVLink command to set servo PWM.
    """
    msg = vehicle.message_factory.command_long_encode(
        0, 0,
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
        0,
        channel,
        pwm,
        0, 0, 0, 0, 0
    )
    vehicle.send_mavlink(msg)
    vehicle.flush()


# PAYLOAD DROP SEQUENCE

def drop_payload(vehicle):
    """
    Executes payload drop safely:
    1. Open servo
    2. Wait
    3. Close servo
    """
    print("[PAYLOAD] Releasing payload")

    # Open
    set_servo(vehicle, SERVO_CHANNEL, SERVO_OPEN_PWM)
    time.sleep(SERVO_PULSE_TIME)

    # Close (important for next payload)
    set_servo(vehicle, SERVO_CHANNEL, SERVO_CLOSE_PWM)

    print("[PAYLOAD] Payload released")
