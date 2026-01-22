from nav import *

vehicle = connect_vehicle("127.0.0.1:14550")  # SITL or telemetry port
arm_and_takeoff(vehicle, 10)

#safe test coordinates
goto_gps(vehicle, 19.1334, 72.9132, 10)

time.sleep(5)
return_to_launch(vehicle)
