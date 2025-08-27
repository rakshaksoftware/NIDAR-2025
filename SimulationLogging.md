## 24 AUG
- Install homebrew  
- Install Ardupilot SITl  
- Install Gazebo Harmonic   

- On macOS, you will need to run Gazebo using two terminals, one for the server and another for the GUI:  
launch server in one terminal  
`gz sim -v 4 shapes.sdf -s  # Fortress uses "ign gazebo" instead of "gz sim"`    
launch gui in a separate terminal  
`gz sim -v 4 -g  # Fortress uses "ign gazebo" instead of "gz sim"`

## 25 AUG
> While setting up ardupilot SITL, we kept facing an issue where running the sim_vehicle.py file in our virtual environment kept giving the error "future module not found", even though the module was installed on the system. Turns out, the problem was that MAVProxy was installed globally on our system but not inside our virtual environment, so sim_vehicle.py couldn’t detect MAVProxy. Once we reinstalled MAVProxy inside the venv, ardupilot started running fine.

- In order to run `sim_vehicle.py` script, you need to be in `ardupilot/ArduCopter` directory (maybe that determines the vehicle type)
