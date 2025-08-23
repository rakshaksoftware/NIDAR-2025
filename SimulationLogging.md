## 24 AUG
- Install homebrew  
- Install Ardupilot SITl  
- Install Gazebo Harmonic   

- On macOS, you will need to run Gazebo using two terminals, one for the server and another for the GUI:  
launch server in one terminal  
`gz sim -v 4 shapes.sdf -s  # Fortress uses "ign gazebo" instead of "gz sim"`    
launch gui in a separate terminal  
`gz sim -v 4 -g  # Fortress uses "ign gazebo" instead of "gz sim"`
