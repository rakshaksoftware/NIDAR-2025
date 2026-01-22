# test_play.py
import subprocess
subprocess.Popen(["aplay", "/home/rabbit/police_siren.wav"])
print("spawned aplay")
