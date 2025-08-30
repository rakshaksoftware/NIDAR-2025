# Jetson Nano – YOLOv8 & Jetson Inference Setup  

## Overview  
This project documents the process of setting up object detection on the **Jetson Nano**, using both **YOLOv8 (ONNX → TensorRT)** and the **built-in Jetson Inference models**. The goal is to test and compare performance on images (and later, live webcam feed).  

---

## Progress Log  

### Day 1  
- Flashed the Jetson Nano SD card and booted successfully.  
- Downloaded the YOLOv8 `.onnx` model file.  
- Encountered **webcam issues**:  
  - Tried switching from `camera=0` to `camera=1`.  
  - Tested via `jetson-inference`, `detectnet`, etc.  
  - Still unable to detect the webcam (likely a camera hardware/driver issue).  
- Verified that Jetson Nano comes with its own **inference models**.  
- Plan: Compare YOLOv8 vs Jetson Inference once the webcam issue is resolved.  
- Installed `jetson-inference` but errors persisted during execution.  

---

### Day 2  
- **Docker container setup:**  
  - Successfully ran Jetson Inference inside Docker.  
  - Learned useful commands (`Ctrl+Shift+C` for copy inside container).  
- **ONNX to TensorRT conversion:**  
  - Converting YOLOv8 `.onnx` file to `.engine` file.  
- **Testing detectnet.py:**  
  - Successfully tested the inbuilt **DetectNet** model on an image.  
- **Installed PyCUDA:**  
  - Followed this guide and installation was successful:  
    👉 [PyCUDA on Jetson Nano](https://medium.com/dropout-analytics/pycuda-on-jetson-nano-7990decab299)  
- Running YOLOv8 inference on **images** (instead of webcam) for now, comparing results against Jetson’s built-in model.  

---

## Running a New Docker Container  
```bash
./docker/run.sh
```

### 28 Aug
- To install pyrealsense: https://github.com/IntelRealSense/librealsense/issues/6964

- Changed desktop appearance
- Found intel realsense viewer depth camera
- First install SDK package of realsense
- **Building librealsense - Using vcpkg**:: https://github.com/IntelRealSense/librealsense
   - You can download and install librealsense using the vcpkg dependency manager:
   - git clone https://github.com/Microsoft/vcpkg.git
   - cd vcpkg
     -./bootstrap-vcpkg.sh
     - ./vcpkg integrate install
     - ./vcpkg install realsense2
   - But this is of no use
   - Then again tried installing librealsense
   - https://github.com/IntelRealSense/librealsense/blob/master/doc/installation_jetson.md
- but then to use for python, want to install **pyrealsense2**
   - https://github.com/IntelRealSense/librealsense/issues/6964
   - https://github.com/IntelRealSense/librealsense/releases/

- tried installing this but now its asking for python greater version
- Installing Python 3.9: https://arcanesciencelab.wordpress.com/2021/02/14/building-python-3-9-1-on-jetpack-4-5-and-the-jetson-xavier-nx/
- But this command cmake ../ -DFORCE_RSUSB_BACKEND=ON -DBUILD_PYTHON_BINDINGS:bool=true -DPYTHON_EXECUTABLE=... gave error as it was detecting python older version only
- So, changed command to cmake ../ -DFORCE_RSUSB_BACKEND=ON -DBUILD_PYTHON_BINDINGS:bool=true -DPYTHON_EXECUTABLEusr/bin/local/python3.9 {directory of new python version}. And, now it was able to access.
- 
### 30 Aug

- To check pyrealsense2 :
```bash
python3.9 -m pip show pyrealsense2
``` 
- But on import pyrealsense2 , it gives error as version `GLIBCXX_3.4.26' not found

