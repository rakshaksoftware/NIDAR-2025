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
