from ultralytics import YOLO
import cv2

# 1. Load TensorRT engine
print("[INFO] Loading YOLO11n TensorRT engine...")
model = YOLO("yolo11n.engine", task="detect")  # Ensure the .engine file is in the same directory
print("[INFO] Engine loaded.")

# Set up GStreamer pipeline for the camera (IMX477)
gst = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! "
    "appsink drop=1"
)

print("[INFO] Opening camera with GStreamer...")
cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)

# Check if the camera opened successfully
if not cap.isOpened():
    print("[ERROR] Failed to open IMX477 via GStreamer. Check pipeline or OpenCV build.")
    raise SystemExit

print("[INFO] Camera opened successfully.")

# 3. Live loop to capture and process frames
while True:
    ret, frame = cap.read()

    if not ret:
        print("[ERROR] Failed to read frame from camera.")
        break

    # YOLO11n inference (only detect 'person' class)
    results = model(frame, imgsz=640, device=0, classes=[0])
    annotated = results[0].plot()  # Annotate the frame with detections

    # Display the annotated frame
    cv2.imshow("YOLO11n TensorRT - IMX477", annotated)

    # Break loop if ESC is pressed
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()

print("[INFO] Stopped.")

