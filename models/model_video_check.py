from ultralytics import YOLO
import cv2

# 1. Load TensorRT engine
print("[INFO] Loading YOLO11n TensorRT engine...")
model = YOLO("best.engine", task="detect")  # Ensure the .engine file is in the same directory
print("[INFO] Engine loaded.")

# 2. Open the .mp4 video file
video_path = "dataset_raw3_640.mp4"  # Replace with the path to your video file
print(f"[INFO] Opening video file: {video_path}...")

cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print(f"[ERROR] Failed to open video file {video_path}.")
    raise SystemExit

print("[INFO] Video file opened successfully.")

# 3. Live loop to capture and process frames
while True:
    ret, frame = cap.read()

    if not ret:
        print("[INFO] End of video reached or failed to read frame.")
        break

    # YOLO11n inference (only detect 'person' class)
    results = model(frame, imgsz=640, device=0, classes=[0])
    annotated = results[0].plot()  # Annotate the frame with detections

    # Display the annotated frame
    cv2.imshow("YOLO11n TensorRT - Video", annotated)

    # Break loop if ESC is pressed
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()

print("[INFO] Stopped.")

