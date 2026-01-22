from ultralytics import YOLO
import cv2
import time

# Load two YOLO TensorRT engine models
model_1 = YOLO("best.engine", task="detect")  # Replace with your first model file

# Print loading info
print("[INFO] Loaded YOLO models.")

# Open the video file (replace with your video path)
video_path = "dataset_raw3_640.mp4"  # Replace with your .mp4 video file
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print(f"[ERROR] Failed to open video file {video_path}.")
    raise SystemExit

print("[INFO] Video file opened successfully.")

# Function to calculate FPS and latency for a model
def check_inference_latency(model, cap):
    frame_count = 0
    total_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Measure the time for inference
        start_time = time.time()

        # Run inference
        results = model(frame, imgsz=640, device=0)  # Inference on the frame
        annotated = results[0].plot()  # Annotate the frame with detection results

        # Measure end time for inference
        end_time = time.time()

        # Calculate latency (time taken per frame)
        inference_time = end_time - start_time
        total_time += inference_time
        frame_count += 1

        # Display the annotated frame
        cv2.imshow(f"{model} Inference", annotated)

        # Exit if ESC is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Calculate FPS and average latency
    fps = frame_count / total_time if total_time > 0 else 0
    avg_latency = total_time / frame_count if frame_count > 0 else 0

    return fps, avg_latency

# Function to run and evaluate both models
def evaluate_models(cap):
    print("[INFO] Evaluating Model 1...")
    fps_1, avg_latency_1 = check_inference_latency(model_1, cap)
    print(f"[INFO] Model 1 - FPS: {fps_1:.2f}, Average Latency: {avg_latency_1:.4f} seconds per frame")

    # Reset video capture to the beginning for the second model


# Run the evaluation
evaluate_models(cap)

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()

print("[INFO] Stopped.")

