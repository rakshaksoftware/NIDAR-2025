import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# Load YOLO pretrained model (person class is included in COCO dataset)
# You can use 'yolov8n.pt' (nano), 'yolov8s.pt' (small), etc.
model = YOLO("yolov8n.pt")

# Configure depth/color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:
        # Wait for a coherent pair of frames
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert image to numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Run YOLO detection
        results = model(color_image)

        # Draw results on the frame
        annotated_frame = results[0].plot()  

        # Show image
        cv2.imshow("YOLO Human Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
