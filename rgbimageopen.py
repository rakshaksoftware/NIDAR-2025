import pyrealsense2 as rs
import numpy as np
import cv2

# Create pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable color stream
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:
        # Wait for a coherent color frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert image to numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Show image
        cv2.imshow('RealSense RGB', color_image)

        # Save image and break on 's' key
        key = cv2.waitKey(1)
        if key & 0xFF == ord('s'):
            cv2.imwrite('color_image.jpg', color_image)
            print("Saved color_image.jpg")
            break
        elif key & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
