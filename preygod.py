import cv2
import numpy as np
import pyrealsense2 as rs

# --- Constants ---
MODEL_PATH = "yolov8n.onnx"  # Make sure this model is in the same directory
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.5  # Confidence threshold for a detection to be considered valid
NMS_THRESHOLD = 0.45   # Non-Maximum Suppression threshold to remove duplicate boxes
CONFIDENCE_THRESHOLD = 0.5 # Confidence threshold for the class score

# Class names for COCO dataset, which YOLOv8 is trained on. "person" is at index 0.
CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

# --- RealSense Camera Setup ---
print("Configuring Intel RealSense camera...")
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

# Enable color stream
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)
print("Camera feed started.")

# --- Model Loading ---
print(f"Loading YOLOv8 model from '{MODEL_PATH}'...")
try:
    net = cv2.dnn.readNet(MODEL_PATH)
except cv2.error as e:
    print(f"Error loading the model: {e}")
    print("Please ensure the model file 'yolov8n.onnx' is in the same directory.")
    exit()
print("Model loaded successfully.")


def preprocess(image):
    """Prepares the image for the YOLOv8 model."""
    row, col, _ = image.shape
    max_dim = max(row, col)
    # Create a square image with black pixels as padding
    square_image = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
    square_image[0:row, 0:col] = image

    # Resize to the model's input size and create a blob
    blob = cv2.dnn.blobFromImage(square_image, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    return blob, max_dim

def postprocess(output, max_dim, image):
    """Processes the model output to get bounding boxes for people."""
    predictions = np.squeeze(output[0]).T

    boxes = []
    confidences = []
    class_ids = []

    # Calculate scaling factors
    x_factor = max_dim / INPUT_WIDTH
    y_factor = max_dim / INPUT_HEIGHT

    for pred in predictions:
        # The first 4 elements are bbox, the rest are class scores
        confidence = np.max(pred[4:])
        if confidence > CONFIDENCE_THRESHOLD:
            class_id = np.argmax(pred[4:])
            # Check if the detected object is a person (class_id 0)
            if class_id == 0:
                # Extract bounding box coordinates
                cx, cy, w, h = pred[:4]
                
                # Scale coordinates back to original image size
                left = int((cx - w/2) * x_factor)
                top = int((cy - h/2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                
                box = [left, top, width, height]
                
                boxes.append(box)
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression to remove redundant boxes
    indices = cv2.dnn.NMSBoxes(boxes, np.array(confidences), SCORE_THRESHOLD, NMS_THRESHOLD)
    
    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            
            # Draw bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw label
            label = f"Person: {confidences[i]:.2f}"
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            top = max(y, label_size[1])
            cv2.rectangle(image, (x, top - label_size[1]), (x + label_size[0], top + base_line), (0, 255, 0), cv2.FILLED)
            cv2.putText(image, label, (x, top), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    return image

# --- Main Loop ---
try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        # Pre-process the image
        blob, scale_factor = preprocess(color_image)

        # Set the input to the network
        net.setInput(blob)

        # Forward pass through the network
        outputs = net.forward(net.getUnconnectedOutLayersNames())

        # Post-process the output
        result_image = postprocess(outputs, scale_factor, color_image)

        # Display the resulting frame
        cv2.imshow("RealSense Person Detection", result_image)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    print("Stopping camera feed.")
    pipeline.stop()
    cv2.destroyAllWindows()
