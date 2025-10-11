import cv2
import numpy as np
import onnxruntime as ort
import time

# --- Configuration ---
# Path to your ONNX model
MODEL_PATH = 'best.onnx'
# Camera ID (0 is usually the first camera/USB Arducam). Adjust as needed.
CAMERA_ID = 0 
# Target resolution for the model's input (YOLO models are often 640x640)
INPUT_WIDTH, INPUT_HEIGHT = 640, 640
# Confidence threshold to filter weak detections
CONF_THRESHOLD = 0.5
# Non-Maximum Suppression (NMS) threshold to remove overlapping boxes
IOU_THRESHOLD = 0.4
# Class ID for 'human' in your model's output (usually 0 for YOLO-trained models)
TARGET_CLASS_ID = 0
CLASS_NAME = 'human'

# --- Initialization ---

# 1. Load ONNX Session
print(f"Loading ONNX model from: {MODEL_PATH}")
try:
    # Use 'CUDAExecutionProvider' for GPU acceleration if available and set up
    # Otherwise, 'CPUExecutionProvider' is the default fallback
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(MODEL_PATH, providers=providers)
    
    # Get the input name from the ONNX graph
    input_name = session.get_inputs()[0].name
    print("ONNX Runtime session successfully created.")
except Exception as e:
    print(f"Error initializing ONNX Runtime session: {e}")
    exit()

# 2. Open Camera
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    print(f"Error: Could not open camera {CAMERA_ID}. Check connection and ID.")
    exit()

# Set camera frame size (optional, but can control performance)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print(f"Camera opened. Press 'q' to exit.")


# --- Helper Function for Preprocessing ---
def preprocess_frame(frame):
    """Resizes, scales, and transposes the frame for model input."""
    # Convert BGR to RGB (required by many YOLO models)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize the image to the model's input size (e.g., 640x640)
    resized = cv2.resize(img_rgb, (INPUT_WIDTH, INPUT_HEIGHT))
    
    # Scale pixel values to 0-1 range and convert to float32
    input_data = np.array(resized, dtype=np.float32) / 255.0
    
    # HWC -> CHW (Transpose to [1, 3, H, W] format)
    input_data = np.transpose(input_data, (2, 0, 1))
    input_tensor = np.expand_dims(input_data, 0)
    
    return input_tensor

# --- Helper Function for Post-processing (YOLO Output) ---
def postprocess_and_draw(frame, output, ratio_x, ratio_y):
    """
    Parses YOLO output, applies NMS, and draws boxes on the frame.
    
    Note: This function assumes a standard YOLO output format where 
    the bounding box is followed by class probabilities.
    """
    
    # The output is typically [1, N, 85] (N detections, 4 box + 1 confidence + 80 classes)
    detections = output[0].T  # Transpose to [85, N] or similar
    
    # Extract box and scores
    boxes = detections[0:4, :].T  # xywh (center x, center y, width, height)
    scores = detections[4, :]
    
    # The remaining columns are class scores (often 80 classes)
    class_scores = detections[5:, :].T
    
    # Get the best class score and index for each detection
    class_ids = np.argmax(class_scores, axis=1)
    
    # Combine object confidence with class confidence for final score
    final_scores = scores * np.max(class_scores, axis=1)

    # Filter detections based on confidence threshold
    valid_indices = np.where(final_scores >= CONF_THRESHOLD)[0]
    
    # Lists to hold final processed boxes, confidences, and class IDs
    filtered_boxes = []
    filtered_scores = []
    
    for i in valid_indices:
        # We only care about the target class ('human')
        if class_ids[i] == TARGET_CLASS_ID:
            box = boxes[i]
            
            # Convert box from center (xywh) to corner (xyxy) format
            center_x, center_y, w, h = box
            x1 = int((center_x - w / 2) * ratio_x)
            y1 = int((center_y - h / 2) * ratio_y)
            x2 = int((center_x + w / 2) * ratio_x)
            y2 = int((center_y + h / 2) * ratio_y)
            
            filtered_boxes.append([x1, y1, x2, y2])
            filtered_scores.append(final_scores[i])

    # Apply Non-Maximum Suppression (NMS) to remove redundant, overlapping boxes
    if filtered_boxes:
        indices = cv2.dnn.NMSBoxes(
            bboxes=filtered_boxes,
            scores=filtered_scores,
            score_threshold=CONF_THRESHOLD, 
            nms_threshold=IOU_THRESHOLD
        )
        
        # Draw final bounding boxes
        for i in indices:
            # OpenCV NMSBoxes returns a tuple/array of indices
            if isinstance(i, (np.ndarray, list)):
                 i = i[0]
            
            x1, y1, x2, y2 = filtered_boxes[i]
            confidence = filtered_scores[i]
            
            # Draw the bounding box (Green)
            color = (0, 255, 0) 
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw the label text
            label = f"{CLASS_NAME}: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame

# --- Main Detection Loop ---

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        
        # Get the original frame dimensions
        original_h, original_w = frame.shape[:2]
        # Calculate scaling ratios for remapping boxes to the original image size
        ratio_x = original_w / INPUT_WIDTH
        ratio_y = original_h / INPUT_HEIGHT

        # 1. Preprocess the frame
        input_tensor = preprocess_frame(frame)
        
        # 2. Run Inference
        # Note: input_name is retrieved from the ONNX session
        start_time = time.time()
        onnx_output = session.run(None, {input_name: input_tensor})
        end_time = time.time()
        
        # Optional: Print performance
        # print(f"Inference Time: {(end_time - start_time) * 1000:.2f} ms")

        # 3. Post-process and Draw
        frame = postprocess_and_draw(frame, onnx_output, ratio_x, ratio_y)
        
        # Display the resulting frame
        cv2.imshow('Human Detection (ONNX) on Jetson Orin Nano', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Exiting application.")
