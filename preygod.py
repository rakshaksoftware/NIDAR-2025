import cv2
import numpy as np
import pyrealsense2 as rs
import onnxruntime as ort

# Load YOLO ONNX model (exported YOLOv8 .onnx file)
onnx_model_path = "yolov8n.onnx"
session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

# Get model input/output details
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
input_shape = session.get_inputs()[0].shape  # e.g. [1, 3, 640, 640]

# YOLO class 0 = person
PERSON_CLASS_ID = 0

def preprocess(image, new_shape=(640, 640)):
    """Resize, normalize, BGR->RGB, CHW format"""
    img = cv2.resize(image, new_shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)   # Add batch dimension
    return img

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        # Preprocess for YOLO
        img_input = preprocess(color_image, (640, 640))

        # Run inference
        preds = session.run([output_name], {input_name: img_input})[0]

        # preds shape is [batch, num_boxes, (cx, cy, w, h, conf, class...)]
        preds = preds[0]  # remove batch dim

        h, w, _ = color_image.shape
        for det in preds:
            conf = det[4]
            cls = int(det[5])
            if cls == PERSON_CLASS_ID and conf > 0.5:
                # Convert YOLO format (cx, cy, w, h) to xyxy
                cx, cy, bw, bh = det[0:4]
                x1 = int((cx - bw/2) * w / 640)
                y1 = int((cy - bh/2) * h / 640)
                x2 = int((cx + bw/2) * w / 640)
                y2 = int((cy + bh/2) * h / 640)

                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(color_image, f"Person {conf:.2f}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("YOLO ONNX Person Detection", color_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
