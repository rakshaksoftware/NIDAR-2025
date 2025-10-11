import cv2
import numpy as np

# Load the ONNX model
model_path = "best.onnx"
net = cv2.dnn.readNetFromONNX(model_path)

# (Optional) Use CUDA if available
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Load class names — assuming only one class: "human"
classes = ["human"]

# Initialize video capture (0 = webcam, or give path to a video file)
cap = cv2.VideoCapture(0)

# Set input image size (depends on training size, e.g., 640x640 for YOLOv8)
input_width, input_height = 640, 640
conf_threshold = 0.5
nms_threshold = 0.4

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare the image for the network
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (input_width, input_height), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward()  # Forward pass

    # Reshape output if necessary (depends on your model)
    # For YOLOv5/YOLOv8 ONNX, output shape: [1, N, 85]
    outputs = np.array(outputs)
    outputs = np.squeeze(outputs)

    # Lists to store results
    class_ids = []
    confidences = []
    boxes = []

    image_height, image_width = frame.shape[:2]

    for detection in outputs:
        confidence = detection[4]
        if confidence > conf_threshold:
            scores = detection[5:]
            class_id = np.argmax(scores)
            if scores[class_id] > conf_threshold and classes[class_id] == "human":
                w, h = int(detection[2] * image_width), int(detection[3] * image_height)
                x, y = int((detection[0] * image_width) - w / 2), int((detection[1] * image_height) - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Draw bounding boxes
    for i in indices:
        i = int(i)
        box = boxes[i]
        x, y, w, h = box
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Human Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
