import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2

# --- Constants ---
ENGINE_PATH = "best.engine"
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
CONFIDENCE_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CLASSES = ["human"]

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

# --- Load TensorRT engine ---
def load_engine(engine_file_path):
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

engine = load_engine(ENGINE_PATH)
context = engine.create_execution_context()

# --- Allocate buffers ---
def allocate_buffers(engine):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate device memory
        device_mem = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append(device_mem)
        else:
            outputs.append(device_mem)
    return inputs, outputs, bindings, stream

inputs, outputs, bindings, stream = allocate_buffers(engine)

# --- Preprocess image ---
def preprocess(image):
    row, col, _ = image.shape
    max_dim = max(row, col)
    square_image = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
    square_image[0:row, 0:col] = image
    blob = cv2.resize(square_image, (INPUT_WIDTH, INPUT_HEIGHT))
    blob = blob.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))  # HWC -> CHW
    blob = np.expand_dims(blob, axis=0)   # Add batch dimension
    return blob, max_dim

# --- Postprocess ---
def postprocess(predictions, max_dim, image):
    # predictions: (num_boxes, 85) for YOLOv8 (4 bbox + 1 obj + 80 classes)
    boxes, confidences, class_ids = [], [], []

    x_factor = max_dim / INPUT_WIDTH
    y_factor = max_dim / INPUT_HEIGHT

    for pred in predictions:
        confidence = np.max(pred[4:])
        if confidence > CONFIDENCE_THRESHOLD:
            class_id = np.argmax(pred[4:])
            if class_id == 0:  # person/human
                cx, cy, w, h = pred[:4]
                left = int((cx - w / 2) * x_factor)
                top = int((cy - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                boxes.append([left, top, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD)
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"Human: {confidences[i]:.2f}"
            cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

    return image

# --- Main Loop ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        blob, scale_factor = preprocess(frame)
        blob = np.ascontiguousarray(blob)

        # Copy input to device
        cuda.memcpy_htod_async(inputs[0], blob, stream)
        # Run inference
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # Copy output back to host
        output_shape = (engine.max_batch_size, np.prod(engine.get_binding_shape(1)))
        host_output = np.empty(output_shape, dtype=np.float32)
        cuda.memcpy_dtoh_async(host_output, outputs[0], stream)
        stream.synchronize()

        predictions = host_output.reshape(-1, host_output.shape[-1])
        result_image = postprocess(predictions, scale_factor, frame)
        cv2.imshow("Person Detection TRT", result_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
