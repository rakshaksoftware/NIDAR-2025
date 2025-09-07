import pyrealsense2 as rs
import numpy as np
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
# Your existing imports and model setup
# from your_model_file import TRTEngineWrapper, preprocess, postprocess, draw_boxes

# === Model setup ===
# Replace with actual class labels, engine path etc.
CLASS_LABEL = "person"
INPUT_SHAPE = (640, 480)
ENGINE_PATH = "yolobuiltin.engine"

class TRTEngineWrapper:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.input_shape = self.engine.get_binding_shape(0)
        self.output_shape = self.engine.get_binding_shape(1)
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()

    def load_engine(self, engine_path):
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            return runtime.deserialize_cuda_engine(f.read())

    def allocate_buffers(self):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})

        return inputs, outputs, bindings, stream

    def infer(self, input_tensor):
        np.copyto(self.inputs[0]['host'], input_tensor.ravel())

        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.context.execute_async_v2(self.bindings, self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()

        return self.outputs[0]['host'].reshape(self.output_shape)

trt_engine = TRTEngineWrapper(ENGINE_PATH)

# === RealSense pipeline ===
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

        # Preprocess and inference
        input_tensor = preprocess(color_image)
        output = trt_engine.infer(input_tensor)
        boxes, scores = postprocess(output)

        # Draw detections
        image_with_boxes = draw_boxes(color_image, boxes, scores)

        # Display live detections
        cv2.imshow("Human Detection (Live)", image_with_boxes)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
