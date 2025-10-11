import cv2
import numpy as np

model_path = "best.onnx"
net = cv2.dnn.readNetFromONNX(model_path)

# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

classes = ["human"]  # your model's class names
cap = cv2.VideoCapture(0)

# change to your model train size (common: 640)
IN_W, IN_H = 640, 640
CONF_THR = 0.5
NMS_THR = 0.4

# enable to see raw shapes for debugging
DEBUG = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    ih, iw = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (IN_W, IN_H), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward()

    outputs = np.array(outputs)
    if DEBUG:
        print("raw outputs shape:", outputs.shape)

    # squeeze leading singleton dims
    outputs = np.squeeze(outputs)
    # after squeeze, outputs could be:
    # - (N, 85)  typical YOLO (x,y,w,h,obj,cls0,...clsN)
    # - (M, 6)   [x1,y1,x2,y2,conf,class_id]
    # - (K, )    unexpected -> skip
    if outputs.ndim == 1:
        # nothing to process
        continue
    if outputs.ndim == 2 and outputs.shape[1] < 5:
        # unexpected format
        if DEBUG:
            print("unexpected detection vector length:", outputs.shape)
        continue

    boxes = []
    confidences = []
    class_ids = []

    # iterate rows (each detection)
    for det in outputs:
        det = np.asarray(det).flatten()
        # case A: YOLO-like (cx,cy,w,h,obj_conf, classes...)
        if det.size >= 6:
            # assume first 4 are bbox (normalized), 4 = objectness/conf
            cx, cy, w, h = det[0], det[1], det[2], det[3]
            obj_conf = float(det[4])
            scores = det[5:]
            if DEBUG:
                print("det size>=6 -> obj_conf", obj_conf, "scores_len", scores.size)

            # skip if objectness too low
            if obj_conf < CONF_THR:
                continue

            if scores.size == 0:
                # no class scores provided; treat objectness as class confidence and class 0
                class_conf = obj_conf
                cls_id = 0
            else:
                cls_id = int(np.argmax(scores))
                class_conf = float(scores[cls_id])

            # final score threshold
            if class_conf < CONF_THR:
                continue

            # only accept if class name is "human"
            if cls_id >= len(classes) or classes[cls_id] != "human":
                continue

            # convert normalized center->pixel xywh
            bw = int(w * iw)
            bh = int(h * ih)
            bx = int(cx * iw - bw / 2)
            by = int(cy * ih - bh / 2)

        # case B: some models output [x1,y1,x2,y2,conf,class_id]
        elif det.size == 6:
            x1, y1, x2, y2, conf, cls = det
            bw = int(x2 - x1)
            bh = int(y2 - y1)
            bx = int(x1)
            by = int(y1)
            class_conf = float(conf)
            cls_id = int(cls)
            if class_conf < CONF_THR or cls_id >= len(classes) or classes[cls_id] != "human":
                continue

        else:
            # fallback: skip unexpected shapes
            if DEBUG:
                print("skipping detection with size", det.size)
            continue

        # clamp box
        bx = max(0, min(bx, iw - 1))
        by = max(0, min(by, ih - 1))
        bw = max(1, min(bw, iw - bx))
        bh = max(1, min(bh, ih - by))

        boxes.append([bx, by, bw, bh])
        confidences.append(float(class_conf))
        class_ids.append(int(cls_id))

    # perform NMS if there are boxes
    if len(boxes) > 0:
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THR, NMS_THR)
        if len(idxs) > 0:
            # idxs may be nested arrays depending on OpenCV version
            idxs = np.array(idxs).flatten()
            for i in idxs:
                x, y, w, h = boxes[int(i)]
                label = f"{classes[class_ids[int(i)]]}: {confidences[int(i)]:.2f}"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Human Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
