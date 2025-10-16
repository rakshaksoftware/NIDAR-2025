import torch
import os
import cv2
import numpy as np
import time
from PIL import Image, ImageDraw
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor


def RT_DETR_v2_OBJ_DET_VIDEO(
    video_path,
    selected_classes,
    class_names,
    specific_class_color=None,
    default_color=(0, 255, 0),
    threshold=0.4,
    resize_to=(640, 640),  # 👈 resize to this resolution
    model=None,
    processor=None,
    device=None,
    show=True
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🟢 Using device: {device}")

    if model is None:
        print("🔄 Loading RT-DETRv2 model...")
        model = RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r18vd")
    if processor is None:
        processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd")

    model.to(device).eval()

    os.makedirs("result", exist_ok=True)
    input_filename = os.path.splitext(os.path.basename(video_path))[0]
    output_path = f"result/{input_filename}_RT-DETRv2_detected.mp4"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video file: {video_path}")
        return

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = resize_to
    out = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*'mp4v'), orig_fps, (width, height)
    )

    selected_class_indices = [class_names.index(cls) for cls in selected_classes]
    color_map = {cls: specific_class_color.get(cls, default_color)
                 if specific_class_color else default_color
                 for cls in selected_classes}

    print("🔵 Processing video... Press 'q' to stop.\n")

    frame_count = 0
    total_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for faster processing
        frame = cv2.resize(frame, resize_to)

        start_time = time.time()

        # Convert to PIL and preprocess
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        inputs = processor(images=pil_image, return_tensors="pt").to(device)

        with torch.inference_mode():
            outputs = model(**inputs)

        target_sizes = torch.tensor([pil_image.size[::-1]], device=device)
        detections = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=threshold
        )[0]

        boxes = detections["boxes"].cpu().numpy()
        scores = detections["scores"].cpu().numpy()
        labels = detections["labels"].cpu().numpy()

        draw = ImageDraw.Draw(pil_image)
        for box, score, label in zip(boxes, scores, labels):
            if label not in selected_class_indices:
                continue
            class_name = class_names[label]
            if class_name not in selected_classes:
                continue

            x1, y1, x2, y2 = box.astype(int)
            color = color_map[class_name]
            label_text = f"{class_name}: {score:.2f}"
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            text_bbox = draw.textbbox((0, 0), label_text)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            text_y = max(y1 - text_h, 0)
            draw.rectangle([x1, text_y, x1 + text_w, text_y + text_h], fill=color)
            draw.text((x1, text_y), label_text, fill="white")

        result_frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # FPS calculation
        end_time = time.time()
        frame_time = end_time - start_time
        total_time += frame_time
        frame_count += 1
        fps = 1 / frame_time

        cv2.putText(result_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        if show:
            cv2.imshow("RT-DETRv2 Object Detection", result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("🛑 Stopped by user.")
                break

        out.write(result_frame)

    avg_fps = frame_count / total_time if total_time > 0 else 0
    print(f"\n✅ Saved video: {output_path}")
    print(f"📈 Average FPS: {avg_fps:.2f}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()


coco_classes = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

specific_colors = {"person": (255, 0, 0)}  # Red for person
RT_DETR_v2_OBJ_DET_VIDEO(
    video_path="video.MP4",
    selected_classes=["person"],
    class_names=coco_classes,
    specific_class_color=specific_colors,
    threshold=0.5,
    resize_to=(640, 640),
    show=True
)
