[link](https://www.mdpi.com/2504-446X/9/8/514?utm_source=chatgpt.com)


### This is the paragraph from the above website##
YOLO is widely recognized as a state-of-the-art object detection framework for real-time applications. Although newer versions like YOLOv9 [29], YOLOv10 [30], YOLOv11 [31], and YOLOv12 [32] have been released recently, in this study, we selected YOLOv5 for several reasons. First, YOLOv5 provides a mature and stable implementation with extensive documentation and an active developer community, facilitating the integration of custom modules such as additional detection heads, feature fusion layers, and attention mechanisms. Second, YOLOv5 has established compatibility with NVIDIA TensorRT and ONNX export pipelines, which is essential for efficient deployment on embedded systems like the Jetson Nano. Third, preliminary experiments conducted as part of this work demonstrated that applying the same architectural modifications to YOLOv8 yielded comparable detection accuracy (mAP@50 and mAP@50:95) to YOLOv5 on our aerial Search and Rescue dataset while requiring a higher computational burden. YOLOv5 allows for architectural definition, supporting changes in model depth, width, number of classes, and specific modules in the backbone, neck, and head. YOLOv5s, the smallest full-scale version, was selected for its balance between accuracy and efficiency.

## Datasets##
Since more than half of the images in the VisDrone dataset contain densely crowded clusters of people, it may not be ideal to train a model on this dataset for general Search and Rescue especially in Nidar.

And in the Kaggle Dataset, We dont have enough images that are from the human top view,this may result in failure of detecting the human directly below the drone.So we have to add some more images ,Some Images of this type,have been merged with the present Kaggle Datset,And some more yet to be added.
