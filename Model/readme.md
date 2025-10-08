## Progress:

### Date: 7 Oct
1. Not using the visdrone dataset:
   1. Bcoz some images have many overlapping bounding boxes and too many people in a single image
   2. The bounding box in a few images contains more then one people
2. Working on refining the dataset from google, kaggle- mostly based on drone based images captured in open areas
3. Will train with yolov5, yolov8, yolov11 - and see which works best for human detection

### Date:28 Sep
1. Researched on SRO based human detection dataset, found very good datasets like:

   a. **C2A dataset**: Contains various **disaster scenes**(floods, fire, landslides, accidents,etc) with humans in **various position** (bent, kneeling, lying, sitting, upright) suitable for rescue based                 operations.
   Links: https://github.com/Ragib-Amin-Nihal/C2A/blob/main/README.md , https://arxiv.org/pdf/2408.04922

   b. **Visdronee**: Contains drone based images of 11 different classes( including people and pedestrains)

   c. **CCTV based images**: https://www.kaggle.com/datasets/constantinwerner/human-detection-dataset

2. Some new models found:
   
    A. Visibility-Enhanced DINO (VE-DINO) Model : https://www.mdpi.com/2624-6511/8/1/12
