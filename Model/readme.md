## Progress:
### Date: 11 Oct
1. Training on servers is now possible and would use this for upcoming model training.
2. Did the live detection inside the casde lab and found that the model is not properly able to find the humans and the possible reasons could be:
     1. the room's environment dosn't matches with the images which we trained it on
     2. the training data doesn't have good enough dataset
     3. the model is very heavy for jetson orion
3. Solution:
     1. accumulate a very good dataset which would accurately represent the original scenario
     2. train on yolov5 and yolov8 's smaller model for easy computation

### Date: 9 Oct
1. Trained yolo11s on merged dataset and achieved precision 0f 0.68 and recall of 0.38 with
   1. map: 0.2217
   2. map50: 0.4482
   3. map75: 0.2003
2. model performance has decreased due to varying dataset
3. but the model is able to predict well on data set which it hasn't seen before which was not the case with 11n
4. will try to train it on different(selective) dataset so that it generalises well and will try older version of yolo too.
### Date: 8 Oct
1. Trained yolo11n on a dataset and achieved precision 0f 0.9376 and recall of 0.8838 with
   1. map: 0.6209
   2. map50: 0.9381
   3. map75: 0.7025
2. Now plan to merge various data sets and train it on yolov11 s,m,l and if possible then x
### Date: 7 Oct
1. Not using the visdrone dataset:
   1. Bcoz some images have many overlapping bounding boxes and too many people in a single image
   2. The bounding box in a few images contains more then one people
2. Working on refining the dataset from google, kaggle- mostly based on drone based images captured in open areas
3. Will train with yolov5, yolov8, yolov11 - and see which works best for human detection

### Date:28 Sep
1. Researched on SRO based human detection dataset, found very good datasets like:
    1. **C2A dataset**: Contains various **disaster scenes**(floods, fire, landslides, accidents,etc) with humans in **various position** (bent, kneeling, lying, sitting, upright) suitable for rescue based                 operations.
   Links: https://github.com/Ragib-Amin-Nihal/C2A/blob/main/README.md , https://arxiv.org/pdf/2408.04922
    2. **Visdronee**: Contains drone based images of 11 different classes( including people and pedestrains)
    3. **CCTV based images**: https://www.kaggle.com/datasets/constantinwerner/human-detection-dataset

3. Some new models found:
    1. Visibility-Enhanced DINO (VE-DINO) Model : https://www.mdpi.com/2624-6511/8/1/12
