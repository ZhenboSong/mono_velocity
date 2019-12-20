# Crop Velocity



## Tusimple Dataset

Specific cars are annoted by bounding boxes in the last frame of each sequence for both training and testing dataset.
Relative position and velocity are given for training in json file format. 

### data preparation
For each sequence, a pair of images is picked up along with their corresponding ground truth annotation.
All relative directories of data-pairs are saved in the text file line by line. 


## Code
###  cropdata.py
To train in multi-batches, the number of cars are unified to 4 for every sequence.
The supplementary input information is pre-calculated in this dataloader:   

```
self.args.cam_fx / (t_bbox[2] - t_bbox[0]),
self.args.cam_fy / (t_bbox[3] - t_bbox[1],
self.args.cam_fy / (t_bbox[3] - t_bbox[1]),
(t_bbox[0] - self.args.cam_cx) / self.args.cam_fx,
(t_bbox[1] - self.args.cam_cy) / self.args.cam_fy,
(t_bbox[2] - self.args.cam_cx) / self.args.cam_fx,
(t_bbox[3] - self.args.cam_cy) / self.args.cam_fy
```

###  train_crop_velocity.py
Modify the arguments, then train
```
python train_crop_velocity.py
```

###  test_crop_velocity.py
The testing results are saved to a json file for evaluation.
```
python test_crop_velocity.py
```

###  evaluate.py
Modify the prediction file directory and its corresponding groud truth directory, then evalute
```
python evaluate.py
```


