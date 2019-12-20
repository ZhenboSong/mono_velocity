# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Miscellaneous utility functions
"""

import torch
import numpy as np
import json


def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def convert_to_roi_format(my_boxes, spatial_rate=1.0/32.0):
    bbox_obj = my_boxes[:, 3]
    bbox = []
    for i in bbox_obj:
        bbox.append(i)
    boxes = spatial_rate * torch.from_numpy(np.array(bbox).astype(np.float32))

    boxes_id = torch.from_numpy(np.zeros(len(boxes)).astype(np.int32))

    velocity = torch.from_numpy(np.array(my_boxes[:, 0:3]).astype(np.float32))

    return boxes, boxes_id, velocity


def convert_to_flow_format(my_boxes, spatial_rate=1.0/32.0):
    bbox = my_boxes[:, 9:13]
    boxes = spatial_rate * torch.from_numpy(np.array(bbox).astype(np.float32))

    boxes_id = torch.from_numpy(np.zeros(len(boxes)).astype(np.int32))

    location = torch.from_numpy(np.array(my_boxes[:, 0:3]).astype(np.float32))

    obj_size = torch.from_numpy(np.array(my_boxes[:, 3:6]).astype(np.float32))

    velocity = torch.from_numpy(np.array(my_boxes[:, 6:9]).astype(np.float32))

    r_and_t = torch.from_numpy(np.array(my_boxes[:, 13:]).astype(np.float32))

    return boxes, boxes_id, location, obj_size, velocity, r_and_t


def convert_json(json_file):
    with open(json_file, 'r') as load_f:
        json_dict = json.load(load_f)
    data_num = len(json_dict)

    bboxes = []
    poses = []
    veloes = []
    max_pos = 0
    max_pos_id = 0
    for i in range(data_num):
        bbox = json_dict[i]['bbox']
        position = json_dict[i]['position']
        velocity = json_dict[i]['velocity']
        veloes.append(velocity)
        poses.append(position)
        bboxes.append([bbox['left'], bbox['top'], bbox['right'], bbox['bottom']])
        if position[0] > max_pos:
            max_pos_id = i
            max_pos = position[0]

    if data_num == 1:
        bboxes = bboxes * 4
        poses = poses * 4
        veloes = veloes * 4
    elif data_num == 2:
        bboxes = bboxes * 2
        poses = poses * 2
        veloes = veloes * 2
    elif data_num == 3:
        bbox = json_dict[max_pos_id]['bbox']
        position = json_dict[max_pos_id]['position']
        velocity = json_dict[max_pos_id]['velocity']
        veloes.append(velocity)
        poses.append(position)
        bboxes.append([bbox['left'], bbox['top'], bbox['right'], bbox['bottom']])

    boxes = torch.from_numpy(np.array(bboxes).astype(np.float32))
    location = torch.from_numpy(np.array(poses).astype(np.float32))
    velocity = torch.from_numpy(np.array(veloes).astype(np.float32))

    return boxes, location, velocity


def convert_json_eval(json_file):
    with open(json_file, 'r') as load_f:
        json_dict = json.load(load_f)
    data_num = len(json_dict)

    bboxes = []
    poses = []
    veloes = []

    for i in range(data_num):
        bbox = json_dict[i]['bbox']
        position = [0, 0]
        velocity = [0, 0]
        veloes.append(velocity)
        poses.append(position)
        bboxes.append([bbox['left'], bbox['top'], bbox['right'], bbox['bottom']])

    if data_num == 1:
        bboxes = bboxes * 4
        poses = poses * 4
        veloes = veloes * 4
    elif data_num == 2:
        bboxes = bboxes * 2
        poses = poses * 2
        veloes = veloes * 2
    elif data_num == 3:
        bbox = json_dict[0]['bbox']
        position = [0, 0]
        velocity = [0, 0]
        veloes.append(velocity)
        poses.append(position)
        bboxes.append([bbox['left'], bbox['top'], bbox['right'], bbox['bottom']])

    boxes = torch.from_numpy(np.array(bboxes).astype(np.float32))
    location = torch.from_numpy(np.array(poses).astype(np.float32))
    velocity = torch.from_numpy(np.array(veloes).astype(np.float32))

    return boxes, location, velocity, data_num
