mscoco_category2name = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat',
    10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
    18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
    28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball',
    38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
    46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich',
    55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
    64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote',
    76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book',
    85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair dryer', 90: 'toothbrush'
}

# Mapping between Pascal and output of RT-DETR
model_to_dataset_mapping = {
    4: 1, # aeroplane -> airplane
    1: 2, # bicycle 
    14: 3, # bird
    8: 4, # boat
    39: 5, # bottle
    5: 6, # bus
    2: 7, # car
    15: 8, # cat
    56: 9, # chair
    19: 10, # cow
    60: 11, # diningtable -> dining table
    16: 12, # dog
    17: 13, # horse
    3: 14, # motorbike -> motorcycle
    0: 15, # person
    58: 16, # pottedplant -> potted plant
    18: 17, # sheep
    57: 18, # sofa -> couch
    6: 19, # train
    62: 20, # tvmonitor -> tv
    63: 20, # tvmonitor -> laptop
}

# Different colore for each classes (by GPT)
mscoco_category2color = {
    1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255), 4: (255, 255, 0), 5: (255, 165, 0), 6: (128, 0, 128),
    7: (0, 255, 255), 8: (255, 105, 180), 9: (75, 0, 130), 10: (255, 255, 255), 11: (255, 69, 0), 13: (139, 69, 19), 
    14: (255, 223, 0), 15: (160, 82, 45), 16: (173, 216, 230), 17: (128, 128, 128), 18: (165, 42, 42), 19: (102, 51, 0),
    20: (153, 255, 153), 21: (139, 0, 0), 22: (255, 228, 181), 23: (0, 0, 0), 24: (255, 255, 153), 25: (218, 165, 32),
    27: (0, 128, 0), 28: (0, 0, 128), 31: (192, 192, 192), 32: (128, 0, 0), 33: (0, 139, 139), 34: (255, 20, 147),
    35: (70, 130, 180), 36: (72, 61, 139), 37: (255, 140, 0), 38: (240, 230, 140), 39: (160, 160, 160), 40: (222, 184, 135),
    41: (255, 99, 71), 42: (128, 128, 0), 43: (85, 107, 47), 44: (210, 180, 140), 46: (205, 133, 63), 47: (255, 239, 213),
    48: (192, 192, 192), 49: (169, 169, 169), 50: (128, 128, 128), 51: (112, 128, 144), 52: (255, 255, 0), 53: (255, 0, 255),
    54: (255, 182, 193), 55: (255, 165, 0), 56: (0, 128, 0), 57: (255, 127, 80), 58: (255, 69, 0), 59: (139, 69, 19),
    60: (255, 215, 0), 61: (139, 0, 0), 62: (165, 42, 42), 63: (128, 0, 0), 64: (85, 107, 47), 65: (255, 218, 185),
    67: (70, 130, 180), 70: (105, 105, 105), 72: (173, 216, 230), 73: (0, 255, 255), 74: (240, 128, 128),
    75: (100, 149, 237), 76: (184, 134, 11), 77: (255, 140, 0), 78: (46, 139, 87), 79: (160, 82, 45), 80: (205, 133, 63),
    81: (32, 178, 170), 82: (0, 0, 128), 84: (139, 69, 19), 85: (255, 215, 0), 86: (255, 182, 193), 87: (255, 105, 180),
    88: (255, 20, 147), 89: (70, 130, 180), 90: (128, 0, 128)
}

mscoco_category2label = {k: i for i, k in enumerate(mscoco_category2name.keys())}
mscoco_label2category = {v: k for k, v in mscoco_category2label.items()}