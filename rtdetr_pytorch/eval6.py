from pycocotools.coco import COCO
import os
import json
import numpy as np
import re
from glob import glob
from collections import defaultdict
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.coco.coco_eval import CocoEvaluator

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

mscoco_category2label = {k: i for i, k in enumerate(mscoco_category2name.keys())}
mscoco_label2category = {v: k for k, v in mscoco_category2label.items()}

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

def parse_bbox(tensor_str):
    """Convert a PyTorch tensor string to a list of floats."""
    numbers = re.findall(r"tensor\(([\d.e-]+)", tensor_str)
    return [float(num) for num in numbers] if numbers else []

def load_predictions(predictions_folder):
    """Load model predictions from RT-DETR output text files, mapping numerical labels to category names."""
    pred_annotations = defaultdict(list)

    for txt_file in glob(os.path.join(predictions_folder, "*.txt")):
        image_name = os.path.basename(txt_file).replace(".txt", "")

        with open(txt_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if "label:" in line and "bbox:" in line:
                parts = line.strip().split(", bbox: ")
                label_conf = parts[0].replace("label: ", "").split()  # Separiamo label e confidenza

                # La confidenza è l'ultimo elemento della lista dopo 'label' e deve essere un numero
                try:
                    conf = float(label_conf[-1])  # Confidenza
                    label = int(label_conf[0])  # INTERPRETIAMO COME INTERO
                except ValueError:
                    print(f"Errore nella conversione di label/confidenza per {image_name}: {label_conf}")
                    continue  # Salta questa predizione se la confidenza non è un numero valido

                # Controllo con mappatura numerica
                if label in model_to_dataset_mapping:
                    label = model_to_dataset_mapping[label]
                else:
                    label = -1

                bbox = parse_bbox(parts[1])  # Estrai i valori numerici per la bounding box

                pred_annotations[image_name].append({
                    "category": label,  # Usa il numero della categoria
                    "confidence": conf,
                    "bbox": bbox
                })

    return pred_annotations

def filter_predictions(predictions):
    """Filtra le predizioni eliminando quelle con category -1"""
    filtered_predictions = {}

    for image_name, annotations in predictions.items():
        valid_annotations = [ann for ann in annotations if ann["category"] != -1]
        if valid_annotations:
            filtered_predictions[image_name] = valid_annotations

    return filtered_predictions

if __name__ == "__main__":

    GT_JSON_PATH = "../PascalCOCO/valid/_annotations.coco.json"

    # 1. Carico il ground truth
    coco_gt = COCO(GT_JSON_PATH)
    # print(coco_gt)

    # 2. Creo l'evaluator
    evaluator = CocoEvaluator(coco_gt, iou_types=["bbox", "segm"])

    PREDICTIONS_FOLDER = "../PascalCOCO/predictions" 
    predictions = load_predictions(PREDICTIONS_FOLDER)
    filtered_predictions = filter_predictions(predictions)

    # # 4. Aggiorno l'evaluator
    evaluator.update(predictions)

    # 5. Sincronizzo tra processi (se multi-GPU)
    # evaluator.synchronize_between_processes()

    # 6. Accumulo i risultati
    evaluator.accumulate()

    # 7. Stampo i risultati
    evaluator.summarize()
