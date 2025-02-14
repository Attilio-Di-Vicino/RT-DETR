import os
import json
import numpy as np
import re
from glob import glob
from collections import defaultdict
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from src.data.coco.coco_eval import CocoEvaluator
import os
import json
import numpy as np
import re
from glob import glob
from collections import defaultdict
# from src.data.coco.coco_dataset import mscoco_category2label, mscoco_label2category

mscoco_category2name = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush'
}

mscoco_category2label = {k: i for i, k in enumerate(mscoco_category2name.keys())}
mscoco_label2category = {v: k for k, v in mscoco_category2label.items()}


def parse_bbox(tensor_str):
    numbers = re.findall(r"tensor\(([-\d.e]+)\)", tensor_str)
    return [float(num) for num in numbers] if numbers else []


def load_ground_truth(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    image_id_to_name = {img["id"]: img["file_name"] for img in data["images"]}
    gt_annotations = defaultdict(lambda: {"file_name": "", "annotations": []})
    for ann in data["annotations"]:
        image_name = image_id_to_name[ann["image_id"]]
        gt_annotations[image_name]["file_name"] = image_name
        gt_annotations[image_name]["annotations"].append({
            "category": ann["category_id"],
            "bbox": ann["bbox"]
        })
    return gt_annotations


def load_predictions(predictions_folder):
    pred_annotations = defaultdict(list)
    for txt_file in glob(os.path.join(predictions_folder, "*.txt")):
        image_name = os.path.basename(txt_file).replace(".txt", "")
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if "label:" in line and "bbox:" in line:
                parts = line.strip().split(", bbox: ")
                label_conf = parts[0].replace("label: ", "").split()
                try:
                    conf = float(label_conf[-1])
                    label_str = " ".join(label_conf[:-1])
                    category_id = mscoco_label2category.get(label_str, None)
                    if category_id is None:
                        continue
                    bbox = parse_bbox(parts[1])
                    pred_annotations[image_name].append({
                        "category_id": category_id,
                        "confidence": conf,
                        "bbox": bbox
                    })
                except ValueError:
                    continue
    return pred_annotations


def evaluate_model(gt_annotations, pred_annotations, iou_threshold=0.5):
    tp, fp, fn = 0, 0, 0
    for image_name, gt_data in gt_annotations.items():
        gt_boxes = gt_data["annotations"]
        preds = pred_annotations.get(image_name, [])
        matched = set()
        for pred in preds:
            best_iou, best_gt_idx = 0, -1
            for i, gt in enumerate(gt_boxes):
                iou = calculate_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou and pred["category_id"] == gt["category"]:
                    best_iou, best_gt_idx = iou, i
            if best_iou >= iou_threshold and best_gt_idx not in matched:
                tp += 1
                matched.add(best_gt_idx)
            else:
                fp += 1
        fn += len(gt_boxes) - len(matched)
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    return {"precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}


def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xa, ya = max(x1, x2), max(y1, y2)
    xb, yb = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    intersection = max(0, xb - xa) * max(0, yb - ya)
    union = (w1 * h1) + (w2 * h2) - intersection
    return intersection / union if union > 0 else 0


if __name__ == "__main__":
    GT_JSON_PATH = "../PascalCOCO/valid/_annotations.coco.json"
    PREDICTIONS_FOLDER = "../PascalCOCO/predictions"

    gt_annotations = load_ground_truth(GT_JSON_PATH)
    pred_annotations = load_predictions(PREDICTIONS_FOLDER)

    evaluation_results = evaluate_model(gt_annotations, pred_annotations)
    print(evaluation_results)
