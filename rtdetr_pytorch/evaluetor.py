import os
import json
import numpy as np
import re
from glob import glob
from collections import defaultdict

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


def parse_bbox(tensor_str):
    """Convert a PyTorch tensor string to a list of floats."""
    numbers = re.findall(r"tensor\(([\d.e-]+)", tensor_str)
    return [float(num) for num in numbers] if numbers else []


def load_ground_truth(json_path):
    """Load ground truth annotations from a COCO-style JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Create a mapping of image_id -> file_name
    image_id_to_name = {img["id"]: img["file_name"] for img in data["images"]}

    gt_annotations = defaultdict(lambda: {"file_name": "", "annotations": []})
    categories = {cat["id"]: cat["name"] for cat in data["categories"]}
    
    for ann in data["annotations"]:
        image_id = ann["image_id"]
        image_name = image_id_to_name.get(image_id, None)  # Get the filename
        
        if image_name:  # Ensure the mapping exists
            gt_annotations[image_name]["file_name"] = image_name  # Store file name
            gt_annotations[image_name]["annotations"].append({
                "category": categories[ann["category_id"]],
                "bbox": ann["bbox"]  # Format: [x, y, width, height]
            })
    
    return gt_annotations

def load_predictions(predictions_folder, categories):
    """Load model predictions from RT-DETR output text files, mapping numerical labels to category names."""
    pred_annotations = defaultdict(list)
    
    for txt_file in glob(os.path.join(predictions_folder, "*.txt")):
        image_name = os.path.basename(txt_file).replace(".txt", "")
        
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            if "label:" in line and "bbox:" in line:
                parts = line.strip().split(", bbox: ")
                label_conf = parts[0].replace("label: ", "").split()
                label = int(label_conf[0])  # Numerica (0, 1, 2...)
                conf = float(label_conf[1])
                
                # Mappatura da numero a categoria stringa
                category_name = categories.get(label + 1, "Unknown")  # Fallback a "Unknown" se non c'è match
                
                bbox = parse_bbox(parts[1])  # Estrai i valori numerici
                
                pred_annotations[image_name].append({
                    "category": category_name,  # Usa il nome della categoria
                    "confidence": conf,
                    "bbox": bbox
                })
    
    return pred_annotations


# def load_predictions(predictions_folder):
#     """Load model predictions from RT-DETR output text files."""
#     pred_annotations = defaultdict(list)
    
#     for txt_file in glob(os.path.join(predictions_folder, "*.txt")):
#         image_name = os.path.basename(txt_file).replace(".txt", "")
        
#         with open(txt_file, 'r') as f:
#             lines = f.readlines()
        
#         for line in lines:
#             if "label:" in line and "bbox:" in line:
#                 parts = line.strip().split(", bbox: ")
#                 label_conf = parts[0].replace("label: ", "").split()
#                 label, conf = int(label_conf[0]), float(label_conf[1])
                
#                 bbox = parse_bbox(parts[1])  # Extract numerical values
                
#                 pred_annotations[image_name].append({
#                     "category": label,
#                     "confidence": conf,
#                     "bbox": bbox
#                 })
    
#     return pred_annotations


def calculate_iou(box1, box2):
    """Calculate IoU (Intersection over Union) between two bounding boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xa, ya = max(x1, x2), max(y1, y2)
    xb, yb = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    
    intersection = max(0, xb - xa) * max(0, yb - ya)
    union = (w1 * h1) + (w2 * h2) - intersection
    
    return intersection / union if union > 0 else 0


def evaluate_model(gt_annotations, pred_annotations, iou_threshold=0.5):
    """Evaluate the model using IoU, Precision, Recall, and mAP."""
    tp, fp, fn = 0, 0, 0
    
    for image_name, gt_data in gt_annotations.items():  # Iterate over filenames
        preds = pred_annotations.get(image_name, [])
        gt_boxes = gt_data["annotations"]  # Extract GT bounding boxes

        matched = set()
        
        for pred in preds:
            best_iou = 0
            best_gt_idx = -1
            
            for i, gt in enumerate(gt_boxes):
                iou = calculate_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
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

def compute_ap(recall, precision):
    """Compute Average Precision (AP) given recall and precision values."""
    recall = np.concatenate(([0.], recall, [1.]))
    precision = np.concatenate(([0.], precision, [0.]))

    # Compute precision envelope
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])

    # Find recall change points
    indices = np.where(recall[1:] != recall[:-1])[0]

    # Sum up areas under curve
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
    return ap

def evaluate_mAP(gt_annotations, pred_annotations, iou_thresholds=np.arange(0.5, 1.0, 0.05)):
    """Evaluate mAP (Mean Average Precision) at different IoU thresholds."""
    aps = []
    
    for iou_threshold in iou_thresholds:
        results = evaluate_model(gt_annotations, pred_annotations, iou_threshold)
        precision, recall = results['precision'], results['recall']
        aps.append(compute_ap(np.array([recall]), np.array([precision])))

    return np.mean(aps), aps


if __name__ == "__main__":
    GT_JSON_PATH = "../PascalCOCO/valid/_annotations.coco.json"  
    PREDICTIONS_FOLDER = "../PascalCOCO/predictions"  
    
    gt_data = load_ground_truth(GT_JSON_PATH)
    pred_data = load_predictions(PREDICTIONS_FOLDER, mscoco_category2name)

    # Itera su tutte le immagini e le relative predizioni
    for image_name, predictions in pred_data.items():
        # Filtra le predizioni per l'immagine, mantenendo solo quelle con categoria diversa da "Unknown"
        filtered_predictions = [prediction for prediction in predictions if prediction['category'] != 'Unknown']
        
        # Sostituisci le predizioni originali con quelle filtrate
        pred_data[image_name] = filtered_predictions

    # Ora pred_data contiene solo le predizioni con categorie valide
    # Puoi stampare il risultato filtrato per vedere le modifiche
    for image_name, predictions in pred_data.items():
        print(f"Predizioni per l'immagine: {image_name}")
        for prediction in predictions:
            print(f"Categoria: {prediction['category']} - Confidenza: {prediction['confidence']}")


    
    # results = evaluate_model(gt_data, pred_data, iou_threshold=0.0)
    mAP, ap_list = evaluate_mAP(gt_data, pred_data, np.arange(0.5, 1.0, 0.05))

    print(f"mAP: {mAP:.4f}")
    print(f"AP50: {ap_list[0]:.4f}, AP75: {ap_list[5]:.4f}")

    # print("Evaluation Results:")
    # print(f"Precision: {results['precision']:.4f}")
    # print(f"Recall: {results['recall']:.4f}")
    # print(f"True Positives: {results['tp']}")
    # print(f"False Positives: {results['fp']}")
    # print(f"False Negatives: {results['fn']}")