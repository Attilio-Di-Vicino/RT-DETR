# from pycocotools.coco import COCO
import os
import json
import numpy as np
import re
from glob import glob
from collections import defaultdict
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util import model_to_dataset_mapping, mscoco_category2color, mscoco_category2label, mscoco_label2category

def convert_to_xywh(boxes):
    """Convert a bounding box from [xmin, ymin, xmax, ymax] to [x, y, width, height]."""
    xmin, ymin, xmax, ymax = boxes
    return [xmin, ymin, xmax - xmin, ymax - ymin]

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
    return gt_annotations, categories

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
                # Let's separate the label and the confidence
                label_conf = parts[0].replace("label: ", "").split()
                try:
                    label_num = int(label_conf[0])  # The first value is the category number (e.g. 0)
                    conf = float(label_conf[1])    # The second value is the confidence (e.g. 0.91)
                except ValueError as e:
                    print(f"Error converting label/confidence for {image_name}: {label_conf} | Error: {e}")
                    continue 
                # Get category name using dictionary
                label_name = mscoco_category2label.get(label_num, "unknown")  # Returns 'unknown' if not found
                # Map the numeric label with the mapping 'model_to_dataset_mapping'
                if label_num in model_to_dataset_mapping:
                    label_num = model_to_dataset_mapping[label_num]
                else:
                    label_num = - 1
                bbox = parse_bbox(parts[1])
                # Convert bboxes from [xmin, ymin, xmax, ymax] to [x, y, width, height] to match with Pascal labels
                bbox_xywh = convert_to_xywh(bbox)
                pred_annotations[image_name].append({
                    "category": label_num,  
                    "category_name": label_name, 
                    "confidence": conf, 
                    "bbox": bbox_xywh 
                })
    return pred_annotations

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
    incorrect_labels = defaultdict(int)  # Keep track of incorrect predictions
    correct_labels = defaultdict(int)  # Keep track of correct predictions
    for image_name, gt_data in gt_annotations.items():  # Iterate over filenames
        preds = pred_annotations.get(image_name, [])
        gt_boxes = gt_data["annotations"]  # Extract GT bounding boxes
        matched = set()
        for pred in preds:
            # Filter out predictions with "Unknown" category
            if pred['category'] == 'Unknown':
                continue
            # Check if the predicted category is correct
            correct_category = any(gt['category'] == pred['category'] for gt in gt_boxes)
            if correct_category:
                correct_labels[pred['category']] += 1
            else:
                incorrect_labels[pred['category']] += 1
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
    return {"precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn, "incorrect_labels": incorrect_labels, "correct_labels": correct_labels}

def evaluate_mAP(gt_annotations, pred_annotations, iou_thresholds=np.arange(0.5, 1.0, 0.05)):
    """Evaluate mAP (Mean Average Precision) at different IoU thresholds."""
    aps = []
    incorrect_labels_total = defaultdict(int)
    correct_labels_total = defaultdict(int)
    for iou_threshold in iou_thresholds:
        results = evaluate_model(gt_annotations, pred_annotations, iou_threshold)
        precision, recall = results['precision'], results['recall']
        aps.append(compute_ap(np.array([recall]), np.array([precision])))
        # Aggregate incorrect and correct label predictions
        for label, count in results["incorrect_labels"].items():
            incorrect_labels_total[label] += count
        for label, count in results["correct_labels"].items():
            correct_labels_total[label] += count
    return np.mean(aps), aps, incorrect_labels_total, correct_labels_total

def filter_predictions(predictions):
    """Filtra le predizioni eliminando quelle con category -1"""
    filtered_predictions = {}
    for image_name, annotations in predictions.items():
        valid_annotations = [ann for ann in annotations if ann["category"] != -1]
        if valid_annotations:
            filtered_predictions[image_name] = valid_annotations
    return filtered_predictions

if __name__ == "__main__":
    GT_JSON_PATH = "../../PascalCOCO/valid/_annotations.coco.json"
    PREDICTIONS_FOLDER = "../../PascalCOCO/predictions"  
    gt_data, category_pascal = load_ground_truth(GT_JSON_PATH)
    predictions = load_predictions(PREDICTIONS_FOLDER)
    filtered_predictions = filter_predictions(predictions)
    mAP, ap_list, incorrect_labels, correct_labels = evaluate_mAP(gt_data, predictions, np.arange(0.5, 1.0, 0.05))
    print(f"mAP: {mAP*100:.4f}")
    print(f"AP50: {ap_list[0]*100:.4f}, AP75: {ap_list[5]*100:.4f}")
