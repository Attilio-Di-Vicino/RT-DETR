import os
import json
import numpy as np
import re
from glob import glob
from collections import defaultdict

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
                label = int(label_conf[0])  # Numerical label (0, 1, 2...)
                conf = float(label_conf[1])
                
                # Map from numeric label to category name
                category_name = categories.get(label + 1, "Unknown")  # Fallback to "Unknown" if no match
                
                bbox = parse_bbox(parts[1])  # Extract numerical values
                
                pred_annotations[image_name].append({
                    "category": category_name,  # Use category name
                    "confidence": conf,
                    "bbox": bbox
                })
    
    return pred_annotations


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


def prediction_statistics(pred_annotations):
    """Generate prediction statistics including total predictions, class counts, and predictions per class."""
    total_predictions = 0
    class_counts = defaultdict(int)

    for image_name, predictions in pred_annotations.items():
        total_predictions += len(predictions)
        for pred in predictions:
            class_counts[pred['category']] += 1
    
    return total_predictions, class_counts


if __name__ == "__main__":
    GT_JSON_PATH = "../PascalCOCO/valid/_annotations.coco.json"  
    PREDICTIONS_FOLDER = "../PascalCOCO/predictions"  
    
    gt_data = load_ground_truth(GT_JSON_PATH)
    pred_data = load_predictions(PREDICTIONS_FOLDER, mscoco_category2name)

    # Filter out predictions with "Unknown" category
    for image_name, predictions in pred_data.items():
        filtered_predictions = [prediction for prediction in predictions if prediction['category'] != 'Unknown']
        pred_data[image_name] = filtered_predictions

    # Calculate mAP
    mAP, ap_list, incorrect_labels, correct_labels = evaluate_mAP(gt_data, pred_data, np.arange(0.5, 1.0, 0.05))

    print(f"mAP: {mAP:.4f}")
    print(f"AP50: {ap_list[0]:.4f}, AP75: {ap_list[5]:.4f}")

    # Output incorrect and correct label counts
    if incorrect_labels or correct_labels:
        print("\nIncorrectly and Correctly predicted labels:")
        for label in set(incorrect_labels.keys()).union(set(correct_labels.keys())):
            correct_count = correct_labels.get(label, 0)
            incorrect_count = incorrect_labels.get(label, 0)
            print(f"Category: {label}, Correct Predictions: {correct_count}, Incorrect Predictions: {incorrect_count}")
    else:
        print("\nNo incorrect predictions.")

    # Output prediction statistics
    total_predictions, class_counts = prediction_statistics(pred_data)
    print(f"\nTotal Predictions: {total_predictions}")
    print(f"Number of Classes: {len(class_counts)}")
    print("Predictions per Class:")
    for category, count in class_counts.items():
        print(f"  {category}: {count}")