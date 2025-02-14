import os
import json
from glob import glob
from collections import defaultdict
import ast

def parse_bbox(bbox_str):
    """Parsa una stringa di bbox del tipo '[x_min, y_min, width, height]' in lista di float."""
    try:
        bbox = ast.literal_eval(bbox_str)  # Usa literal_eval per trasformare la stringa in lista
        if len(bbox) == 4:
            return [float(x) for x in bbox]
        else:
            raise ValueError(f"Formato bbox errato: {bbox_str}")
    except Exception as e:
        print(f"Errore nel parsing del bbox: {e}")
        return []

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

if __name__ == "__main__":
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
    
    PREDICTIONS_FOLDER = "../PascalCOCO/predictions"
    pred_annotations = load_predictions(PREDICTIONS_FOLDER, mscoco_category2name)
    
    print(pred_annotations)  # Aggiungi per vedere il risultato delle predizioni
