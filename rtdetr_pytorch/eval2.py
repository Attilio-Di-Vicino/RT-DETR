import os
import json
import re
from glob import glob
from collections import defaultdict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

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

category_name_to_id = {v: k for k, v in mscoco_category2name.items()}

import re

def parse_bbox_from_tensor_block(lines):
    """ Estrae i valori bbox da più righe di tensori """
    tensor_str = " ".join(lines).replace('\n', ' ')
    bbox_values = re.findall(r"tensor\(([\d\.\-]+)", tensor_str)
    if len(bbox_values) != 4:
        raise ValueError(f"Formato bbox non valido: {lines}")
    return [float(v) for v in bbox_values]

def read_predictions(predictions_folder):
    pred_annotations = defaultdict(list)
    txt_files = glob(os.path.join(predictions_folder, "*.txt"))

    for txt_file in txt_files:
        image_name = os.path.splitext(os.path.basename(txt_file))[0]
        with open(txt_file, 'r') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line or "Execution time" in line:
                i += 1
                continue

            parts = line.split()
            if "label:" in parts:
                try:
                    class_id = int(parts[1])
                    confidence = float(parts[2].replace(',', ''))

                    # Raccogliere il blocco bbox che si estende su più righe
                    if "bbox:" in parts:
                        bbox_lines = [line]
                        i += 1
                        while i < len(lines) and "tensor(" in lines[i]:
                            bbox_lines.append(lines[i].strip())
                            i += 1

                        bbox = parse_bbox_from_tensor_block(bbox_lines)

                        x_min, y_min, x_max, y_max = bbox
                        width = x_max - x_min
                        height = y_max - y_min

                        pred_annotations[image_name].append({
                            "category": class_id,
                            "bbox": [x_min, y_min, width, height],
                            "confidence": confidence
                        })
                    else:
                        i += 1
                except Exception as e:
                    print(f"Errore parsing {txt_file} (linea {i}): {e}")
                    i += 1
            else:
                i += 1

    return pred_annotations

def convert_predictions_to_coco_format(pred_annotations, gt_json_path, category_name_to_id):
    with open(gt_json_path, 'r') as f:
        data = json.load(f)

    file_to_image_id = {img["file_name"].split(".")[0]: img["id"] for img in data["images"]}

    coco_predictions = []

    for image_name, predictions in pred_annotations.items():
        image_id = file_to_image_id.get(image_name)
        if image_id is None:
            continue

        for pred in predictions:
            category_id = category_name_to_id.get(pred["category"])
            if category_id is None:
                continue

            coco_predictions.append({
                "image_id": image_id,
                "category_id": category_id,
                "bbox": pred["bbox"],
                "score": pred["confidence"]
            })

    return coco_predictions

def evaluate_with_pycocotools(gt_json_path, pred_json_path):
    coco_gt = COCO(gt_json_path)
    coco_dt = coco_gt.loadRes(pred_json_path)

    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == "__main__":
    GT_JSON_PATH = "../PascalCOCO/valid/_annotations.coco.json"
    PREDICTIONS_FOLDER = "../PascalCOCO/predictions"

    pred_annotations = read_predictions(PREDICTIONS_FOLDER)

    if not pred_annotations:
        print("Nessuna predizione valida trovata.")
    else:
        coco_predictions = convert_predictions_to_coco_format(
            pred_annotations, GT_JSON_PATH, mscoco_category2name
        )

        pred_json_path = "predictions_coco_format.json"
        with open(pred_json_path, 'w') as f:
            json.dump(coco_predictions, f)

        print(f"Salvato file di predizioni in formato COCO: {pred_json_path}")

        evaluate_with_pycocotools(GT_JSON_PATH, pred_json_path)