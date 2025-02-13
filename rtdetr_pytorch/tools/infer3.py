import torch
import torch.nn as nn 
import torchvision.transforms as T
from torch.cuda.amp import autocast
import numpy as np 
from PIL import Image, ImageDraw, ImageFont
import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse
import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS
import numpy as np
import time

def postprocess(labels, boxes, scores, iou_threshold=0.55):
    def calculate_iou(box1, box2):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        union_area = box1_area + box2_area - inter_area
        iou = inter_area / union_area if union_area != 0 else 0
        return iou
    merged_labels = []
    merged_boxes = []
    merged_scores = []
    used_indices = set()
    for i in range(len(boxes)):
        if i in used_indices:
            continue
        current_box = boxes[i]
        current_label = labels[i]
        current_score = scores[i]
        boxes_to_merge = [current_box]
        scores_to_merge = [current_score]
        used_indices.add(i)
        for j in range(i + 1, len(boxes)):
            if j in used_indices:
                continue
            if labels[j] != current_label:
                continue  
            other_box = boxes[j]
            iou = calculate_iou(current_box, other_box)
            if iou >= iou_threshold:
                boxes_to_merge.append(other_box.tolist())  
                scores_to_merge.append(scores[j])
                used_indices.add(j)
        xs = np.concatenate([[box[0], box[2]] for box in boxes_to_merge])
        ys = np.concatenate([[box[1], box[3]] for box in boxes_to_merge])
        merged_box = [np.min(xs), np.min(ys), np.max(xs), np.max(ys)]
        merged_score = max(scores_to_merge)
        merged_boxes.append(merged_box)
        merged_labels.append(current_label)
        merged_scores.append(merged_score)
    return [np.array(merged_labels)], [np.array(merged_boxes)], [np.array(merged_scores)]
def slice_image(image, slice_height, slice_width, overlap_ratio):
    img_width, img_height = image.size
    
    slices = []
    coordinates = []
    step_x = int(slice_width * (1 - overlap_ratio))
    step_y = int(slice_height * (1 - overlap_ratio))
    
    for y in range(0, img_height, step_y):
        for x in range(0, img_width, step_x):
            box = (x, y, min(x + slice_width, img_width), min(y + slice_height, img_height))
            slice_img = image.crop(box)
            slices.append(slice_img)
            coordinates.append((x, y))
    return slices, coordinates
def merge_predictions(predictions, slice_coordinates, orig_image_size, slice_width, slice_height, threshold=0.30):
    merged_labels = []
    merged_boxes = []
    merged_scores = []
    orig_height, orig_width = orig_image_size
    for i, (label, boxes, scores) in enumerate(predictions):
        x_shift, y_shift = slice_coordinates[i]
        scores = np.array(scores).reshape(-1)
        valid_indices = scores > threshold
        valid_labels = np.array(label).reshape(-1)[valid_indices]
        valid_boxes = np.array(boxes).reshape(-1, 4)[valid_indices]
        valid_scores = scores[valid_indices]
        for j, box in enumerate(valid_boxes):
            box[0] = np.clip(box[0] + x_shift, 0, orig_width)  
            box[1] = np.clip(box[1] + y_shift, 0, orig_height)
            box[2] = np.clip(box[2] + x_shift, 0, orig_width)  
            box[3] = np.clip(box[3] + y_shift, 0, orig_height) 
            valid_boxes[j] = box
        merged_labels.extend(valid_labels)
        merged_boxes.extend(valid_boxes)
        merged_scores.extend(valid_scores)
    return np.array(merged_labels), np.array(merged_boxes), np.array(merged_scores)
# def draw(images, file_name, predictions_path, labels, boxes, scores, thrh = 0.6, path = ""):
#     for i, im in enumerate(images):
#         draw = ImageDraw.Draw(im)
#         scr = scores[i]
#         lab = labels[i][scr > thrh]
#         box = boxes[i][scr > thrh]
#         scrs = scores[i][scr > thrh]
#         for j,b in enumerate(box):
#             draw.rectangle(list(b), outline='red',)
#             draw.text((b[0], b[1]), text=f"label: {lab[j].item()} {round(scrs[j].item(),2)}", font=ImageFont.load_default(), fill='blue')
#             with open(predictions_path, "w") as f:
#                 f.write(f"label: {lab[j].item()} {round(scrs[j].item(),2)}, bbox: {list(b)}")

#         if path == "":
#             im.save(f"{file_name}.jpg")  
#         else:
#             os.makedirs(path, exist_ok=True)  
#             im.save(os.path.join(path, f"{file_name}.jpg"))
# def draw(images, file_name, predictions_path, labels, boxes, scores, thrh = 0.6, path = ""):
#     # Apre il file in modalità append prima di iniziare a iterare sulle immagini
#     with open(predictions_path, "w") as f:  # Cambia "w" con "a" per appendere, se vuoi aggiungere al file
#         for i, im in enumerate(images):
#             draw = ImageDraw.Draw(im)
#             scr = scores[i]
#             lab = labels[i][scr > thrh]
#             box = boxes[i][scr > thrh]
#             scrs = scores[i][scr > thrh]
#             for j,b in enumerate(box):
#                 draw.rectangle(list(b), outline='red',)
#                 draw.text((b[0], b[1]), text=f"label: {lab[j].item()} {round(scrs[j].item(),2)}", font=ImageFont.load_default(), fill='blue')
#                 # Scrive nel file le informazioni sul risultato
#                 f.write(f"label: {lab[j].item()} {round(scrs[j].item(),2)}, bbox: {list(b)}\n")

#             if path == "":
#                 im.save(f"{file_name}.jpg")  
#             else:
#                 os.makedirs(path, exist_ok=True)  
#                 im.save(os.path.join(path, f"{file_name}.jpg"))
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
    89: 'hair dryer',
    90: 'toothbrush'
}

mscoco_category2color = {
    1: (255, 0, 0),    # Rosso per 'person'
    2: (0, 255, 0),    # Verde per 'bicycle'
    3: (0, 0, 255),    # Blu per 'car'
    4: (255, 255, 0),  # Giallo per 'motorcycle'
    5: (255, 165, 0),  # Arancione per 'airplane'
    6: (128, 0, 128),  # Viola per 'bus'
    7: (0, 255, 255),  # Ciano per 'train'
    8: (255, 105, 180),# Rosa per 'truck'
    9: (75, 0, 130),   # Indaco per 'boat'
    10: (255, 255, 255), # Bianco per 'traffic light'
    11: (255, 69, 0),  # Rosso arancio per 'fire hydrant'
    13: (139, 69, 19), # Marrone per 'stop sign'
    14: (255, 223, 0), # Oro per 'parking meter'
    15: (160, 82, 45), # Marrone chiaro per 'bench'
    16: (173, 216, 230), # Azzurro per 'bird'
    17: (128, 128, 128), # Grigio per 'cat'
    18: (165, 42, 42),   # Marrone scuro per 'dog'
    19: (102, 51, 0),    # Terra per 'horse'
    20: (153, 255, 153), # Verde chiaro per 'sheep'
    21: (139, 0, 0),     # Rosso scuro per 'cow'
    22: (255, 228, 181), # Pesca per 'elephant'
    23: (0, 0, 0),       # Nero per 'bear'
    24: (255, 255, 153), # Giallo chiaro per 'zebra'
    25: (218, 165, 32),  # Oro scuro per 'giraffe'
    27: (0, 128, 0),     # Verde scuro per 'backpack'
    28: (0, 0, 128),     # Blu scuro per 'umbrella'
    31: (192, 192, 192), # Argento per 'handbag'
    32: (128, 0, 0),     # Marrone scuro per 'tie'
    33: (0, 139, 139),   # Verde acqua scuro per 'suitcase'
    34: (255, 20, 147),  # Rosa intenso per 'frisbee'
    35: (70, 130, 180),  # Acciaio per 'skis'
    36: (72, 61, 139),   # Indaco scuro per 'snowboard'
    37: (255, 140, 0),   # Arancione scuro per 'sports ball'
    38: (240, 230, 140), # Khaki per 'kite'
    39: (160, 160, 160), # Grigio medio per 'baseball bat'
    40: (222, 184, 135), # Sabbia per 'baseball glove'
    41: (255, 99, 71),   # Rosso tomato per 'skateboard'
    42: (128, 128, 0),   # Oliva per 'surfboard'
    43: (85, 107, 47),   # Verde bosco per 'tennis racket'
    44: (210, 180, 140), # Beige per 'bottle'
    46: (205, 133, 63),  # Marrone per 'wine glass'
    47: (255, 239, 213), # Crema per 'cup'
    48: (192, 192, 192), # Argento per 'fork'
    49: (169, 169, 169), # Grigio scuro per 'knife'
    50: (128, 128, 128), # Grigio per 'spoon'
    51: (112, 128, 144), # Grigio bluastro per 'bowl'
    52: (255, 255, 0),   # Giallo per 'banana'
    53: (255, 0, 255),   # Magenta per 'apple'
    54: (255, 182, 193), # Rosa chiaro per 'sandwich'
    55: (255, 165, 0),   # Arancione per 'orange'
    56: (0, 128, 0),     # Verde scuro per 'broccoli'
    57: (255, 127, 80),  # Corallo per 'carrot'
    58: (255, 69, 0),    # Rosso arancio per 'hot dog'
    59: (139, 69, 19),   # Marrone per 'pizza'
    60: (255, 215, 0),   # Oro per 'donut'
    61: (139, 0, 0),     # Rosso scuro per 'cake'
    62: (165, 42, 42),   # Marrone per 'chair'
    63: (128, 0, 0),     # Marrone scuro per 'couch'
    64: (85, 107, 47),   # Verde bosco per 'potted plant'
    65: (255, 218, 185), # Pesca chiaro per 'bed'
    67: (70, 130, 180),  # Acciaio per 'dining table'
    70: (105, 105, 105), # Grigio scuro per 'toilet'
    72: (173, 216, 230), # Azzurro per 'tv'
    73: (0, 255, 255),   # Ciano per 'laptop'
    74: (240, 128, 128), # Rosso chiaro per 'mouse'
    75: (100, 149, 237), # Blu cobalto per 'remote'
    76: (184, 134, 11),  # Oro scuro per 'keyboard'
    77: (255, 140, 0),   # Arancione scuro per 'cell phone'
    78: (46, 139, 87),   # Verde mare per 'microwave'
    79: (160, 82, 45),   # Marrone per 'oven'
    80: (205, 133, 63),  # Marrone per 'toaster'
    81: (32, 178, 170),  # Verde scuro per 'sink'
    82: (0, 0, 128),     # Blu scuro per 'refrigerator'
    84: (139, 69, 19),   # Marrone per 'book'
    85: (255, 215, 0),   # Oro per 'clock'
    86: (255, 182, 193), # Rosa chiaro per 'vase'
    87: (255, 105, 180), # Rosa per 'scissors'
    88: (255, 20, 147),  # Rosa intenso per 'teddy bear'
    89: (70, 130, 180),  # Acciaio per 'hair dryer'
    90: (128, 0, 128)    # Viola per 'toothbrush'
}

def draw(images, file_name, predictions_path, labels, boxes, scores, thrh=0.6, path=""):
    with open(predictions_path, "w") as f:
        for i, im in enumerate(images):
            draw = ImageDraw.Draw(im)
            scr = scores[i]
            lab = labels[i][scr > thrh]
            box = boxes[i][scr > thrh]
            scrs = scores[i][scr > thrh]

            for j, b in enumerate(box):
                label_id = lab[j].item() + 1
                label_name = mscoco_category2name.get(label_id, "Unknown")  # Nome della classe
                color = mscoco_category2color.get(label_id, (255, 255, 255))  # Bianco di default
                
                # Disegna rettangolo con linea più spessa
                draw.rectangle(list(b), outline=color, width=3)
                
                # Crea testo con label e score
                text = f"{label_name} {round(scrs[j].item(), 2)}"
                
                # Font più leggibile
                try:
                    font = ImageFont.truetype("arial.ttf", 16)  # Font più chiaro se disponibile
                except:
                    font = ImageFont.load_default()  # Fallback
                
                # Scrive il testo sopra il box
                draw.text((b[0], b[1] - 10), text, font=font, fill=color)
                
                # Scrive nel file di output
                f.write(f"label: {label_name} {round(scrs[j].item(),2)}, bbox: {list(b)}\n")

            # Salva l'immagine
            save_path = os.path.join(path, f"{file_name}.jpg") if path else f"{file_name}.jpg"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            im.save(save_path)

            
def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)
    print(f"[INFO] Dataset: {args.input}")
    print(f"[INFO] Device: {args.device}")
    print(f"[INFO] NC: {args.numberofboxes}")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu') 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')
    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)
    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs
    
    model = Model().to(args.device)

    image_files = [f for f in os.listdir(args.input) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    image_folder = os.path.join(args.output, "images")
    os.makedirs(image_folder, exist_ok=True)

    execution_time = []

    for img_file in image_files:
        # start_time = time.time()
        print(f"[INFO] img_file: {img_file}")
        im_pil = Image.open(os.path.join(args.input, img_file)).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([w, h])[None].to(args.device)
        
        transforms = T.Compose([
            T.Resize((640, 640)),  
            T.ToTensor(),
        ])
        im_data = transforms(im_pil)[None].to(args.device)
        if args.sliced:
            num_boxes = args.numberofboxes
            
            aspect_ratio = w / h
            num_cols = int(np.sqrt(num_boxes * aspect_ratio)) 
            num_rows = int(num_boxes / num_cols)
            slice_height = h // num_rows
            slice_width = w // num_cols
            overlap_ratio = 0.2
            slices, coordinates = slice_image(im_pil, slice_height, slice_width, overlap_ratio)
            predictions = []
            start_time = time.time()
            for i, slice_img in enumerate(slices):
                slice_tensor = transforms(slice_img)[None].to(args.device)
                with autocast():  # Use AMP for each slice
                    output = model(slice_tensor, torch.tensor([[slice_img.size[0], slice_img.size[1]]]).to(args.device))
                torch.cuda.empty_cache() 
                labels, boxes, scores = output
                
                labels = labels.cpu().detach().numpy()
                boxes = boxes.cpu().detach().numpy()
                scores = scores.cpu().detach().numpy()
                predictions.append((labels, boxes, scores))
            end_time = time.time()
            
            merged_labels, merged_boxes, merged_scores = merge_predictions(predictions, coordinates, (h, w), slice_width, slice_height)
            labels, boxes, scores = postprocess(merged_labels, merged_boxes, merged_scores)
        else:
            start_time = time.time()
            output = model(im_data, orig_size)
            end_time = time.time()
            labels, boxes, scores = output

        # Salvataggio delle predizioni in un file di testo
        predictions_folder = os.path.join(args.output, "predictions")
        os.makedirs(predictions_folder, exist_ok=True)

        predictions_path = os.path.join(predictions_folder, f"{img_file}.txt")

        # file_name = img_file.removesuffix(".jpg")
        draw([im_pil], img_file, predictions_path, labels, boxes, scores, 0.6, image_folder)
        # end_time = time.time()
        elapsed_time = end_time - start_time
        with open(predictions_path, "a") as f:
                f.write(f"\nExecution time: {elapsed_time:.4f} sec\n")
        execution_time.append(elapsed_time)
    
    total_time = sum(execution_time)
    average_time = sum(execution_time) / len(execution_time) if execution_time else 0
    info_path = "info.txt"
    gpu_info = "No GPU available"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated(0)
        free_memory = total_memory - allocated_memory
        gpu_info = f"GPU: {gpu_name}, Total Memory: {total_memory // (1024**2)} MB, Allocated Memory: {allocated_memory // (1024**2)} MB, Free Memory: {free_memory // (1024**2)} MB"

    # Scrivere i dati nel file info.txt
    info_path = "info.txt"
    if total_time > 0:
        fps = len(execution_time) / total_time
    else:
        fps = len(execution_time)
    with open(info_path, "a") as f:
        f.write(f"------------------------------------------------\n")  # Scrivi il tempo medio
        if torch.cuda.is_available() and args.device == "cuda":
            f.write(f"Using GPU: TRUE\n")
        f.write(f"Slide: {args.sliced}\n")
        f.write(f"Total time: {total_time:.4f} sec\n")  # Scrivi il tempo medio
        f.write(f"Average execution time: {average_time:.4f} sec\n")  # Scrivi il tempo medio
        f.write(f"Total images processed: {len(execution_time)}\n")  # Scrivi il numero di immagini processate
        f.write(f"FPS: {fps:.2f}\n")
        f.write(f"{gpu_info}\n")  # Scrivi le informazioni sulla GPU
        f.write(f"------------------------------------------------")  # Scrivi il tempo medio

    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, )
    parser.add_argument('-r', '--resume', type=str, )
    parser.add_argument('-i', '--input', type=str, required=True, help="Path to the folder containing images")
    parser.add_argument('-o', '--output', type=str, required=True, help="Path to the folder containing outputs")
    parser.add_argument('-s', '--sliced', type=bool, default=False)
    parser.add_argument('-d', '--device', type=str, default='cpu')
    parser.add_argument('-nc', '--numberofboxes', type=int, default=25)
    args = parser.parse_args()
    main(args)