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
def draw(images, file_name, predictions_path, labels, boxes, scores, thrh = 0.6, path = ""):
    # Apre il file in modalità append prima di iniziare a iterare sulle immagini
    with open(predictions_path, "w") as f:  # Cambia "w" con "a" per appendere, se vuoi aggiungere al file
        for i, im in enumerate(images):
            draw = ImageDraw.Draw(im)
            scr = scores[i]
            lab = labels[i][scr > thrh]
            box = boxes[i][scr > thrh]
            scrs = scores[i][scr > thrh]
            for j,b in enumerate(box):
                draw.rectangle(list(b), outline='red',)
                draw.text((b[0], b[1]), text=f"label: {lab[j].item()} {round(scrs[j].item(),2)}", font=ImageFont.load_default(), fill='blue')
                # Scrive nel file le informazioni sul risultato
                f.write(f"label: {lab[j].item()} {round(scrs[j].item(),2)}, bbox: {list(b)}\n")

            if path == "":
                im.save(f"{file_name}.jpg")  
            else:
                os.makedirs(path, exist_ok=True)  
                im.save(os.path.join(path, f"{file_name}.jpg"))

            
def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)
    print(f"[INFO] Dataset: {args.input}")
    print(f"[INFO] Device: {args.device}")
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
    fps = len(execution_time) / total_time
    with open(info_path, "a") as f:
        f.write(f"------------------------------------------------\n")  # Scrivi il tempo medio
        if torch.cuda.is_available() and args.device == "cuda":
            f.write(f"Using GPU: TRUE\n")
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