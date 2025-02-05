import os
import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
import numpy as np
from src.core import YAMLConfig
from torch.cuda.amp import autocast
from src.solver import TASKS

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


def draw(images, labels, boxes, scores, thrh=0.6, output_dir=""):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)
        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scores[i][scr > thrh]
        for j,b in enumerate(box):
            draw.rectangle(list(b), outline='red', width=3)
            draw.text((b[0], b[1]), text=f"label: {lab[j]} {round(scrs[j], 2)}", font=ImageFont.load_default(), fill='blue')
        # Salva l'immagine modificata
        output_path = os.path.join(output_dir, f"result_{i}.jpg")
        im.save(output_path)
        print(f"Immagine salvata in {output_path}")


def main(args):
    # Crea una directory di output se non esiste
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Carica il modello
    cfg = YAMLConfig(args.config, resume=args.resume)
    checkpoint = torch.load(args.resume, map_location='cpu') 
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
    else:
        state = checkpoint['model']

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(args.device)

    # Carica tutte le immagini dalla cartella
    image_dir = args.image_dir
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Trasformazione per preprocessare le immagini
    transforms = T.Compose([T.Resize((640, 640)), T.ToTensor()])

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        im_pil = Image.open(image_path).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([w, h])[None].to(args.device)

        # Pre-elabora l'immagine
        im_data = transforms(im_pil).unsqueeze(0).to(args.device)

        # Esegui il modello
        with autocast():
            output = model(im_data, orig_size)

        # Estrai le predizioni
        labels, boxes, scores = output
        labels = labels.cpu().detach().numpy()
        boxes = boxes.cpu().detach().numpy()
        scores = scores.cpu().detach().numpy()

        # Esegui il post-processing
        labels, boxes, scores = postprocess(labels, boxes, scores)

        # Disegna i bounding box sull'immagine
        draw([im_pil], labels, boxes, scores, thrh=0.6, output_dir=output_dir)
        print(f"Elaborata immagine: {image_file}")
    
    print("Elaborazione completata!")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='Path to config file')
    parser.add_argument('-r', '--resume', type=str, help='Path to checkpoint file')
    parser.add_argument('-d', '--device', type=str, default='cpu', help='Device to use for inference (cpu/cuda)')
    parser.add_argument('-i', '--image-dir', type=str, required=True, help='Directory containing the images to process')
    args = parser.parse_args()
    main(args)