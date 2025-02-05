import os
import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
import numpy as np
from src.core import YAMLConfig
from torch.cuda.amp import autocast
from src.solver import TASKS
from rtdetr_pytorch.tools.infer import postprocess, draw


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