import torch
import os
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from hubconf import rtdetr_r50vd

def main():
    model = rtdetr_r50vd()
    model.eval()
    
    # Trasformazione per il preprocessing delle immagini
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Percorso della cartella di validazione
    image_dir = "PascalCOCO/valid"
    output_dir = "PascalCOCO/output"  # Cartella per salvare i risultati
    os.makedirs(output_dir, exist_ok=True)

    # Scansiona la cartella e ottieni la lista delle immagini
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Loop su tutte le immagini
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        
        # 1Carica l'immagine
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)  # Aggiunge la dimensione batch
        
        # Esegui il modello
        with torch.no_grad():
            outputs = model(input_tensor, orig_target_sizes=[(image.height, image.width)])
        
        # Visualizza i risultati con bounding box
        # fig, ax = plt.subplots(1, figsize=(8, 6))
        # ax.imshow(image)

        if "boxes" in outputs and "labels" in outputs:  # Controlla il formato dell'output
            boxes = outputs["boxes"].cpu().numpy()  # Bounding box
            labels = outputs["labels"].cpu().numpy()  # Classi degli oggetti
            scores = outputs["scores"].cpu().numpy()  # Confidenza

            for box, label, score in zip(boxes, labels, scores):
                if score > 0.5:  # Filtra i risultati con bassa confidenza
                    x_min, y_min, x_max, y_max = box
                    rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                            linewidth=2, edgecolor="red", facecolor="none")
                    # ax.add_patch(rect)
                    # ax.text(x_min, y_min - 5, f"Class {label} ({score:.2f})", color="red", fontsize=10)

        # Salva l'immagine con i bounding box
        output_path = os.path.join(output_dir, image_file)
        plt.axis("off")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

        print(f"Processata: {image_file} → {output_path}")

    print("🎯 Inferenza completata su tutte le immagini!")

if __name__ == "__main__":
    main()