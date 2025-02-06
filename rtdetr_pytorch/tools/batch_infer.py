# import os
# import argparse
# from pathlib import Path
# import torch
# import multiprocessing as mp
# from functools import partial
# import subprocess

# def run_inference(image_path, args):
#     """Run inference on a single image using CUDA."""
#     print(f"Processing {image_path} on GPU {torch.cuda.current_device()}...")

#     command = [
#         "python", "tools/infer.py",
#         "-c", args.config,
#         "-r", args.weights,
#         "-f", str(image_path)
#     ]

#     subprocess.run(command)

# def main(args):
#     # Check GPU availability
#     if not torch.cuda.is_available():
#         print("CUDA is not available. Running on CPU...")
#         device_count = 1
#     else:
#         device_count = torch.cuda.device_count()
#         print(f"Running on {device_count} GPUs!")

#     # Create output folder if it doesn't exist
#     os.makedirs(args.output, exist_ok=True)

#     video_path = "video.mp4"  # Sostituiscilo con il tuo file
#     cap = cv2.VideoCapture(video_path)

#     # Controlla se il video si apre correttamente
#     if not cap.isOpened():
#         print("Errore nell'apertura del video")
#         exit()

#     frame_count = 0  # Contatore dei frame

#     while cap.isOpened():
#         ret, frame = cap.read()  # Legge un frame alla volta
#         if not ret:
#             break  # Se il video è finito, esci dal loop

#         # Salva il frame o processalo con il modello
#         frame_count += 1
#         print(f"Processing frame {frame_count}")  # Debug

#         # Use multiprocessing to parallelize on GPUs
#         with mp.Pool(processes=device_count) as pool:
#             pool.map(partial(run_inference, args=args), frame)

#         # Mostra il frame (opzionale)
#         cv2.imshow("Frame", frame)
        
#         # Premi 'q' per uscire prima
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Chiudi tutto
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == '__main__':
#     mp.set_start_method('spawn', force=True)  # <-- Imposta 'spawn' per CUDA
#     parser = argparse.ArgumentParser(description="Batch inference on a folder of images using CUDA.")
#     parser.add_argument("-c", "--config", type=str, required=True, help="Path to the config file")
#     parser.add_argument("-r", "--weights", type=str, required=True, help="Path to the model weights")
#     parser.add_argument("-f", "--folder", type=str, required=True, help="Folder containing images")
#     parser.add_argument("-o", "--output", type=str, default="results/", help="Output folder for results")
#     args = parser.parse_args()
    
#     main(args)
import os
import argparse
import cv2
import torch
import multiprocessing as mp
from pathlib import Path
from functools import partial
import subprocess

def run_inference(image_path, args, gpu_id):
    """Esegue inferenza su un singolo frame usando CUDA."""
    print(f"Processing {image_path} on GPU {gpu_id}...")

    output_path = image_path.replace("temp_frames", "processed_frames")
    
    command = [
        "python", "tools/infer.py",
        "-c", args.config,
        "-r", args.weights,
        "-f", str(image_path),
        "-o", str(output_path)  # Specifica l'output dell'inferenza
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # Assegna la GPU corretta

    subprocess.run(command, env=env)

def main(args):
    # Controlla se CUDA è disponibile
    if not torch.cuda.is_available():
        print("CUDA non disponibile, verrà usata solo la CPU.")
        device_count = 1
    else:
        device_count = torch.cuda.device_count()
        print(f"Rilevate {device_count} GPU!")

    # Crea le cartelle per i frame
    os.makedirs("temp_frames", exist_ok=True)
    os.makedirs("processed_frames", exist_ok=True)

    # Carica il video
    video_path = args.video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Errore nell'apertura del video")
        exit()

    # Ottieni FPS e dimensioni del video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_count = 0
    frame_paths = []

    # Estrai frame e salvali come immagini temporanee
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_path = f"temp_frames/frame_{frame_count:05d}.jpg"
        cv2.imwrite(frame_path, frame)  # Salva il frame come immagine
        frame_paths.append(frame_path)  # Aggiungi il percorso alla lista

    cap.release()
    print(f"Totale frame estratti: {frame_count}")

    # Parallelizza il processo su più GPU
    with mp.Pool(processes=device_count) as pool:
        pool.map(partial(run_inference, args=args), enumerate(frame_paths))

    print("Inferenza completata!")

    # Ricostruisci il video
    output_video = args.output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec per MP4
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for i in range(1, frame_count + 1):
        processed_frame_path = f"processed_frames/frame_{i:05d}.jpg"
        frame = cv2.imread(processed_frame_path)
        if frame is not None:
            out.write(frame)

    out.release()
    print(f"Video salvato come {output_video}")

    # Pulizia: rimuovi i frame temporanei
    for path in frame_paths:
        os.remove(path)
    os.rmdir("temp_frames")

    for i in range(1, frame_count + 1):
        os.remove(f"processed_frames/frame_{i:05d}.jpg")
    os.rmdir("processed_frames")

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="RT-DETR inference su un video MP4 e salvataggio output.")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path al file di configurazione")
    parser.add_argument("-r", "--weights", type=str, required=True, help="Path ai pesi del modello")
    parser.add_argument("-v", "--video", type=str, required=True, help="Path del video")
    parser.add_argument("-o", "--output", type=str, default="output.mp4", help="File video di output")

    args = parser.parse_args()
    
    main(args)