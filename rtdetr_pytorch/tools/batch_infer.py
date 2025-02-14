import os
import argparse
import cv2
import torch
import multiprocessing as mp
import time
from pathlib import Path
from functools import partial
import subprocess

def run_inference(params):
    """Esegue inferenza su un singolo frame usando CUDA."""
    image_path, args, gpu_id = params 
    print(f"Processing {image_path} on GPU {gpu_id}...")
    output_path = image_path.replace("temp_frames", "processed_frames")
    command = [
        "python", "tools/infer.py",
        "-c", args.config,
        "-r", args.weights,
        "-f", str(image_path),
        "-d", str("cuda")
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # Assegna la GPU corretta

    subprocess.run(command, env=env)

def get_gpu_info():
    """Get GPU information."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)  
        allocated_memory = torch.cuda.memory_allocated(0) // (1024 * 1024)  
        free_memory = total_memory - allocated_memory
    else:
        gpu_name = "CPU"
        total_memory, allocated_memory, free_memory = 0, 0, 0

    return gpu_name, total_memory, allocated_memory, free_memory

def main(args):
    if not torch.cuda.is_available():
        print("CUDA non disponibile, verrà usata solo la CPU.")
        device_count = 1
    else:
        device_count = torch.cuda.device_count()
        print(f"Rilevate {device_count} GPU!")

    os.makedirs("temp_frames", exist_ok=True)
    os.makedirs("processed_frames", exist_ok=True)
    video_path = args.video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Errore nell'apertura del video")
        exit()
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0 

    frame_count = 0
    frame_paths = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_path = f"temp_frames/frame_{frame_count:05d}.jpg"
        cv2.imwrite(frame_path, frame)  
        frame_paths.append(frame_path)  

    cap.release()
    print(f"Totale frame estratti: {frame_count}")

    with mp.Pool(processes=device_count) as pool:
        params = [(frame_path, args, i % device_count) for i, frame_path in enumerate(frame_paths)]
        pool.map(run_inference, params)
    print("Inferenza completata!")

    output_video = args.output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec per MP4
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for i in range(1, frame_count + 1):
        processed_frame_path = f"processed_frames/frame_{i:05d}.jpg"
        frame = cv2.imread(processed_frame_path)
        if frame is not None:
            out.write(frame)

    out.release()
    print(f"Video saved as {output_video}")

    # get GPU info
    gpu_name, total_mem, allocated_mem, free_mem = get_gpu_info()

    # write on video.txt
    with open("video.txt", "a") as f:
        f.write(f"Total images/frame processed: {frame_count}\n")
        f.write(f"Video FPS: {fps}, Duration: {duration:.2f} sec, Total Frames: {total_frames}\n")
        f.write(f"GPU: {gpu_name}, Total Memory: {total_mem} MB, Allocated Memory: {allocated_mem} MB, Free Memory: {free_mem} MB\n\n")

    print("Informazioni salvate in video.txt")
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