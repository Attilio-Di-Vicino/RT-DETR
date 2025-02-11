import cv2
import os

# Cartella contenente i frame
frame_folder = "../frames/images"
output_video = "output.mp4"
fps = 23

# Ottieni la lista dei frame ordinati
frames = sorted([f for f in os.listdir(frame_folder) if f.endswith(".png") or f.endswith(".jpg")])

# Leggi il primo frame per ottenere le dimensioni
first_frame = cv2.imread(os.path.join(frame_folder, frames[0]))
height, width, layers = first_frame.shape

# Definisci il codec e crea il VideoWriter (usa "mp4v" per MP4)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Scrivi i frame nel video
for frame in frames:
    img = cv2.imread(os.path.join(frame_folder, frame))
    video.write(img)

# Rilascia il video writer
video.release()
print(f"Video salvato come {output_video}")
