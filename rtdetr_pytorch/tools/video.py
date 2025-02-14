import cv2
import os

frame_folder = "../frames/images"
output_video = "output.mp4"
fps = 23
frames = sorted([f for f in os.listdir(frame_folder) if f.endswith(".png") or f.endswith(".jpg")])
first_frame = cv2.imread(os.path.join(frame_folder, frames[0]))
height, width, layers = first_frame.shape
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
for frame in frames:
    img = cv2.imread(os.path.join(frame_folder, frame))
    video.write(img)
video.release()
print(f"Video saved as {output_video}")
