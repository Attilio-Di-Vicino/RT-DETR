import os
import argparse
from pathlib import Path
import subprocess

def main(args, ):
    # Create output folder if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Get list of images
    image_folder = Path(args.folder)
    image_paths = list(image_folder.glob("*.jpg"))

    if len(image_paths) == 0:
        print(f"No images found in {args.folder}")
        exit()

    # Run inference on each image
    for image_path in image_paths:
        print(f"Processing {image_path}...")
        output_image_path = os.path.join(args.output, image_path.name)

        command = [
            "python", "tools/infer.py",
            "-c", args.config,
            "-r", args.weights,
            "-f", str(image_path)
        ]

        subprocess.run(command)

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description="Batch inference on a folder of images.")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("-r", "--weights", type=str, required=True, help="Path to the model weights")
    parser.add_argument("-f", "--folder", type=str, required=True, help="Folder containing images")
    parser.add_argument("-o", "--output", type=str, default="results/", help="Output folder for results")
    args = parser.parse_args()
    main(args)