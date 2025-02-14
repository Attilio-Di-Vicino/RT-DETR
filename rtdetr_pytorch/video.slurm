#!/bin/bash
#SBATCH -A IscrC_LASAN24_0
#SBATCH --output=rt_detr_%j.out
#SBATCH --error=rt_detr_%j.err
#SBATCH -p boost_usr_prod
#SBATCH --time=01:59:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=123000
#SBATCH --job-name=pinns_angio_gpu

DATASET=$1
module load cuda/12.1
source ~/rtdetr/bin/activate
echo "GPU state before execution:"
nvidia-smi
python tools/batch_infer.py \
    -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
    -r weights/rtdetr_r50vd_6x_coco.pth \
    -v ../PascalCOCO/video/video.mp4 \
    -o output.mp4
echo "GPU state after execution:"
nvidia-smi

