#!/bin/bash
#SBATCH -A IscrC_LASAN24_0
#SBATCH --output=rt_detr_%j.out
#SBATCH --error=rt_detr_%j.err
#SBATCH -p boost_usr_prod
#SBATCH --time=23:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=123000
#SBATCH --job-name=rtdetr

DATASET=$1
module load cuda/12.1
source ~/rtdetr/bin/activate
python tools/batch_infer.py \
    -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
    -r weights/rtdetr_r50vd_6x_coco.pth \
    -v ../PascalCOCO/video/$DATASET.mp4 \
    -o out_$DATASET.mp4

