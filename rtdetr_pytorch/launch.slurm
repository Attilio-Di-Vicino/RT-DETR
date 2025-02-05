#!/bin/bash
#SBATCH -A IscrC_LASAN24_0
#SBATCH --output=rt_detr_%j.out
#SBATCH --error=rt_detr_%j.err
#SBATCH -p boost_usr_prod
#SBATCH --time 01:59:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=4 # 4 tasks out of 32
#SBATCH --gres=gpu:1        # 4 gpus per node out of 4
#SBATCH --mem=123000          # memory per node out of 494000MB (481GB)
#SBATCH --job-name=pinns_angio_gpu

module load cuda/12.1 
source ~/rtdetr/bin/activate
python tools/batch_infer.py \
    -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
    -r weights/rtdetr_r50vd_6x_coco.pth \
    -f ../PascalCOCO/valid2 \
    