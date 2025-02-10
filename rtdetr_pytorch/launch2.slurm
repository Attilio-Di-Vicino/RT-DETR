#!/bin/bash
#SBATCH -A IscrC_LASAN24_0
#SBATCH --output=rt_detr_%j.out
#SBATCH --error=rt_detr_%j.err
#SBATCH -p boost_usr_prod
#SBATCH --time=01:59:00      # HH:MM:SS
#SBATCH -N 1                 # 1 nodo richiesto
#SBATCH --ntasks-per-node=4  # 4 tasks su 32 disponibili
#SBATCH --gres=gpu:2         # 1 GPU richiesta su un nodo che ne ha 4
#SBATCH --mem=123000         # Memoria richiesta (123GB su ~481GB)
#SBATCH --job-name=pinns_angio_gpu

DATASET=$1

module load cuda/12.1 
source ~/rtdetr/bin/activate

# Verifica GPU prima di eseguire lo script
echo "Test GPU prima dell'esecuzione:"
nvidia-smi || echo "Errore: impossibile accedere alla GPU"

# Esegui lo script con srun (opzionale)
srun python tools/infer3.py \
    -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
    -r weights/rtdetr_r50vd_6x_coco.pth \
    -i ../PascalCOCO/$DATASET \
    -o ../PascalCOCO/ \
    -s False \
    -d cuda