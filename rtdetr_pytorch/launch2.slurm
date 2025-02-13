#!/bin/bash
#SBATCH -A IscrC_LASAN24_0
#SBATCH --output=rt_detr_%j.out
#SBATCH --error=rt_detr_%j.err
#SBATCH -p boost_usr_prod
#SBATCH --time=10:00:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=123000
#SBATCH --job-name=pinns_angio_gpu

DATASET=$1
NC=$2

# Carica i moduli necessari
module load cuda/12.1
source ~/rtdetr/bin/activate

# Verifica lo stato della GPU prima dell'esecuzione
echo "Stato della GPU prima dell'esecuzione:"
nvidia-smi

# Esegui il tuo script Python
python tools/infer3.py \
    -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
    -r weights/rtdetr_r50vd_6x_coco.pth \
    -i ../PascalCOCO/$DATASET \
    -o ../PascalCOCO/ \
    -d cuda \
    -nc $NC

# Verifica lo stato della GPU dopo l'esecuzione
echo "Stato della GPU dopo l'esecuzione:"
nvidia-smi