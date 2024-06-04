#!/bin/bash

#SBATCH --account=jjparkcv1
#SBATCH --partition=spgpu
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name="bc_transformer"
#SBATCH --output=/home/niksrid/JJ/bc_algos/outputs/bc_transformer.log
#SBATCH --mail-type=BEGIN,END,NONE,FAIL,REQUEUE

source /sw/pkgs/arc/python3.10-anaconda/2023.03/etc/profile.d/conda.sh
conda activate mental-models
cd /home/niksrid/JJ/bc_algos
python scripts/test.py --config config/bc_transformer.json --dataset /home/niksrid/bc_algos/datasets/proc/dataset_v15_diff --weights /home/niksrid/bc_algos/outputs/bac_issac_100_high_res/model_599.pth --output outputs/
wait