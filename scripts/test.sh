#!/bin/bash

#SBATCH --account=jjparkcv1
#SBATCH --partition=spgpu
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --job-name="bct_isaac_var1_test"
#SBATCH --output=/nfs/turbo/coe-jjparkcv/niksrid/outputs/bct_isaac_var1_test.log
#SBATCH --mail-type=BEGIN,END,NONE,FAIL,REQUEUE

source /sw/pkgs/arc/python3.10-anaconda/2023.03/etc/profile.d/conda.sh
conda activate bc_algos
cd /home/niksrid/JJ/bc_algos
python scripts/test.py --config config/bc_transformer.json --dataset /nfs/turbo/coe-jjparkcv/datasets/isaac-gym-pick-place/simple/preprocessed/dataset_v16_diff --weights /nfs/turbo/coe-jjparkcv/niksrid/outputs/bct_isaac_var1/weights/model.pth --output /nfs/turbo/coe-jjparkcv/niksrid/outputs/bct_isaac_var1/rollout
wait