#!/bin/bash
#SBATCH --job-name=helpsteer_part2
#SBATCH --partition=gpu_h200
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --account=pi_sk2433
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module load miniconda
source activate final_proj
cd /home/nk725/final_project

# Toggle between BASELINE and CLAMPED before submitting
export RUN_TYPE=BASELINE
python 2_train_helpsteer.py
