#!/bin/bash
#SBATCH --job-name=helpsteer_base
#SBATCH --partition=gpu_h200
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --account=pi_sk2433
#SBATCH --output=/home/nk725/final_project/logs/%x_%j.out
#SBATCH --error=/home/nk725/final_project/logs/%x_%j.err

module load miniconda
source activate final_proj
cd /home/nk725/final_project

python 2_train_helpsteer.py --run_type BASELINE
