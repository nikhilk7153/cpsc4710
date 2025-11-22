#!/bin/bash
#SBATCH --job-name=find_feature
#SBATCH --partition=gpu_h200          # <--- You MUST type this to get H200s
#SBATCH --gpus=h200:1                 # Request 1 H200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16            
#SBATCH --mem=100G                    
#SBATCH --time=02:00:00
#SBATCH --account=pi_sk2433           # <--- This is the value from your 'sacctmgr' output

# Load environment
module load miniconda
source activate final_proj  # Use 'source activate' instead of 'conda activate' in scripts

# Run Step 1: Find the length feature
python 1_find_feature_helpsteer.py

