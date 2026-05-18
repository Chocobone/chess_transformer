#!/usr/bin/bash

#SBATCH -J Gambit
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_ugrad_advisor_x
#SBATCH -w moana-u8
#SBATCH -t 1-0
#SBATCH -o logs/slurm-%A.out


source /data/yho7374/anaconda3/etc/profile.d/conda.sh
conda activate training

cd /data/yho7374/repos/chess_rust
python training/transformer.py

exit 0
