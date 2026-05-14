#!/usr/bin/bash

#SBATCH -J Gambit
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p debug_eebme_ugrad
#SBATCH -w moana-u8
#SBATCH -t 1-0
#SBATCH -o logs/slurm-%A.out


/data/yho7374/anaconda3/bin/conda init
source ~/.bashrc
conda activate training

cd /data/yho7374/repos/chess_rust/preprocess

python data_filtering.py
python chess_tokenize.py
