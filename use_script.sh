#!/bin/sh
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sk2324
#SBATCH --mail-user=ac2921
#SBATCH --mail-user=xg1020
source .venv/bin/activate
source /vol/cuda/12.4.0/setup.csh
nvidia-smi
nvcc --version
pip install -r /vol/bitbucket/xg1020/nlp-tone-detect/requirements.txt
python /vol/bitbucket/xg1020/nlp-tone-detect/use_model.py
