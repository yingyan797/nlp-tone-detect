#!/bin/sh
#SBATCH --gres=gpu:1

source .venv/bin/activate
source /vol/cuda/12.4.0/setup.csh
nvidia-smi
nvcc --version
pip install -r /vol/bitbucket/xg1020/nlp-tone-detect/requirements.txt
python /vol/bitbucket/xg1020/nlp-tone-detect/analyse_by_severity.py
