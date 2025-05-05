#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --constraint=vram48
#SBATCH --mem=15G

module load conda/latest
conda activate phi4-env
# python inference_gpu.py
python vllm_test.py