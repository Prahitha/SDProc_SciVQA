#!/bin/bash
#SBATCH -p gpu-preempt
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --time=00:30:00
#SBATCH --mem=100G

source ../LlamaV-o1-env/bin/activate
python inference.py