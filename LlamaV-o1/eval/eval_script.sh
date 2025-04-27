#!/bin/bash
#SBATCH -p gpu-preempt
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --time=00:30:00
#SBATCH --mem=100G

source ../bin/activate
python llamav-o1.py