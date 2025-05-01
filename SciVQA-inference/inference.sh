#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --nodelist=gpu033
#SBATCH --time=01:30:00
#SBATCH --mem=60G
#SBATCH --qos=short

module load conda/latest
conda activate LlamaV-o1-env
python batch_inference_older_gpu.py --image_dir_path ../SciVQA/images_test --data_type test --start_idx 2440 --end_idx 2499