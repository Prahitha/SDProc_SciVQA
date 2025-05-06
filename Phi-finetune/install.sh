#!/bin/bash

git config --global user.name "NagaHarshita"
git config --global user.email "nagaharshitamarupaka@gmail.com"
git clone https://github.com/Prahitha/SDProc_SciVQA.git
git checkout phi-4-vllm-inference
cd Phi-finetune
mkdir scivqa_data
cd scivqa_data
git clone https://huggingface.co/datasets/katebor/SciVQA
unzip SciVQA/images_validation
bash finetune.sh