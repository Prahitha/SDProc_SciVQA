#!/bin/bash

git config --global user.name "NagaHarshita"
git config --global user.email "nagaharshitamarupaka@gmail.com"
mkdir scivqa_data
cd scivqa_data
git clone https://huggingface.co/datasets/katebor/SciVQA
unzip SciVQA/images_validation
unzip SciVQA/images_test
unzip SciVQA/images_train
cd ..
bash finetune.sh
