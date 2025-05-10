#!/bin/bash

curl -LsSf https://astral.sh/uv/install.sh | sh

source $HOME/.local/bin/env bash
# Step 2: Install vLLM using uv
uv pip install vllm

# Step 3: Launch vLLM OpenAI-compatible API server with Phi-4-multimodal-instruct
python -m vllm.entrypoints.openai.api_server \
  --model "/workspace/SDProc_SciVQA/phi4-modular/Qwen2.5-VL-7B-Instruct-SciVQA" \
  --dtype auto \
  --trust-remote-code \
  --max-model-len 32768

#curl -LsSf https://astral.sh/uv/install.sh | sh
#uv pip install vllm

#python -m vllm.entrypoints.openai.api_server --model 'microsoft/Phi-4-multimodal-instruct' --dtype auto --trust-remote-code --max-model-len 131072 --enable-lora --max-lora-rank 320  --limit-mm-per-prompt image=3 --max-loras 2 

