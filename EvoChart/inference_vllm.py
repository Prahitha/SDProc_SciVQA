from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Fine-tuned model on Hugging Face
model_id = "microsoft/Phi-3.5-mini-instruct"

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Create chat prompt (or raw prompt if not chat-style)
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "How does gradient descent work?"}
]

# Apply chat template if it's chat-model
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True)

# Load the fine-tuned Phi-3.5 model with large context support
llm = LLM(
    model=model_id,
    trust_remote_code=True,
    max_model_len=131072  # 128K tokens
)

# Define generation parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=1000
)

# Generate text
outputs = llm.generate([prompt], sampling_params)
print(outputs[0].outputs[0].text)
