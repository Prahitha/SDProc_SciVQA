from vllm import LLM, SamplingParams

# Initialize the model
llm = LLM(model="microsoft/Phi-4-mini-instruct",
          dtype="float16",
        )  # Replace with your model

# Generate text
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=100)
prompts = ["Write a short poem about machine learning."]
outputs = llm.generate(prompts, sampling_params)

# Print the generated text
for output in outputs:
    print(output.prompt)
    print(output.outputs[0].text)
    print("-" * 50)