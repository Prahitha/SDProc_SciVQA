from PIL import Image
import os
import torch
import json
from tqdm import tqdm
from transformers import MllamaForConditionalGeneration, AutoProcessor
from datasets import load_dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_beams", type=int, default=4)
args = parser.parse_args()

model_id = "omkarthawakar/LlamaV-o1"


model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
).eval()
processor = AutoProcessor.from_pretrained(model_id)
num_beams = args.num_beams
output_path = f"results_llamaVo1_beams{num_beams}.json"

max_new_tokens = 1024
summary_prompt = "\nSummarize how you will approach the problem and explain the steps you will take to reach the answer."
caption_prompt = "Provide a detailed description of the image, particularly emphasizing the aspects related to the question."
reasoning_prompt = "Provide a chain-of-thought, logical explanation of the problem. This should outline step-by-step reasoning."
conclusion_prompt = "State the final answer in a clear and direct format. It must match the correct answer exactly."
def generate_inner(question, image):
    kwargs = {
        'max_new_tokens': max_new_tokens,
        "top_p": 0.9,
        "pad_token_id": 128004,
        "bos_token_id": 128000,
        "do_sample": True,
        "eos_token_id": [
            128001,
            128008,
            128009
        ],
        "temperature": 0.6,
        "num_beams": num_beams,
        "use_cache": True,

    }
    messages = [
        {
            'role': 'user', 
            'content': [
                {'type': 'image'},
                {'type': 'text', 'text': question+summary_prompt}
            ],
        }
    ]

    def infer(messages: dict) -> str:
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(image, input_text, return_tensors='pt').to(model.device)
        output = model.generate(**inputs, **kwargs)
        return processor.decode(output[0][inputs['input_ids'].shape[1]:]).replace('<|eot_id|>', '').replace("<|end_of_text|>", "")

    def tmp(inp, out):
        return [
            {
                'role': 'assistant',
                'content': [
                    {'type': 'text', 'text': inp}
                ]
            },
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': out}
                ]
            }
        ]
    out = infer(messages)
    messages.extend(tmp(out, caption_prompt))
    out = infer(messages)
    messages.extend(tmp(out, reasoning_prompt))
    reasoning = infer(messages)
    messages.extend(tmp(reasoning, conclusion_prompt))
    kwargs['max_new_tokens'] = 50
    out = infer(messages)
    print(f"Question: {question}\nAnswer: {out}")
    return out, reasoning    


def reasoning_steps_answer(img, question, choices):
    
    predicted_answer, reasoning = generate_inner(question, img)
    return predicted_answer, reasoning

print(f"Evaluating with {num_beams=}")
print("="*50)


def get_old_results(idx):
    with open(output_path, "r") as json_file:
        old_data = json.load(json_file)
    for d in old_data:
        if d['idx'] == idx:
            return d


ds = load_dataset("omkarthawakar/VRC-Bench", split="test")

all_data = []
for data in tqdm(ds):
    try:
        image = data["image"]
        question = data["question"]
        final_answer = data["final_answer"]
        idx = data["idx"]
        reasoning_answer = data["steps"]
        question += "\nPlease select the correct option by its letter." if "Choices" in question else ""
        model_answer, reasoning = generate_inner(question, image)
        
        all_data.append({
            "idx": idx,
            "question": question,
            "final_answer": final_answer,
            "answer": reasoning_answer,
            "llm_response": reasoning+"\n\n\n"+model_answer,
        })
    except Exception as e:
        print("Error :", e)
        continue

model_pref = model_id.replace("/", "_")
with open(f"results_llamaVo1_beams{num_beams}.json", "w") as json_file:
    json.dump(all_data, json_file, indent=4)
