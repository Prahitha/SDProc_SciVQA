import requests
import torch
import os
import json
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

model_id = "omkarthawakar/LlamaV-o1"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

with open('./SciVQA/validation_2025-03-27_18-34-44.json', 'r') as f:
    data = json.load(f)

data = data[:10]

def generate_inner(question, image, caption, qa_pair_type, answer_options):
    kwargs = {
        'max_new_tokens': 2048,
        "top_p": 0.9,
        "pad_token_id": 128004,
        "bos_token_id": 128000,
        "do_sample": True,
        "eos_token_id": [
            128001,
            128008,
            128009
        ],
        "temperature": 0.1,
        "num_beams": 5

    }
    messages = [
        {
            'role': 'user', 
            'content': [
                {'type': 'image'},
                {'type': 'text', 'text': question+"\nSummarize how you will approach the problem and explain the steps you will take to reach the answer."}
            ],
        }
    ]

    def infer(messages: dict) -> str:
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

        device = model.device

        inputs = processor(image, input_text, return_tensors='pt').to(device)
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

    with torch.no_grad():
        out = infer(messages)

    caption_prompt = f"Provide a detailed description of the image titled {{{caption}}}, particularly emphasizing the aspects related to the question."
    messages.extend(tmp(out, caption_prompt))
    out = infer(messages)
    # messages = add_turn(messages, "Provide a detailed description of the image titled '{caption}', particularly emphasizing the aspects related to the question.", plan)
    # messages_caption = messages + tmp(out, caption_prompt)

    with torch.no_grad():
        description = infer(messages)

    reasoning_prompt = (
        "Provide a chain-of-thought, logical explanation of the problem. "
        "In case of multiple plots, carefully compare the y-axis and x-axis scales across plots and note any differences. "
        "Identify which lines correspond to which entities and distinguish them from shaded regions, which represent confidence intervals. "
        "Pay attention to overlapping lines or narrow peaks. Provide a step-by-step reasoning to reach the answer."
    )
    messages.extend(tmp(out, reasoning_prompt))
    out = infer(messages)
    # messages = add_turn(messages, reasoning_prompt, description)

    with torch.no_grad():
        reasoning = infer(messages)

    conclusion_prompt = "Return a short, exact answer, no more than a few words. Do not explain or describe. The answer does not have to be a full sentence."
    messages.extend(tmp(out, conclusion_prompt))
    out = infer(messages)

    # messages = add_turn(messages, conclusion_prompt, reasoning)

    with torch.no_grad():
        conclusion = infer(messages)

    if "unanswerable" in qa_pair_type:
        final_prompt = "Return exactly this sentence: 'It is not possible to answer this question based only on the provided data.'"
    elif "closed-ended" in qa_pair_type and "finite answer set" in qa_pair_type:
        if "non-binary" in qa_pair_type and answer_options:
            final_prompt = f"Based on the reasoning above, match it to one of the provided answer options: {{{answer_options}}}. You are given with options A, B, C, D as answer_options: \
            A: The blue line, \
            B: The red line, \
            C: The gray line, \
            D: All of the above. If C: The gray line is correct, then return C. Choose the one that best fits. Only return the letter in the final answer. Do not add anything else."
        elif "binary" in qa_pair_type:
            final_prompt = "Return either 'Yes' or 'No'. Do not add anything else - not even punctuation marks."
        else:
            final_prompt = "Give the exact correct answer, with no extra explanation."
    else:
        final_prompt = "Give a direct, concise answer to the question only."

    messages.extend(tmp(out, final_prompt))
    out = infer(messages)
    
    # messages = add_turn(messages, final_prompt, conclusion)
    with torch.no_grad():
        final_answer = infer(messages)
    
    return final_answer.strip().replace(".", ""), reasoning # can we do summarization or BERT-based model for returning just the important part of the answer?

images_dir = './SciVQA/images_validation'

for entry in data:
    image_file = entry['image_file']
    question = entry['question']
    caption = entry['caption']
    qa_pair_type = entry['qa_pair_type']
    answer_options = entry['answer_options']

    image_path = os.path.join(images_dir, image_file)
    image = Image.open(image_path)

    result, reasoning = generate_inner(question, image, caption, qa_pair_type, answer_options)
    print(result)

