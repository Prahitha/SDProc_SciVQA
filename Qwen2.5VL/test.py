from PIL import Image
from datasets import load_dataset
from unsloth import FastVisionModel, is_bfloat16_supported # FastLanguageModel for LLMs
import torch

model, tokenizer = FastVisionModel.from_pretrained(
    "https://huggingface.co/Infyn/Qwen2.5-VL-7B-Instruct-SciVQA",
    # fast_inference = True, # Enable vLLM fast inference -> not available for Qwen2.5
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
    # gpu_memory_utilization = 0.6, # Reduce if out of memory
)

dataset = load_dataset("katebor/SciVQA", split = "train")

extract_dir = "/workspace/dataset"

instruction = """Provide a detailed, step-by-step logical analysis of the image and question.
1. Identify the type of visualization
2. Describe what the axes represent
3. Explain key trends, patterns, or anomalies
4. Connect these observations to the specific question"""

def convert_to_conversation(annotation):

        # Load the image
        image_path = extract_dir + "/images_train/" + annotation['image_file']

        # Get question and metadata
        question = annotation['question']
        qa_pair_type = annotation['qa_pair_type']
        figure_type = annotation['figure_type']
        figures = annotation['figs_numb']
        caption = annotation['caption']

        # Format the answer
        answer = f'{annotation["answer"]}<|end|><|endoftext|>'

        # Handle answer options
        choices = []
        if annotation['answer_options']:
            for i, option in enumerate(annotation['answer_options']):
                # Assuming answer_options is a list of values or a list of dicts
                if isinstance(option, dict):
                    letter = list(option.keys())[0]
                    value = list(option.values())[0]
                    choices.append(f"{letter}: {value}")
                else:
                    # If it's just a list of values, create A, B, C, etc. options
                    letter = chr(65 + i) # A, B, C, ...
                    choices.append(f"{letter}: {option}")

        # Create prompt based on question type
        if "closed-ended" in qa_pair_type and "finite answer set" in qa_pair_type:
            if "non-binary" in qa_pair_type and choices:
                final_prompt = (
                    f"Match the answer to one or more of the provided answer options: {{{', '.join(choices)}}}. "
                    "Return only the corresponding letter(s) of the correct answer(s). "
                    "Do not explain your choice, do not rephrase the answer, and do not repeat the option text. "
                    "Only output the letter(s) corresponding to the correct choice. "
                    "If multiple letters are correct, separate them by commas without spaces (for example: B,C). "
                    "If all options are correct, return A,B,C,D. "
                    "Do not add anything else."
                )
            elif "binary" in qa_pair_type:
                final_prompt = "Return either 'Yes' or 'No'. Do not add anything else - not even punctuation marks."
            else:
                final_prompt = (
                    "Give the exact correct answer, with no extra explanation. If the reasoning says the answer cannot be determined "
                    "or that the information is insufficient, return exactly: 'It is not possible to answer this question based only on the provided data.'"
                )
        else:
            final_prompt = (
                "Give the exact correct answer, with no extra explanation. If the reasoning says the answer cannot be determined "
                "or that the information is insufficient, return exactly: 'It is not possible to answer this question based only on the provided data.'"
            )

        # Format the user message with image reference
        question_text = instruction + "\n" + final_prompt + "\n" + question

        # Create the item dictionary for this single example
        conversation = [
                { "role": "user",
                  "content" : [
                    {"type" : "text",  "text"  : question_text},
                    {"type" : "image", "image" : Image.open(image_path)}]
                },
                { "role" : "assistant",
                  "content" : [
                    {"type" : "text",  "text"  : answer}]
                },
            ]

        return { "messages" : conversation }


converted_dataset = [convert_to_conversation(sample) for sample in dataset]

converted_dataset[0]['messages']

def extract_image(prompt):
    for message in prompt:
        if message['role'] == 'user':
            for content_item in message['content']:
                if content_item['type'] == 'image':
                    return content_item['image']

def extract_user(prompt):
    for message in prompt:
        if message['role'] == 'user':
            return message

FastVisionModel.for_inference(model) # Enable for inference!

image = extract_image(converted_dataset[2]["messages"])
messages = [extract_user(converted_dataset[2]["messages"])]

input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
inputs = tokenizer(
    image,
    input_text,
    add_special_tokens = False,
    return_tensors = "pt",
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
                   use_cache = True, temperature = 0.2, min_p = 0.1)