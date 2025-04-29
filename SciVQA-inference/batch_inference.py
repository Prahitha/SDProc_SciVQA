import torch
from PIL import Image
import json
from transformers import MllamaForConditionalGeneration, AutoProcessor
import argparse
from datasets import load_dataset
import tqdm
from PIL import Image
from dotenv import load_dotenv
import os
import torch
import json
from tqdm import tqdm
from datasets import load_dataset
import argparse
import openai

from pydantic import BaseModel, Field, HttpUrl, model_validator
from typing import List, Optional
from PIL import Image
import os
from datetime import datetime

parser = argparse.ArgumentParser(description="LlamaV-o1 Batch Inference")

parser.add_argument(
    "--image_dir_path",
    type=str,
    help="Path to the images",
    required=True
)

parser.add_argument(
    "--data_type",
    type=str,
    default='validation',
    help="train, test, validation",
)
parser.add_argument(
    "--samples",
    type=int,
    default=10,
    help="No of samples from the selected data type"
)
parser.add_argument(
    "--num_beams", 
    type=int, 
    default=4,
    help="Number of beams for beam search"
)

args = parser.parse_args()


class QAImageData(BaseModel):
    instance_id: str
    image_file: str
    figure_id: str
    caption: str
    figure_type: str
    compound: bool
    figs_numb: str
    qa_pair_type: str
    question: str
    answer: str
    answer_options: dict
    venue: str
    categories: str
    source_dataset: str
    paper_id: str
    pdf_url: HttpUrl

    def load_image(self) -> Image.Image:
        """Loads and returns the image given the directory path."""
        image_path = os.path.join(args.image_dir_path, self.image_file)
        return Image.open(image_path)

    @model_validator(mode="before")
    def merge_answer_options(cls, values):
        options = values.get('answer_options')
        if isinstance(options, list):
            merged = {}
            for item in options:
                merged.update(item)
            values['answer_options'] = merged
        return values


class SciQVALlamaVO1Inference():

    def _split_model(self):
        device_map = {}
        num_gpus = torch.cuda.device_count()
        rank, world_size = 0, 1

        num_gpus = num_gpus // world_size

        num_layers = 60  # Llama 3.2 11B has 60 layers

        # Manual cost accounting
        # Vision encoder and projector treated like extra layers
        total_cost = num_layers + 5 + 7
        num_layers_per_gpu = total_cost // num_gpus  # Even split

        # Now distribute
        num_layers_per_gpu = [num_layers_per_gpu] * num_gpus
        num_layers_per_gpu[0] -= 5  # ViT cost adjustment
        num_layers_per_gpu[-1] -= 7  # projector + lm_head adjustment

        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for j in range(num_layer):
                device_map[f'language_model.model.layers.{layer_cnt}'] = rank + \
                    world_size * i
                layer_cnt += 1

        # Special components
        device_map['vision_model'] = rank  # GPU 0
        device_map['language_model.model.embed_tokens'] = rank  # GPU 0
        device_map['language_model.model.rotary_emb'] = rank  # GPU 0
        device_map['language_model.model.norm'] = rank + \
            world_size * (num_gpus - 1)  # GPU 1
        device_map['language_model.lm_head'] = rank + \
            world_size * (num_gpus - 1)  # GPU 1
        device_map['multi_modal_projector'] = rank + \
            world_size * (num_gpus - 1)  # GPU 1

        return device_map

    def __init__(self, model_path='omkarthawakar/LlamaV-o1'):
        self.device_map = "auto" if torch.cuda.device_count() < 2 else self._split_model()
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device_map,
        ).eval()
        self.run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.processor = AutoProcessor.from_pretrained(model_path)
        os.makedirs("results", exist_ok=True)
        self.kwargs = {
            'max_new_tokens': 1024,
            "top_p": 0.9,
            "pad_token_id": 128004,
            "bos_token_id": 128000,
            "do_sample": True,
            "eos_token_id": [
                128001,
                128008,
                128009
            ],
            "temperature": 0.4,
            "num_beams": args.num_beams,
            "use_cache": True,
        }
        self.outputs = []

    def _get_caption_prompt(self, caption):
        return f"Provide a detailed description of the image titled {{{caption}}}, particularly emphasizing the aspects related to the question."

    def _get_summary_prompt(self, question):
        return (
            question +
            "\nSummarize how you will approach the problem and explain the steps you will take to reach the answer."
        )

    def _get_reasoning_prompt(self):
        return (
            "Provide a chain-of-thought, logical explanation of the problem. "
            "In case of multiple plots, carefully compare the y-axis and x-axis scales across plots and note any differences. "
            "Identify which lines correspond to which entities and distinguish them from shaded regions, which represent confidence intervals. "
            "Pay attention to overlapping lines or narrow peaks. Provide a step-by-step reasoning to reach the answer."
        )

    def _get_conclusion_prompt(self):
        return (
            "Return a short, exact answer, no more than a few words. Do not explain or describe. The answer does not have to be a full sentence."
        )

    def _get_non_binary_qa_pair_prompt(self, answer_options):
        return (
            f"Based on the reasoning above, match it to one or more of the provided answer options: {answer_options}. "
            "Return only the corresponding letter(s) of the correct answer(s). "
            "Do not explain your choice, do not rephrase the answer, and do not repeat the option text. "
            "Only output the letter(s) corresponding to the correct choice. "
            "If multiple letters are correct, separate them by commas without spaces (for example: B,C). "
            "If all options are correct, return A,B,C,D. "
            "Do not add anything else."
        )

    def _get_binary_qa_pair_prompt(self):
        return (
            "Return either 'Yes' or 'No'. Do not add anything else - not even punctuation marks."
        )

    def _get_qa_pair_prompt(self):
        return "Give the exact correct answer, with no extra explanation."

    def generate_inner(self, input: QAImageData):
        question = input_data.question
        image = input_data.load_image(args.image_dir_path)
        caption = input_data.caption
        qa_pair_type = input_data.qa_pair_type
        answer_options = input_data.answer_options
        
        def __infer(messages: dict) -> str:
            input_text = self.processor.apply_chat_template(
                messages, add_generation_prompt=True)
            inputs = self.processor(input.load_image(), input_text,
                                    return_tensors='pt').to(self.model.device)
            output = self.model.generate(**inputs, **self.kwargs)
            return self.processor.decode(output[0][inputs['input_ids'].shape[1]:]).replace('<|eot_id|>', '').replace("<|end_of_text|>", "")

        def __tmp(inp, out):
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

        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'image'},
                    {'type': 'text',
                        'text': self._get_summary_prompt(input.question)}
                ],
            }
        ]
        summary_qa = __infer(messages)

        caption_prompt = self._get_caption_prompt(input.caption)
        messages.extend(__tmp(summary_qa, caption_prompt))
        summary_qa = __infer(messages)

        reasoning_prompt = self._get_reasoning_prompt()
        messages.extend(__tmp(summary_qa, reasoning_prompt))
        reasoning_qa = __infer(messages)

        conclusion_prompt = self._get_conclusion_prompt()
        messages.extend(__tmp(summary_qa, conclusion_prompt))
        summary_qa = __infer(messages)
        qa_pair_prompt = ""

        if "closed-ended" in input.qa_pair_type and "finite answer set" in input.qa_pair_type:
            if "non-binary" in input.qa_pair_type and input.answer_options:
                qa_pair_prompt = self._get_non_binary_qa_pair_prompt(
                    input.answer_options)
            elif "binary" in input.qa_pair_type:
                qa_pair_prompt = self._get_binary_qa_pair_prompt()
            else:
                qa_pair_prompt = self._get_qa_pair_prompt()
        else:
            qa_pair_prompt = self._get_qa_pair_prompt()

        messages.extend(__tmp(summary_qa, qa_pair_prompt))
        output = __infer(messages)

        print(f"Question: {input.question}\nAnswer: {output}")
        return output, reasoning_qa

    def evaluate(self):
        pass

    def _should_override_with_unanswerable(self, reasoning: str) -> bool:
        reasoning_lower = reasoning.lower()
        keywords = [
            "cannot be determined",
            "cannot determine",
            "not possible to determine",
            "insufficient information",
            "not enough information",
            "unanswerable",
            "data is missing",
            "lack of information",
            "figure is not provided"
        ]
        return any(keyword in reasoning_lower for keyword in keywords)

    def _process_input(self):
        dataset = load_dataset("katebor/SciVQA", split=args.data_type)

        for idx, data in enumerate(tqdm(dataset)):
            if idx == args.samples:
                break

            try:
                input = QAImageData(**data)
                result, reasoning = self.generate_inner(input)

                if self._should_override_with_unanswerable(reasoning):
                    result = "It is not possible to answer this question based only on the provided data."

                self.outputs.append({
                    "instance_id": input.instance_id,
                    "question": input.question,
                    "answer_pred": result,
                    "reasoning": reasoning,
                })
                if idx % 10 == 0:
                    with open(f"results/{self.run_name}_sciqva_llamaVo1_{idx}.json", "w") as json_file:
                        json.dump(self.outputs, json_file, indent=4)
                        self.outputs = []

            except Exception as e:
                print("Error :", e)
                continue


if __name__ == "__main__":
    SciQVALlamaVO1Inference()._process_input()
