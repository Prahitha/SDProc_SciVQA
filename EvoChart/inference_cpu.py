from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import torch
from transformers.cache_utils import DynamicCache
import time
import requests
import os

import torch
from PIL import Image
import json
from transformers import AutoModelForCausalLM, AutoProcessor
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
from transformers.cache_utils import DynamicCache
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

parser = argparse.ArgumentParser(description="EvoChart Batch Inference")

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
    "--start_idx",
    type=int,
    required=True,
    help="Start index of the range to process."
)
parser.add_argument(
    "--end_idx",
    type=int,
    required=True,
    help="End index of the range to process."
)
parser.add_argument(
    "--samples",
    type=int,
    default=10,
    help="No of samples from the selected data type"
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
    answer: Optional[str]
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


class SciQVAEvoChartInference():

    def __init__(self, model_path='../../Phi-3.5/EvoChart'):
        self.device_map = "auto"
        offload_folder = "../../Phi-3.5/model_offload"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=self.device_map,
            offload_folder=offload_folder,
            trust_remote_code=True,
            use_safetensors=True,
            local_files_only=True,
            torch_dtype="auto",
            _attn_implementation='eager'
        )
        self.run_name = datetime.now().strftime("evochart_run_%Y%m%d_%H%M%S")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True,
                                                       num_crops=4)
        os.makedirs("results", exist_ok=True)
        self.kwargs = {
            "eos_token_id": self.processor.tokenizer.eos_token_id,
            "max_new_tokens": 2560,
            "temperature": 0.0,
            "do_sample": False,
            "use_cache": False,
        }
        self.outputs = []

    def _get_caption_prompt(self, caption):
        return f"Provide a detailed description of the image titled {{{caption}}}, particularly emphasizing the aspects related to the question."

    def _get_summary_prompt(self, caption, question):
        return caption + "\n" + \
            question

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

    def infer(self, input):
        images = [input.load_image()]
        # Create messages in simple format
        messages = [
            {
                'role': 'user',
                'content':  "<|image_1|>\n" + input.caption + "\n" + input.question
            }
        ]

        # Apply chat template
        full_prompt = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Process with images
        inputs = self.processor(full_prompt, images,
                                return_tensors='pt').to(self.model.device)

        input_length = inputs['input_ids'].shape[1]

        # Generate output
        output = self.model.generate(
            **inputs,
            # eos_token_id=self.processor.tokenizer.eos_token_id,
            **self.kwargs
        )

        completion = self.processor.batch_decode(output[:, input_length:],
                                                 skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=False)[0]

        # Add assistant response to messages
        messages.append({"role": "assistant", "content": completion})

        return completion, messages

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
            if idx < args.start_idx:
                continue
            if idx > args.end_idx:
                break
            # if idx == args.samples:
            #     break

            try:
                input = QAImageData(**data)
                result, reasoning = self.infer(input)

                # if self._should_override_with_unanswerable(reasoning):
                #     result = "It is not possible to answer this question based only on the provided data."

                self.outputs.append({
                    "instance_id": input.instance_id,
                    "question": input.question,
                    "answer_pred": result,
                    "reasoning": reasoning,
                })
                if idx % 5 == 0:
                    with open(f"results/{self.run_name}_sciqva_{idx}.json", "w") as json_file:
                        json.dump(self.outputs, json_file, indent=4)

            except Exception as e:
                print("Error :", e)
                continue


if __name__ == "__main__":
    SciQVAEvoChartInference()._process_input()
