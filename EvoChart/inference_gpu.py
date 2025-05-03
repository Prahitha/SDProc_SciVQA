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
import os
import torch
import json
from tqdm import tqdm
from datasets import load_dataset
import argparse


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
    def __init__(self, model_path='Evochart'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_map = "auto"  # Better for multi-GPU; use "cuda:0" for singl
        self.offload_folder = "model_offload"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=self.device_map,
            offload_folder=self.offload_folder,
            trust_remote_code=True,
            use_safetensors=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16,  # or torch.float16 based on support
            _attn_implementation="flash_attention_2"  # ensure flash-attn is installed
        )

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            num_crops=4
        )

        self.run_name = datetime.now().strftime("evochart_run_%Y%m%d_%H%M%S")
        os.makedirs("results", exist_ok=True)

        self.kwargs = {
            "max_new_tokens": 2560,
            "temperature": 0.0,
            "do_sample": False,
            "use_cache": False,
        }

        self.outputs = []

        print(f"✅ Model loaded on: {self.device_map}")

    def _generate_response(self, messages, images):
        """Helper function to generate a single response"""
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
            eos_token_id=self.processor.tokenizer.eos_token_id,
            **self.kwargs
        )

        completion = self.processor.batch_decode(output[:, input_length:],
                                                 skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=False)[0]

        return completion

    def simplified_chain_of_thought(self, input):
        """Ultra-simplified two-step chain-of-thought"""
        images = [input.load_image()]

        # Step 1: Analyze question and extract data
        step1_response = self._analyze_and_extract(input, images)

        # Step 2: Final answer
        final_answer = self._provide_answer(input, images, step1_response)

        return {
            'final_answer': final_answer,
            'analysis': step1_response
        }

    def _analyze_and_extract(self, input, images):
        """Combined analysis and data extraction"""
        messages = [
            {
                'role': 'user',
                'content': f"<|image_1|>\n{input.caption}\n\nQuestion: {input.question}\n\nAnalyze what this question is asking and extract the relevant data from the chart needed to answer it. Show your reasoning."
            }
        ]
        return self._generate_response(messages, images)

    def _provide_answer(self, input, images, analysis):
        """Provide final answer"""
        messages = [
            {
                'role': 'user',
                'content': f"<|image_1|>\n{input.caption}\n\nQuestion: {input.question}\n\nAnalysis: {analysis}\n\nBased on this analysis, what is the answer?"
            }
        ]
        return self._generate_response(messages, images)

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

    def _get_visual_prompt(self):
        return "Rely on the visual content of the graph like labels, axis content to infer the answer. Need not do intense calculations."

    def _get_binary_qa_pair_prompt(self):
        return (
            "Return either 'Yes' or 'No'. Do not add anything else - not even punctuation marks."
        )

    def _get_qa_pair_prompt(self):
        return "Give the exact correct answer, with no extra explanation."

    def direct_qa(self, input):
        """Direct question answering with single call"""
        images = [input.load_image()]

        qa_pair_prompt = ""
        if "closed-ended" in input.qa_pair_type and "finite answer set" in input.qa_pair_type:
            if "non-binary" in input.qa_pair_type and input.answer_options:
                qa_pair_prompt = self._get_non_binary_qa_pair_prompt(
                    input.answer_options)
            elif "binary" in input.qa_pair_type:
                qa_pair_prompt = self._get_binary_qa_pair_prompt()
            else:
                qa_pair_prompt = self._get_qa_pair_prompt()

            if "non-visual" in input.qa_pair_type:
                pass
            elif "visual" in input.qa_pair_prompt:
                qa_pair_prompt += self._get_visual_prompt()
        else:
            qa_pair_prompt = self._get_qa_pair_prompt()

        messages = [
            {
                'role': 'user',
                'content': f"<|image_1|>\n{input.caption}\n\nQuestion: {input.question}\n{qa_pair_prompt}"
            }
        ]

        return self._generate_response(messages, images), messages

    def structured_chain_of_thought(self, input):
        """Structured chain-of-thought with specific analysis steps"""
        images = [input.load_image()]

        # Step 1: Chart elements identification
        step1_response = self._analyze_chart_elements(input, images)

        # Step 2: Data extraction
        step2_response = self._extract_relevant_data(
            input, images, step1_response)

        # Step 3: Question analysis
        step3_response = self._analyze_question(
            input, images, step1_response, step2_response)

        # Step 4: Reasoning process
        step4_response = self._develop_reasoning(
            input, images, step1_response, step2_response, step3_response)

        # Step 5: Final answer
        final_answer = self._generate_final_answer(
            input, images, step1_response, step2_response, step3_response, step4_response)

        return {
            'final_answer': final_answer,
            'chart_analysis': step1_response,
            'data_extraction': step2_response,
            'question_analysis': step3_response,
            'reasoning_process': step4_response
        }

    def _analyze_chart_elements(self, input, images):
        """Step 1: Identify chart elements"""
        messages = [
            {
                'role': 'user',
                'content': f"<|image_1|>\n{input.caption}\n\nIdentify the following elements in this chart:\n1. Chart type\n2. Axes labels and units\n3. Legend items\n4. Data series present\n5. Any notable visual features"
            }
        ]
        return self._generate_response(messages, images)

    def _extract_relevant_data(self, input, images, previous_analysis):
        """Step 2: Extract data points"""
        messages = [
            {
                'role': 'user',
                'content': f"<|image_1|>\n{input.caption}\n\nPrevious analysis: {previous_analysis}\n\nQuestion: {input.question}\n\nExtract the specific data values that might be relevant to answering the question. Include:\n1. Key data points\n2. Trends or patterns\n3. Maximum/minimum values if relevant"
            }
        ]
        return self._generate_response(messages, images)

    def _analyze_question(self, input, images, chart_analysis, data_extraction):
        """Step 3: Understand what the question is asking"""
        messages = [
            {
                'role': 'user',
                'content': f"<|image_1|>\n{input.caption}\n\nChart analysis: {chart_analysis}\n\nData extracted: {data_extraction}\n\nQuestion: {input.question}\n\nAnalyze what this question is asking for:\n1. What type of answer is expected?\n2. What specific data points are needed?\n3. Are there any calculations required?"
            }
        ]
        return self._generate_response(messages, images)

    def _develop_reasoning(self, input, images, chart_analysis, data_extraction, question_analysis):
        """Step 4: Develop step-by-step reasoning"""
        messages = [
            {
                'role': 'user',
                'content': f"<|image_1|>\n{input.caption}\n\nPrevious analyses:\n- Chart: {chart_analysis}\n- Data: {data_extraction}\n- Question: {question_analysis}\n\nDevelop a step-by-step approach to answer the question:\n1. What data points to use\n2. What calculations (if any)\n3. How to interpret the results"
            }
        ]
        return self._generate_response(messages, images)

    def _generate_final_answer(self, input, images, *all_analyses):
        """Step 5: Generate final answer"""
        analysis_summary = "\n".join(
            [f"Step {i+1}: {analysis}" for i, analysis in enumerate(all_analyses)])

        messages = [
            {
                'role': 'user',
                'content': f"<|image_1|>\n{input.caption}\n\nQuestion: {input.question}\n\nAnalysis summary:\n{analysis_summary}\n\nBased on all the analysis above, provide a concise and accurate answer to the question."
            }
        ]
        return self._generate_response(messages, images)

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
            eos_token_id=self.processor.tokenizer.eos_token_id,
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
                result, reasoning = self.direct_qa(input)

                if self._should_override_with_unanswerable(result):
                    result = "It is not possible to answer this question based only on the provided data."
                print(input.question)
                print(result)
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
