import torch
from PIL import Image
import json
from transformers import MllamaForConditionalGeneration, AutoProcessor
import argparse
from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv
import os
import traceback
from datetime import datetime
from pydantic import BaseModel, Field, HttpUrl, model_validator
from typing import List, Optional

# Enable better CUDA memory handling
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

parser = argparse.ArgumentParser(description="LlamaV-o1 Batch Inference")
parser.add_argument("--image_dir_path", type=str, required=True, help="Path to the images")
parser.add_argument("--data_type", type=str, default='validation', help="train, test, validation")
parser.add_argument("--start_idx", type=int, required=True)
parser.add_argument("--end_idx", type=int, required=True)
parser.add_argument("--samples", type=int, help="No of samples from the selected data type")
parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search")
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
        return Image.open(os.path.join(args.image_dir_path, self.image_file))

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

    def __init__(self, model_path='omkarthawakar/LlamaV-o1'):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float32
        ).to(device).eval()

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.device = device
        self.run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        os.makedirs("results", exist_ok=True)

        self.kwargs = {
            'max_new_tokens': 512,
            "top_p": 0.9,
            "pad_token_id": 128004,
            "bos_token_id": 128000,
            "do_sample": True,
            "eos_token_id": [128001, 128008, 128009],
            "temperature": 0.4,
            "num_beams": args.num_beams,
            "use_cache": True,
        }
        self.outputs = []

    def _get_caption_prompt(self, caption):
        return f"Provide a detailed description of the image titled {{{caption}}}, particularly emphasizing the aspects related to the question."

    def _get_summary_prompt(self, question):
        return question + "\nSummarize how you will approach the problem and explain the steps you will take to reach the answer."

    def _get_reasoning_prompt(self):
        return (
            "Provide a chain-of-thought, logical explanation of the problem. "
            "In case of multiple plots, carefully compare the y-axis and x-axis scales across plots and note any differences. "
            "Identify which lines correspond to which entities and distinguish them from shaded regions, which represent confidence intervals. "
            "Pay attention to overlapping lines or narrow peaks. Provide a step-by-step reasoning to reach the answer."
        )

    def _get_conclusion_prompt(self):
        return "Return a short, exact answer, no more than a few words. Do not explain or describe. The answer does not have to be a full sentence."

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
        return "Return either 'Yes' or 'No'. Do not add anything else - not even punctuation marks."

    def _get_qa_pair_prompt(self):
        return "Give the exact correct answer, with no extra explanation."

    def generate_inner(self, input: QAImageData):
        def __infer(messages: dict) -> str:
            input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            image = input.load_image().resize((224, 224))
            inputs = self.processor(image, input_text, return_tensors='pt')
            inputs = {
                k: v.to(self.device).float() if v.dtype.is_floating_point else v.to(self.device)
                for k, v in inputs.items()
            }
            torch.cuda.empty_cache()  # try to free any unused memory before inference
            with torch.no_grad():
                output = self.model.generate(**inputs, **self.kwargs)
            torch.cuda.empty_cache()
            return self.processor.decode(output[0][inputs['input_ids'].shape[1]:]).replace('<|eot_id|>', '').replace("<|end_of_text|>", "")

        def __tmp(inp, out):
            return [
                {'role': 'assistant', 'content': [{'type': 'text', 'text': inp}]},
                {'role': 'user', 'content': [{'type': 'text', 'text': out}]}
            ]

        messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': self._get_summary_prompt(input.question)}]}]
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

        if "closed-ended" in input.qa_pair_type and "finite answer set" in input.qa_pair_type:
            if "non-binary" in input.qa_pair_type and input.answer_options:
                qa_pair_prompt = self._get_non_binary_qa_pair_prompt(input.answer_options)
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

    def _should_override_with_unanswerable(self, reasoning: str) -> bool:
        keywords = ["cannot be determined", "cannot determine", "not possible to determine", "insufficient information", "not enough information", "unanswerable", "data is missing", "lack of information", "figure is not provided"]
        return any(keyword in reasoning.lower() for keyword in keywords)

    def _process_input(self):
        dataset = load_dataset("katebor/SciVQA", split=args.data_type)
        for idx, data in enumerate(tqdm(dataset)):
            if idx < args.start_idx:
                continue
            if idx > args.end_idx:
                break
            try:
                input = QAImageData(**data)
                result, reasoning = self.generate_inner(input)
                if self._should_override_with_unanswerable(reasoning):
                    result = "It is not possible to answer this question based only on the provided data."
                self.outputs.append({"instance_id": input.instance_id, "question": input.question, "answer_pred": result, "reasoning": reasoning})
                if idx % 10 == 0:
                    with open(f"results/{self.run_name}_sciqva_llamaVo1_{idx}.json", "w") as json_file:
                        json.dump(self.outputs, json_file, indent=4)
                        self.outputs = []
            except Exception as e:
                print(f"Error at index {idx}: {e}")
                traceback.print_exc()
                continue

if __name__ == "__main__":
    SciQVALlamaVO1Inference()._process_input()