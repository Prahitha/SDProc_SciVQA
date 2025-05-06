import argparse
import json
import os
import base64
from pathlib import Path
from typing import Optional, Dict, List, Any, Union

import requests
from pydantic import BaseModel, model_validator, HttpUrl, Field
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset

class QAImageData(BaseModel):
    instance_id: str = Field(default_factory=lambda: f"id-{id(object())}")
    image_file: str
    figure_id: str = ""
    caption: str = ""
    figure_type: str = ""
    compound: bool = False
    figs_numb: str = ""
    qa_pair_type: str = ""
    question: str
    answer: Optional[str] = None
    answer_options: dict = {}
    venue: str = ""
    categories: str = ""
    source_dataset: str = ""
    paper_id: str = ""
    pdf_url: HttpUrl = "https://example.com"
    
    @model_validator(mode="before")
    def merge_answer_options(cls, values):
        options = values.get('answer_options')
        if isinstance(options, list):
            if all(isinstance(item, dict) for item in options):
                merged = {}
                for item in options:
                    merged.update(item)
                values['answer_options'] = merged
        return values


def load_and_validate_dataset(split="validation", data_size=None):
    """Load dataset from HuggingFace and validate with Pydantic"""
    print(f"Loading {split} dataset from HuggingFace...")
    
    try:
        # Load the dataset using HuggingFace
        raw_dataset = load_dataset("katebor/SciVQA", split=split)
        
        # Limit size if needed
        if data_size is not None:
            raw_dataset = raw_dataset.select(range(min(data_size, len(raw_dataset))))
        
        # Validate and convert to Pydantic models
        validated_data = []
        
        for i, item in enumerate(tqdm(raw_dataset, desc="Validating data")):
            try:
                # Convert from Dataset format to dictionary
                item_dict = dict(item)
                
                # Validate using Pydantic
                qa_item = QAImageData(**item_dict)
                validated_data.append(qa_item)
            except Exception as e:
                print(f"Error validating item {i}: {e}")
                # Skip invalid items
        
        print(f"Successfully loaded and validated {len(validated_data)} items")
        return validated_data
    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []


class SciVQAInferenceDataset:
    def __init__(self, data_size=None, split="validation", data_dir="./scivqa_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.image_folder = self.data_dir / "images_validation"
        self.image_folder.mkdir(parents=True, exist_ok=True)
        
        # Load and validate the dataset
        self.annotations = load_and_validate_dataset(split, data_size)
        print(f"Dataset loaded with {len(self.annotations)} examples")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        try:
            qa_data = self.annotations[idx]
            
            # Get image path
            image_file_name = qa_data.image_file
            if isinstance(image_file_name, str):
                image_file_name = image_file_name.split('/')[-1]  # Get just the filename

            image_path = self.image_folder / "images_validation" /image_file_name
            #print(image_path) 
            # Check if image exists
            if not os.path.exists(image_path):
                image_path = None
                print("Warning: Image file not found")
            
            # Extract fields from Pydantic model
            choices = qa_data.answer_options
            if isinstance(choices, dict):
                choices_list = [f"{k}: {v}" for k, v in choices.items()]
            elif isinstance(choices, list):
                if all(isinstance(c, dict) for c in choices):
                    choices_list = []
                    for c in choices:
                        for k, v in c.items():
                            choices_list.append(f"{k}: {v}")
                else:
                    choices_list = choices
            else:
                choices_list = []
            
            return {
                'id': qa_data.instance_id,
                'image_path': str(image_path) if image_path else None,
                'question': qa_data.question,
                'caption': qa_data.caption,
                'answer': qa_data.answer if qa_data.answer else "",
                'choices': choices_list,
                'qa_pair_type': qa_data.qa_pair_type,
                'figure_type': qa_data.figure_type
            }
        except Exception as e:
            print(f"Error getting item {idx}")
            # Return a placeholder
            return {
                'id': str(idx),
                'image_path': None,
                'question': '',
                'caption': '',
                'answer': '',
                'choices': [],
                'qa_pair_type': '',
                'figure_type': ''
            }

    def get_batch(self, start_idx, batch_size):
        end_idx = min(start_idx + batch_size, len(self))
        return [self[i] for i in range(start_idx, end_idx)]


def encode_image_to_base64(image_path):
    """Convert an image to base64 encoding, handling None values"""
    if image_path is None:
        print("Warning: No image available")
        return ""

    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        return encoded_string.decode('utf-8')
    except Exception as e:
        print("Error encoding image")
        return ""


def vllm_inference(prompt, image_path, vllm_url="http://localhost:8000/v1/chat/completions"):
    """Send a request to vLLM API server for Phi-4-MM using direct image file"""
    headers = {
        "Content-Type": "application/json"
    }

    try:
        if image_path and os.path.exists(image_path):
            # Read the image file and encode to base64
            with open(image_path, "rb") as f:
                encoded_string = base64.b64encode(f.read()).decode("utf-8")
            
            # Prepare the request payload with image
            payload = {
                "model": "microsoft/Phi-4-multimodal-instruct",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{encoded_string}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 150,
                "temperature": 0.0  # Use 0 for deterministic output
            }
        else:
            # Text-only query
            print("Warning: No valid image data. Proceeding with text-only query.")
            payload = {
                "model": "microsoft/Phi-4-multimodal-instruct",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"{prompt} (Note: Image unavailable)"
                            }
                        ]
                    }
                ],
                "max_tokens": 150,
                "temperature": 0.0
            }

        # Make the API call
        response = requests.post(vllm_url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except Exception as e:
        print(f"Error in vllm_inference: {e}")
        return None

def create_prompt(example, instruction="Answer the question based on the information in the image. Do not hallucinate or infer information from general knowledge."):
    """Create a prompt for the model based on the example"""
    question = example['question']
    caption = example['caption']
    choices = example['choices']
    qa_pair_type = example['qa_pair_type']

    # Build the prompt based on question type
    prompt_parts = []
    prompt_parts.append(f"Caption: {caption}")
    prompt_parts.append(f"Question: {question}")

    # Add choices if present
    if choices and len(choices) > 0:
        for choice in choices:
            prompt_parts.append(choice)

    # Add specific instructions based on question type
    if "closed-ended" in qa_pair_type and "finite answer set" in qa_pair_type:
        if "non-binary" in qa_pair_type and choices:
            prompt_parts.append(
                f"{instruction} Match the answer to one or more of the provided answer options. Return only the corresponding letter(s) of the correct answer(s).")
        elif "binary" in qa_pair_type:
            prompt_parts.append(f"{instruction} Return either 'Yes' or 'No'.")
        else:
            prompt_parts.append(
                f"{instruction} Give the exact correct answer, with no extra explanation.")
    else:
        prompt_parts.append(
            f"{instruction} Give the exact correct answer, with no extra explanation.")

    return "\n".join(prompt_parts)


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with Phi-4-MM on SciVQA dataset using vLLM")
    parser.add_argument("--vllm_url", type=str,
                        default="http://localhost:8000/v1/chat/completions", help="URL for vLLM API server")
    parser.add_argument("--output_file", type=str,
                        default="phi4mm_scivqa_results.json", help="Output JSON file path")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for processing")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Maximum number of examples to process")
    parser.add_argument("--split", type=str, default="validation",
                        choices=["train", "validation"], help="Dataset split to use")
    parser.add_argument("--data_dir", type=str, default="./scivqa_data",
                        help="Directory to store dataset files")
    args = parser.parse_args()

    # Load dataset
    dataset = SciVQAInferenceDataset(
        data_size=args.max_examples, 
        split=args.split,
        data_dir=args.data_dir
    )

    # Results storage
    results = []

    # Process dataset in batches
    n_batches = (len(dataset) + args.batch_size - 1) // args.batch_size
    for batch_idx in tqdm(range(n_batches), desc="Processing batches"):
        start_idx = batch_idx * args.batch_size
        batch = dataset.get_batch(start_idx, args.batch_size)

        for example in tqdm(batch, desc=f"Batch {batch_idx+1}/{n_batches}", leave=False):
            # Prepare data
            prompt = create_prompt(example)
            image_path = example['image_path']

            # Run inference
            response = vllm_inference(prompt, image_path, args.vllm_url)

            # Process response
            if response and 'choices' in response and len(response['choices']) > 0:
                generated_text = response['choices'][0]['message']['content'].strip()

                # Calculate accuracy (exact match)
                ground_truth = example['answer'].strip().lower()
                generated_clean = generated_text.strip().lower()
                is_correct = generated_clean == ground_truth

                # Store result
                result = {
                    'id': example['id'],
                    'question': example['question'],
                    'question_type': example['qa_pair_type'],
                    'figure_type': example['figure_type'],
                    'generated': generated_text,
                    'answer': example['answer'],
                    'correct': is_correct,
                }
                results.append(result)

                # Print progress
                if (len(results) % 10 == 0) or len(results) == 1:
                    print(f"\nProcessed {len(results)}/{len(dataset)} examples")
                    print(f"Question: {example['question']}")
                    print(f"Generated: {generated_text}")
                    print(f"Answer: {example['answer']}")
                    print(f"Correct: {is_correct}")
            else:
                print(f"Failed to get response for example {example['id']}")

            # Save intermediate results
            if len(results) % 20 == 0:
                with open(args.output_file, 'w') as f:
                    json.dump({
                        'results': results,
                        'progress': f"{len(results)}/{len(dataset)}"
                    }, f, indent=2)

    # Calculate metrics and save final results (same as before)
    correct = sum(1 for r in results if r['correct'])
    total = len(results)
    accuracy = correct / total if total > 0 else 0

    # Calculate metrics by question type
    type_metrics = {}
    for result in results:
        q_type = result['question_type']
        if q_type not in type_metrics:
            type_metrics[q_type] = {'correct': 0, 'total': 0}

        type_metrics[q_type]['total'] += 1
        if result['correct']:
            type_metrics[q_type]['correct'] += 1

    # Calculate figure type metrics
    figure_metrics = {}
    for result in results:
        f_type = result['figure_type']
        if f_type not in figure_metrics:
            figure_metrics[f_type] = {'correct': 0, 'total': 0}

        figure_metrics[f_type]['total'] += 1
        if result['correct']:
            figure_metrics[f_type]['correct'] += 1

    # Compute accuracies
    for metrics in [type_metrics, figure_metrics]:
        for key, values in metrics.items():
            values['accuracy'] = values['correct'] / values['total'] if values['total'] > 0 else 0

    # Final results
    final_results = {
        'overall': {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        },
        'by_question_type': type_metrics,
        'by_figure_type': figure_metrics,
        'results': results
    }

    # Save final results
    with open(args.output_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    # Print summary
    print("\n--- Results Summary ---")
    print(f"Overall Accuracy: {accuracy:.4f} ({correct}/{total})")

    print("\nAccuracy by Question Type:")
    for q_type, metrics in type_metrics.items():
        print(f"  {q_type}: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")

    print("\nAccuracy by Figure Type:")
    for f_type, metrics in figure_metrics.items():
        print(f"  {f_type}: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")

    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
