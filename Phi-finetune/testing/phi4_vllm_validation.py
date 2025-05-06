import argparse
import json
import os
import tempfile
import zipfile
import base64
import io
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import hf_hub_download


class SciVQAInferenceDataset:
    def __init__(self, data_size=None, split="validation", data_dir="./scivqa_data"):
        try:
            # Create a fixed directory instead of temp
            self.data_dir = Path(data_dir)
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.image_folder = self.data_dir / f"images_{split}"
            self.image_folder.mkdir(parents=True, exist_ok=True)

            # Download the image file
            print(f"Downloading images for {split} split...")
            image_file_path = hf_hub_download(
                repo_id='katebor/SciVQA',
                filename=f'images_{split}.zip',
                repo_type='dataset',
            )
            print(f'Image file downloaded to: {image_file_path}')

            # Download the JSON file
            print(f"Downloading annotations for {split} split...")
            json_file_path = hf_hub_download(
                repo_id='katebor/SciVQA',
                filename=f'{split}_2025-03-27_18-34-44.json',
                repo_type='dataset',
            )
            print(f'JSON file downloaded to: {json_file_path}')

            # Extract images to the fixed folder instead of temp
            print(f"Extracting images to {self.image_folder}...")
            with zipfile.ZipFile(image_file_path, 'r') as zip_ref:
                zip_ref.extractall(self.image_folder)

            # Load annotations directly from JSON file
            print("Loading JSON annotations...")
            with open(json_file_path, 'r') as f:
                annotations = json.load(f)

            # Convert to list format and limit size if specified
            if isinstance(annotations, dict):
                if 'data' in annotations:
                    self.annotations = annotations['data']
                elif 'annotations' in annotations:
                    self.annotations = annotations['annotations']
                else:
                    # Try the first key if it has a list value
                    for key, value in annotations.items():
                        if isinstance(value, list):
                            self.annotations = value
                            break
                    else:
                        self.annotations = []
            elif isinstance(annotations, list):
                self.annotations = annotations
            else:
                self.annotations = []

            if data_size is not None:
                self.annotations = self.annotations[:data_size]

            print(
                f"Loaded {len(self.annotations)} examples from {split} split")

            # Print a sample to debug
            if len(self.annotations) > 0:
                print("Sample annotation structure:")
                sample = self.annotations[0]
                for key, value in sample.items():
                    print(
                        f"  {key}: {str(value)[:50]}{'...' if len(str(value)) > 50 else ''}")

                # List image files to verify extraction
                # image_files = list(self.image_folder.glob(
                #     "*"))[:5]  # List first 5 files
                # print(
                #     f"Sample image files (showing first 5 of {len(list(self.image_folder.glob('*')))}):")
                # for img in image_files:
                #     print(f"  {img}")

        except Exception as e:
            import traceback
            print(f"Error initializing dataset: {e}")
            print(traceback.format_exc())
            self.annotations = []
            self.image_folder = None

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        try:
            annotation = self.annotations[idx]

            # Try different possible image file keys
            image_file_key = None
            for key in ['image_file', 'image_path', 'image', 'Figure_path']:
                if key in annotation and annotation[key]:
                    image_file_key = key
                    break

            if not image_file_key:
                raise KeyError("Could not find image file key in annotation")

            # Get image path
            image_file_name = annotation[image_file_key]
            # Handle potential path formats
            if isinstance(image_file_name, str):
                image_file_name = image_file_name.split(
                    '/')[-1]  # Get just the filename

            image_path = self.image_folder / image_file_name

            # Check if image exists
            if not image_path.exists():
                print(
                    f"Warning: Image file {image_path} does not exist. Searching for alternatives...")
                # Try to find the image with a slightly different name
                possible_matches = list(
                    self.image_folder.glob(f"*{image_file_name}*"))
                if possible_matches:
                    image_path = possible_matches[0]
                    print(f"Found alternative image: {image_path}")
                else:
                    print(f"No alternative image found for {image_file_name}")
                    # Return a placeholder instead of failing
                    return {
                        'id': annotation.get('instance_id', annotation.get('id', idx)),
                        'image_path': None,
                        'question': annotation.get('question', ''),
                        'caption': annotation.get('caption', ''),
                        'answer': annotation.get('answer', ''),
                        'choices': annotation.get('answer_options', annotation.get('choices', [])),
                        'qa_pair_type': annotation.get('qa_pair_type', ''),
                        'figure_type': annotation.get('figure_type', '')
                    }

            # Extract all the possible fields
            return {
                'id': annotation.get('instance_id', annotation.get('id', idx)),
                'image_path': str(image_path),
                'question': annotation.get('question', ''),
                'caption': annotation.get('caption', ''),
                'answer': annotation.get('answer', ''),
                'choices': annotation.get('answer_options', annotation.get('choices', [])),
                'qa_pair_type': annotation.get('qa_pair_type', ''),
                'figure_type': annotation.get('figure_type', '')
            }
        except Exception as e:
            print(f"Error getting item {idx}: {e}")
            # Return a placeholder
            return {
                'id': idx,
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
    """Convert an image to base64 encoding"""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    return encoded_string.decode('utf-8')


def vllm_inference(prompt, image_base64, vllm_url="http://localhost:8000/v1/chat/completions"):
    """Send a request to vLLM API server for Phi-4-MM"""
    headers = {
        "Content-Type": "application/json"
    }

    # Prepare the request payload
    payload = {
        "model": "Phi-4-multimodal-instruct",
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
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 150,
        "temperature": 0.0,  # Use 0 for deterministic output
    }

    try:
        response = requests.post(vllm_url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error making request to vLLM: {e}")
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
    args = parser.parse_args()

    # Load dataset
    dataset = SciVQAInferenceDataset(
        data_size=args.max_examples, split=args.split)

    results = []

    # Process dataset in batches
    n_batches = (len(dataset) + args.batch_size - 1) // args.batch_size
    for batch_idx in tqdm(range(n_batches), desc="Processing batches"):
        start_idx = batch_idx * args.batch_size
        batch = dataset.get_batch(start_idx, args.batch_size)

        for example in tqdm(batch, desc=f"Batch {batch_idx+1}/{n_batches}", leave=False):
            # Prepare data
            prompt = create_prompt(example)
            image_base64 = encode_image_to_base64(example['image_path'])

            # Run inference
            response = vllm_inference(
                prompt, image_base64, args.vllm_url)

            # Process response
            if response and 'choices' in response and len(response['choices']) > 0:
                generated_text = response['choices'][0]['message']['content'].strip(
                )

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
                    print(
                        f"\nProcessed {len(results)}/{len(dataset)} examples")
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

    # Calculate overall metrics
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
            values['accuracy'] = values['correct'] / \
                values['total'] if values['total'] > 0 else 0

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
        print(
            f"  {q_type}: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")

    print("\nAccuracy by Figure Type:")
    for f_type, metrics in figure_metrics.items():
        print(
            f"  {f_type}: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")

    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()

