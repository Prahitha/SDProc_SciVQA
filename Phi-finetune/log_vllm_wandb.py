from datetime import datetime
import argparse
import json
import os
import base64
from pathlib import Path
from typing import Optional, Dict, List, Any, Union

import requests
import wandb
import numpy as np
from pydantic import BaseModel, model_validator, HttpUrl, Field
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from rouge_score import rouge_scorer
from bert_score import score as bert_score


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
        print(f"Raw dataset loaded with {len(raw_dataset)} examples")

        # Limit size if needed
        if data_size is not None:
            raw_dataset = raw_dataset.select(
                range(min(data_size, len(raw_dataset))))
            print(f"Selected {len(raw_dataset)} examples from the dataset")

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
        
        # Basic validation check
        if len(validated_data) == 0:
            print("ERROR: No valid items found in the dataset!")
        elif len(validated_data) < len(raw_dataset):
            print(f"WARNING: Only {len(validated_data)}/{len(raw_dataset)} items passed validation")
            
        return validated_data

    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Detailed traceback:")
        import traceback
        traceback.print_exc()
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
        
        # Verify that the dataset has been loaded properly
        if len(self.annotations) == 0:
            print("WARNING: No examples loaded. Check your dataset path and settings.")
        else:
            print(f"First example ID: {self.annotations[0].instance_id}")
            print(f"First example question: {self.annotations[0].question}")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        try:
            qa_data = self.annotations[idx]

            # Get image path
            image_file_name = qa_data.image_file
            if isinstance(image_file_name, str):
                image_file_name = image_file_name.split(
                    '/')[-1]  # Get just the filename

            # Handle different possible image paths
            possible_image_paths = [
                self.image_folder / image_file_name,
                self.image_folder / "images_validation" / image_file_name,
                self.data_dir / image_file_name,
                self.data_dir / "images" / image_file_name
            ]
            
            image_path = None
            for path in possible_image_paths:
                if os.path.exists(path):
                    image_path = path
                    break
                    
            if image_path is None:
                print(f"WARNING: Image file not found for {image_file_name}. Tried paths: {possible_image_paths}")

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
            print(f"Error getting item {idx}: {str(e)}")
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
        """Get a batch of examples starting from start_idx with size batch_size"""
        end_idx = min(start_idx + batch_size, len(self))
        batch = []
        for i in range(start_idx, end_idx):
            item = self[i]
            # Only add valid items to the batch
            if item['question'] and (item['answer'] or item['choices']):
                batch.append(item)
            else:
                print(f"Skipping invalid item at index {i}")
        return batch


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


def create_prompt(example, instruction="Answer the question based on the information in the image, caption. Do not hallucinate or infer information from general knowledge."):
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
                f"{instruction} Based on the reasoning above, match it to one or more of the provided answer options: {choices}. "
                "Return only the corresponding letter(s) of the correct answer(s). "
                "Do not explain your choice, do not rephrase the answer, and do not repeat the option text. "
                "Only output the letter(s) corresponding to the correct choice. "
                "If multiple letters are correct, separate them by commas without spaces (for example: B,C). "
                "If all options are correct, return A,B,C,D. "
                "Do not add anything else.")
        elif "binary" in qa_pair_type:
            prompt_parts.append(
                f"{instruction} Return either 'Yes' or 'No'. Do not add anything else - not even punctuation marks.")
        else:
            prompt_parts.append(
                f"{instruction} Give the exact correct answer, with no extra explanation.")
    else:
        prompt_parts.append(
            f"{instruction} Give the exact correct answer, with no extra explanation.")

    return "\n".join(prompt_parts)


def rouge(predictions, references, rouge_type="rouge1"):
    """Calculate ROUGE scores"""
    scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
    scores = []
    
    # Calculate scores for each prediction-reference pair
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        scores.append(score[rouge_type])
    
    # Calculate average scores
    precision = np.mean([score.precision for score in scores])
    recall = np.mean([score.recall for score in scores])
    f1 = np.mean([score.fmeasure for score in scores])
    
    return f1, precision, recall


def bertS(predictions, references):
    """Calculate BERTScore"""
    try:
        # Calculate BERTScore
        P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
        
        # Convert from torch tensors to float
        precision = float(P.mean())
        recall = float(R.mean())
        f1 = float(F1.mean())
        
        return f1, precision, recall
    except Exception as e:
        print(f"Error calculating BERTScore: {e}")
        return 0.0, 0.0, 0.0


def calculate_metrics_batch(predictions, references):
    """Calculate all metrics for a batch of predictions and references"""
    # Calculate ROUGE-1 scores
    rouge1_f1, rouge1_precision, rouge1_recall = rouge(
        predictions, references, "rouge1")
    
    # Calculate ROUGE-L scores
    rougeL_f1, rougeL_precision, rougeL_recall = rouge(
        predictions, references, "rougeL")
    
    # Calculate BERTScore
    bert_f1, bert_precision, bert_recall = bertS(
        predictions, references)
    
    # Return all metrics
    return {
        "rouge1_f1": rouge1_f1,
        "rouge1_precision": rouge1_precision,
        "rouge1_recall": rouge1_recall,
        "rougeL_f1": rougeL_f1,
        "rougeL_precision": rougeL_precision,
        "rougeL_recall": rougeL_recall,
        "bert_f1": bert_f1,
        "bert_precision": bert_precision,
        "bert_recall": bert_recall
    }


def calculate_intermediate_metrics(results_list):
    """Calculate metrics for all results processed so far"""
    if not results_list:
        return {
            'accuracy': 0.0,
            'rouge1_f1': 0.0,
            'rouge1_precision': 0.0,
            'rouge1_recall': 0.0,
            'rougeL_f1': 0.0,
            'rougeL_precision': 0.0,
            'rougeL_recall': 0.0,
            'bert_f1': 0.0,
            'bert_precision': 0.0,
            'bert_recall': 0.0
        }
    
    # Extract predictions and ground truth
    predictions = [r['generated'].strip().lower() for r in results_list]
    references = [r['answer'].strip().lower() for r in results_list]
    
    # Calculate accuracy
    correct = sum(1 for r in results_list if r['correct'])
    accuracy = correct / len(results_list) if results_list else 0
    
    # Calculate all metrics
    metrics = calculate_metrics_batch(predictions, references)
    metrics['accuracy'] = accuracy
    
    return metrics


def log_example_to_wandb(example, response, is_correct):
    """Log individual example results to W&B"""
    # Extract relevant data from example and response
    question = example['question']
    answer = example['answer']
    generated = response['choices'][0]['message']['content'].strip() if response and 'choices' in response else "No response"
    
    # Create a wandb Table for this example
    example_table = wandb.Table(columns=["Question", "Ground Truth", "Prediction", "Correct"])
    example_table.add_data(question, answer, generated, is_correct)
    
    # Calculate metrics for this individual example
    metrics = calculate_metrics_batch([generated], [answer])
    
    # Log to wandb
    wandb.log({
        "examples": example_table,
        "correct": 1 if is_correct else 0,
        "qa_type": example['qa_pair_type'],
        "figure_type": example['figure_type'],
        # Log individual metrics
        "rouge1_f1": metrics["rouge1_f1"],
        "rouge1_precision": metrics["rouge1_precision"],
        "rouge1_recall": metrics["rouge1_recall"],
        "rougeL_f1": metrics["rougeL_f1"],
        "rougeL_precision": metrics["rougeL_precision"],
        "rougeL_recall": metrics["rougeL_recall"],
        "bert_f1": metrics["bert_f1"],
        "bert_precision": metrics["bert_precision"],
        "bert_recall": metrics["bert_recall"]
    })
    
    # If image is available, log it
    if example['image_path'] and os.path.exists(example['image_path']):
        try:
            wandb.log({"example_image": wandb.Image(
                example['image_path'],
                caption=f"Q: {question[:50]}... | A: {answer[:50]}... | Pred: {generated[:50]}...")})
        except Exception as e:
            print(f"Error logging image: {e}")


def main():
    time = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(
        description="Run inference with Phi-4-MM on SciVQA dataset using vLLM")
    parser.add_argument("--vllm_url", type=str,
                        default="http://localhost:8000/v1/chat/completions", help="URL for vLLM API server")
    parser.add_argument("--output_file", type=str,
                        default=f"{time}_phi4mm_validation_scivqa_results.json", help="Output JSON file path")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for processing")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Maximum number of examples to process")
    parser.add_argument("--split", type=str, default="validation",
                        choices=["train", "validation"], help="Dataset split to use")
    parser.add_argument("--data_dir", type=str, default="./scivqa_data",
                        help="Directory to store dataset files")
    parser.add_argument("--verify_dataset", action="store_true",
                        help="Run a verification check on dataset before processing")
    parser.add_argument("--cache_dataset", action="store_true",
                        help="Cache the dataset locally to avoid re-downloading")
    
    # Add evaluation metrics arguments
    parser.add_argument("--metrics", type=str, default="all",
                        choices=["all", "accuracy", "rouge", "bert", "bleu"], 
                        help="Which metrics to calculate")
    parser.add_argument("--rouge_types", type=str, default="rouge1,rougeL",
                        help="Comma-separated list of ROUGE types to calculate")
    
    # Add wandb-specific arguments
    parser.add_argument("--wandb_project", type=str, default="scivqa-phi4mm",
                        help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity name (username or team name)")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name (if not provided, will be auto-generated)")
    parser.add_argument("--wandb_tags", type=str, default=None,
                        help="Comma-separated list of tags for the W&B run")
    parser.add_argument("--wandb_api_key", type=str, default=None,
                        help="W&B API key for authentication")
    parser.add_argument("--wandb_mode", type=str, default="online",
                        choices=["online", "offline", "dryrun", "disabled"],
                        help="W&B logging mode (online, offline, dryrun, disabled)")
    parser.add_argument("--wandb_dir", type=str, default=None,
                        help="Directory where W&B files will be stored")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable W&B logging")
    
    args = parser.parse_args()

    # Set environment variables if provided
    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
    
    # Set wandb mode if provided
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode
        
    # Initialize W&B
    if not args.no_wandb:
        wandb_tags = args.wandb_tags.split(",") if args.wandb_tags else []
        wandb_run_name = args.wandb_run_name or f"phi4mm-scivqa-{time}"
        
        # Parse ROUGE types
        rouge_types = args.rouge_types.split(',') if args.rouge_types else ["rouge1", "rougeL"]
        
        # Configure wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=wandb_run_name,
            tags=wandb_tags,
            dir=args.wandb_dir,  # Set custom directory for wandb files
            config={
                "model": "microsoft/Phi-4-multimodal-instruct",
                "dataset": "katebor/SciVQA",
                "split": args.split,
                "max_examples": args.max_examples,
                "batch_size": args.batch_size,
                "temperature": 0.0,
                "max_tokens": 150,
                "vllm_url": args.vllm_url,
                "metrics": args.metrics,
                "rouge_types": rouge_types
            }
        )
        
        # Display wandb login status and URL
        if args.wandb_mode != "disabled" and args.wandb_mode != "offline":
            print(f"W&B initialized! View run at {wandb.run.get_url()}")
            
        # Define metric tracking for progress
        wandb.define_metric("batch_idx")
        wandb.define_metric("example_idx")
        wandb.define_metric("accuracy", step_metric="example_idx")
        wandb.define_metric("batch_accuracy", step_metric="batch_idx")
        wandb.define_metric("rouge1_f1", step_metric="example_idx")
        wandb.define_metric("rougeL_f1", step_metric="example_idx")
        wandb.define_metric("bert_f1", step_metric="example_idx")

        # Create a Table for question types
        question_types_table = wandb.Table(columns=["Type", "Correct", "Total", "Accuracy"])
        
        # Create a Table for figure types
        figure_types_table = wandb.Table(columns=["Type", "Correct", "Total", "Accuracy"])
        
        # Create confusion matrix data
        if not args.no_wandb:
            # Create a simple confusion matrix with empty data for now
            # This will be updated as we process examples
            wandb.log({
                "confusion_matrix": wandb.plot.confusion_matrix(
                    y_true=[],  # Will be populated as we go
                    preds=[],
                    class_names=["incorrect", "correct"]
                )
            })

    # Load dataset
    dataset = SciVQAInferenceDataset(
        data_size=args.max_examples,
        split=args.split,
        data_dir=args.data_dir
    )
    
    # Verify dataset has valid examples
    if len(dataset) == 0:
        print("ERROR: No valid examples found in the dataset. Exiting.")
        if not args.no_wandb:
            wandb.finish()
        return
        
    # Optional dataset verification
    if args.verify_dataset:
        print("Verifying dataset...")
        valid_examples = 0
        for i in tqdm(range(min(100, len(dataset))), desc="Checking sample examples"):
            item = dataset[i]
            if item['question'] and (item['answer'] or item['choices']):
                valid_examples += 1
                
        print(f"Dataset verification: {valid_examples}/100 examples are valid")
        if valid_examples < 50:
            print("WARNING: Less than 50% of examples are valid. Consider checking your dataset.")
            
    # Results storage
    results = []
    
    # Track metrics for W&B
    running_correct = 0
    running_total = 0

    # Process dataset in batches
    n_batches = (len(dataset) + args.batch_size - 1) // args.batch_size
    for batch_idx in tqdm(range(n_batches), desc="Processing batches"):
        start_idx = batch_idx * args.batch_size
        batch = dataset.get_batch(start_idx, args.batch_size)
        
        # Check if batch has valid items
        if len(batch) == 0:
            print(f"WARNING: Batch {batch_idx+1}/{n_batches} contains no valid items. Skipping.")
            continue
            
        print(f"Processing batch {batch_idx+1}/{n_batches} with {len(batch)} examples")
        batch_correct = 0
        batch_total = 0

        for example in tqdm(batch, desc=f"Batch {batch_idx+1}/{n_batches}", leave=False):
            # Skip items without valid questions or answers
            if not example['question'] or (not example['answer'] and not example['choices']):
                print(f"Skipping example with ID {example['id']} due to missing question or answer")
                continue
                
            # Prepare data
            prompt = create_prompt(example)

            # Run inference
            try:
                print(f"\n{'='*80}")
                print(f"EXAMPLE {running_total+1}")
                print(f"{'='*80}")
                print(f"QUESTION: {example['question']}")
                print(f"GROUND TRUTH: {example['answer']}")
                print(f"PROMPT: {prompt[:200]}...")
                
                image_path = example['image_path']
                response = vllm_inference(prompt, image_path, args.vllm_url)
                
                # Process response
                if response and 'choices' in response and len(response['choices']) > 0:
                    generated_text = response['choices'][0]['message']['content'].strip()
                    print(f"PREDICTION: {generated_text}")
                    
                    # Calculate accuracy (exact match)
                    ground_truth = example['answer'].strip().lower()
                    generated_clean = generated_text.strip().lower()
                    is_correct = generated_clean == ground_truth
                    print(f"EXACT MATCH: {'✅' if is_correct else '❌'}")
                    
                    # Update running metrics
                    running_total += 1
                    batch_total += 1
                    if is_correct:
                        running_correct += 1
                        batch_correct += 1
                    
                    # Calculate running accuracy
                    running_accuracy = running_correct / running_total if running_total > 0 else 0

                    # Calculate metrics for this individual example
                    individual_metrics = calculate_metrics_batch([generated_clean], [ground_truth])
                    
                    # Print detailed metrics for this example
                    print(f"\nExample {running_total} Metrics:")
                    print(f"Question: {example['question']}")
                    print(f"Ground Truth: {ground_truth}")
                    print(f"Generated: {generated_clean}")
                    print(f"Exact Match: {is_correct}")
                    print(f"ROUGE-1 F1: {individual_metrics['rouge1_f1']:.4f}")
                    print(f"ROUGE-1 Precision: {individual_metrics['rouge1_precision']:.4f}")
                    print(f"ROUGE-1 Recall: {individual_metrics['rouge1_recall']:.4f}")
                    print(f"ROUGE-L F1: {individual_metrics['rougeL_f1']:.4f}")
                    print(f"ROUGE-L Precision: {individual_metrics['rougeL_precision']:.4f}")
                    print(f"ROUGE-L Recall: {individual_metrics['rougeL_recall']:.4f}")
                    print(f"BERTScore F1: {individual_metrics['bert_f1']:.4f}")
                    print(f"BERTScore Precision: {individual_metrics['bert_precision']:.4f}")
                    print(f"BERTScore Recall: {individual_metrics['bert_recall']:.4f}")
                    
                    # Store result with metrics
                    result = {
                        'id': example['id'],
                        'question': example['question'],
                        'question_type': example['qa_pair_type'],
                        'figure_type': example['figure_type'],
                        'generated': generated_text,
                        'answer': example['answer'],
                        'correct': is_correct,
                        # Add detailed metrics
                        'rouge1_f1': individual_metrics['rouge1_f1'],
                        'rouge1_precision': individual_metrics['rouge1_precision'],
                        'rouge1_recall': individual_metrics['rouge1_recall'],
                        'rougeL_f1': individual_metrics['rougeL_f1'],
                        'rougeL_precision': individual_metrics['rougeL_precision'],
                        'rougeL_recall': individual_metrics['rougeL_recall'],
                        'bert_f1': individual_metrics['bert_f1'],
                        'bert_precision': individual_metrics['bert_precision'],
                        'bert_recall': individual_metrics['bert_recall'],
                    }
                    results.append(result)

                    # Log to W&B if enabled
                    if not args.no_wandb:
                        # Log this example
                        log_example_to_wandb(example, response, is_correct)
                        
                        # Log running metrics
                        wandb.log({
                            "confusion_matrix": wandb.plot.confusion_matrix(
                                y_true=all_gt,
                                preds=all_preds,
                                class_names=["incorrect", "correct"]
                            )
                        })

                    # Print progress
                    if (running_total % 1 == 0) or running_total == 1:
                        print(
                            f"\nProcessed {running_total}/{len(dataset)} examples")
                        print(f"Question: {example['question']}")
                        print(f"Ground Truth: {ground_truth}")
                        print(f"Generated: {generated_clean}")
                        print(f"Exact Match: {is_correct}")
                        print(f"Current Accuracy: {running_accuracy:.4f}")
                        
                        # Print current metrics
                        print("\nCurrent Running Metrics:")
                        print(f"ROUGE-1 F1: {individual_metrics['rouge1_f1']:.4f}")
                        print(f"ROUGE-L F1: {individual_metrics['rougeL_f1']:.4f}")
                        print(f"BERTScore F1: {individual_metrics['bert_f1']:.4f}")
                        
                    # Save intermediate results
                    if running_total % 5 == 0:
                        # Calculate metrics so far
                        current_metrics = calculate_intermediate_metrics(results)
                        
                        # Save results with current metrics
                        with open(args.output_file, 'w') as f:
                            json.dump({
                                'results': results,
                                'progress': f"{running_total}/{len(dataset)}",
                                'current_metrics': current_metrics
                            }, f, indent=2)
                            
                        print(f"\nIntermediate results saved to {args.output_file} with current metrics")
                        print(f"Current ROUGE-1 F1: {current_metrics['rouge1_f1']:.4f}")
                        print(f"Current ROUGE-L F1: {current_metrics['rougeL_f1']:.4f}")
                        print(f"Current BERTScore F1: {current_metrics['bert_f1']:.4f}")
                else:
                    print(f"Failed to get response for example {example['id']}")
            except Exception as e:
                print(f"Error processing example: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        # Log batch metrics
        batch_accuracy = batch_correct / batch_total if batch_total > 0 else 0
        if not args.no_wandb:
            wandb.log({
                "batch_idx": batch_idx,
                "batch_accuracy": batch_accuracy,
                "batch_correct": batch_correct,
                "batch_total": batch_total
            })
        
        print(f"Batch {batch_idx+1}/{n_batches} complete. Accuracy: {batch_accuracy:.4f} ({batch_correct}/{batch_total})")

    # Calculate metrics and save final results
    correct = sum(1 for r in results if r['correct'])
    total = len(results)
    accuracy = correct / total if total > 0 else 0
    
    # Calculate ROUGE and BERTScore metrics
    all_predictions = [r['generated'] for r in results]
    all_references = [r['answer'] for r in results]
    
    # Calculate all metrics
    metrics = calculate_metrics_batch(all_predictions, all_references)
    
    # Print metrics
    print("\n--- NLG Metrics ---")
    print(f"ROUGE-1 F1: {metrics['rouge1_f1']:.4f}")
    print(f"ROUGE-1 Precision: {metrics['rouge1_precision']:.4f}")
    print(f"ROUGE-1 Recall: {metrics['rouge1_recall']:.4f}")
    print(f"ROUGE-L F1: {metrics['rougeL_f1']:.4f}")
    print(f"ROUGE-L Precision: {metrics['rougeL_precision']:.4f}")
    print(f"ROUGE-L Recall: {metrics['rougeL_recall']:.4f}")
    print(f"BERTScore F1: {metrics['bert_f1']:.4f}")
    print(f"BERTScore Precision: {metrics['bert_precision']:.4f}")
    print(f"BERTScore Recall: {metrics['bert_recall']:.4f}")

    # Calculate metrics by question type
    type_metrics = {}
    for result in results:
        q_type = result['question_type']
        if q_type not in type_metrics:
            type_metrics[q_type] = {
                'correct': 0, 
                'total': 0, 
                'predictions': [], 
                'references': []
            }

        type_metrics[q_type]['total'] += 1
        if result['correct']:
            type_metrics[q_type]['correct'] += 1
            
        # Add prediction and reference for NLG metrics calculation
        type_metrics[q_type]['predictions'].append(result['generated'])
        type_metrics[q_type]['references'].append(result['answer'])

    # Calculate figure type metrics
    figure_metrics = {}
    for result in results:
        f_type = result['figure_type']
        if f_type not in figure_metrics:
            figure_metrics[f_type] = {
                'correct': 0, 
                'total': 0, 
                'predictions': [], 
                'references': []
            }

        figure_metrics[f_type]['total'] += 1
        if result['correct']:
            figure_metrics[f_type]['correct'] += 1
            
        # Add prediction and reference for NLG metrics calculation
        figure_metrics[f_type]['predictions'].append(result['generated'])
        figure_metrics[f_type]['references'].append(result['answer'])

    # Compute accuracies and NLG metrics for each category
    for category_metrics in [type_metrics, figure_metrics]:
        for key, values in category_metrics.items():
            values['accuracy'] = values['correct'] / \
                values['total'] if values['total'] > 0 else 0
                
            # Calculate NLG metrics for this category
            category_nlg_metrics = calculate_metrics_batch(
                values['predictions'], 
                values['references']
            )
            
            # Add NLG metrics to the values dictionary
            for metric_name, metric_value in category_nlg_metrics.items():
                values[metric_name] = metric_value
                
            # Remove the large prediction and reference lists to save memory
            values.pop('predictions')
            values.pop('references')

    # Final results
    final_results = {
        'overall': {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            # Add NLG metrics
            'rouge1_f1': metrics['rouge1_f1'],
            'rouge1_precision': metrics['rouge1_precision'],
            'rouge1_recall': metrics['rouge1_recall'],
            'rougeL_f1': metrics['rougeL_f1'],
            'rougeL_precision': metrics['rougeL_precision'],
            'rougeL_recall': metrics['rougeL_recall'],
            'bert_f1': metrics['bert_f1'],
            'bert_precision': metrics['bert_precision'],
            'bert_recall': metrics['bert_recall']
        },
        'by_question_type': type_metrics,
        'by_figure_type': figure_metrics,
        'results': results
    }

    # Save final results
    with open(args.output_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    # Log final metrics to W&B
    if not args.no_wandb:
        # Log overall metrics
        wandb.log({
            "final_accuracy": accuracy,
            "total_correct": correct,
            "total_examples": total,
            # Log NLG metrics
            "final_rouge1_f1": metrics['rouge1_f1'],
            "final_rouge1_precision": metrics['rouge1_precision'],
            "final_rouge1_recall": metrics['rouge1_recall'],
            "final_rougeL_f1": metrics['rougeL_f1'],
            "final_rougeL_precision": metrics['rougeL_precision'],
            "final_rougeL_recall": metrics['rougeL_recall'],
            "final_bert_f1": metrics['bert_f1'],
            "final_bert_precision": metrics['bert_precision'],
            "final_bert_recall": metrics['bert_recall']
        })
        
        # Log question type metrics to table
        for q_type, metrics in type_metrics.items():
            question_types_table.add_data(
                q_type, 
                metrics['correct'], 
                metrics['total'], 
                metrics['accuracy']
            )
        
        # Log figure type metrics to table
        for f_type, metrics in figure_metrics.items():
            figure_types_table.add_data(
                f_type, 
                metrics['correct'], 
                metrics['total'], 
                metrics['accuracy']
            )
        
        # Log the tables
        wandb.log({
            "question_types_summary": question_types_table,
            "figure_types_summary": figure_types_table
        })
        
        # Create visualizations for all metrics by question type
        metrics_to_plot = [
            ("accuracy", "Accuracy"), 
            ("rouge1_f1", "ROUGE-1 F1"),
            ("rougeL_f1", "ROUGE-L F1"),
            ("bert_f1", "BERTScore F1")
        ]
        
        for metric_key, metric_name in metrics_to_plot:
            q_type_data = [[q_type, metrics[metric_key]] for q_type, metrics in type_metrics.items()]
            wandb.log({
                f"question_type_{metric_key}": wandb.plot.bar(
                    wandb.Table(data=q_type_data, columns=["Question Type", metric_name]),
                    "Question Type",
                    metric_name,
                    title=f"{metric_name} by Question Type"
                )
            })
        
        # Create visualizations for all metrics by figure type
        for metric_key, metric_name in metrics_to_plot:
            f_type_data = [[f_type, metrics[metric_key]] for f_type, metrics in figure_metrics.items()]
            wandb.log({
                f"figure_type_{metric_key}": wandb.plot.bar(
                    wandb.Table(data=f_type_data, columns=["Figure Type", metric_name]),
                    "Figure Type",
                    metric_name,
                    title=f"{metric_name} by Figure Type"
                )
            })
            
        # Create a comprehensive metrics table
        metrics_summary = wandb.Table(
            columns=["Metric", "Overall", "Best Question Type", "Best Figure Type"]
        )
        
        # Helper function to find best category
        def find_best_category(metrics_dict, metric_key):
            best_category = max(metrics_dict.items(), key=lambda x: x[1][metric_key])
            return f"{best_category[0]} ({best_category[1][metric_key]:.4f})"
        
        # Add rows for each metric
        for metric_key, metric_name in metrics_to_plot:
            best_q_type = find_best_category(type_metrics, metric_key)
            best_f_type = find_best_category(figure_metrics, metric_key)
            
            metrics_summary.add_data(
                metric_name,
                f"{final_results['overall'][metric_key]:.4f}",
                best_q_type,
                best_f_type
            )
        
        # Log the comprehensive metrics table
        wandb.log({"metrics_summary": metrics_summary})
        
        # Log the full results JSON as an artifact
        results_artifact = wandb.Artifact(
            name=f"results-{wandb.run.id}", 
            type="evaluation_results",
            description="Full evaluation results with all examples"
        )
        with open(args.output_file, "rb") as f:
            results_artifact.add_file(f, name="results.json")
        
        wandb.log_artifact(results_artifact)
        
        # Finish the wandb run
        wandb.finish()

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
                       
                        
